import gc
import types
import csv
import torch
import numpy as np
import time
import deepspeed
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score

from header import *
from datasets import *
from model import *
from config import *
from metrics import cal_pro_score
from model.openllama import encode_text_with_prompt_ensemble

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--model', type=str)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--log_path', type=str)
    # model configurations
    parser.add_argument('--imagebind_ckpt_path', type=str) # the path that stores the imagebind checkpoint
    parser.add_argument('--stage', type=int) # the maximum sequence length
    parser.add_argument('--image_root_path', type=str) # the maximum sequence length
    parser.add_argument('--epochs', type=int, default=None)  # override yaml epochs
    parser.add_argument('--lambda_align', type=float, default=None)  # Drift separation loss 权重
    parser.add_argument('--lambda_gate', type=float, default=None)   # Gate supervision loss 权重
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    return parser.parse_args()

def initialize_distributed(args):
    args['master_ip'] = os.getenv('MASTER_ADDR', 'localhost')
    args['master_port'] = os.getenv('MASTER_PORT', '6000')
    args['world_size'] = int(os.getenv('WORLD_SIZE', '1'))
    args['local_rank'] = int(os.getenv('RANK', '0')) % torch.cuda.device_count()
    device = args['local_rank'] % torch.cuda.device_count()
    torch.cuda.set_device(device)
    deepspeed.init_distributed(dist_backend='nccl')

def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def config_env(args):
    args['root_dir'] = '../'
    args['mode'] = 'train'
    config = load_config(args)
    args.update(config)
    initialize_distributed(args)
    set_random_seed(args['seed'])

def build_directory(path):
    if os.path.exists(path):
        pass
    else: # recursively construct directory
        os.makedirs(path, exist_ok=True)


def get_eval_size():
    return 224


def evaluate_visa_classes(model, classes_to_test, epoch_i, k_shot=1, round_idx=14):
    """
    Visa Evaluation for specific classes (e.g. pcb2, pcb3)
    """
    root_dir = os.environ.get('VISA_ROOT', '../data/visa')
    datas_csv_path = os.path.join(root_dir, 'split_csv/1cls.csv')
    describles = {
        'pcb2': 'pcb',
        'pcb3': 'pcb',
    }
    
    eval_size = get_eval_size()
    mask_transform = transforms.Compose([
        transforms.Resize(eval_size),
        transforms.CenterCrop(eval_size),
        transforms.ToTensor()
    ])
    
    # 1. Prepare Data Paths
    file_paths = {c: [] for c in classes_to_test}
    normal_img_path = {c: [] for c in classes_to_test}
    
    if not os.path.exists(datas_csv_path):
        print(f"Error: CSV not found at {datas_csv_path}")
        return

    with open(datas_csv_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            c_name = row[0]
            if c_name not in classes_to_test:
                continue
            
            # Test Images
            if row[1] == 'test':
                file_paths[c_name].append(os.path.join(root_dir, row[3]))
            
            # Training Images (for support set)
            if row[1] == 'train':
                # VisA few-shot logic: pick specific indices based on 'round'
                if len(normal_img_path[c_name]) < round_idx * 4 + k_shot:
                    normal_img_path[c_name].append(os.path.join(root_dir, row[3]))

    # Filter Support Set
    for c in classes_to_test:
        if len(normal_img_path[c]) >= round_idx * 4:
            normal_img_path[c] = normal_img_path[c][round_idx * 4:]
        else:
            # Fallback if specific round indices not available
            normal_img_path[c] = normal_img_path[c][-k_shot:] if normal_img_path[c] else []

    # 2. Precompute Caches
    cached_text_feat = {}
    cached_normal_feats = {}
    
    for c_name in classes_to_test:
        prompt = describles.get(c_name, c_name)
        cached_text_feat[c_name] = model._encode_text([prompt], use_disk_cache=True)

        cached_normal_feats[c_name] = None
        if normal_img_path[c_name]:
            cached_normal_feats[c_name], _ = model.encode_image_for_one_shot(
                normal_img_path[c_name],
                use_cache=True,
                dataset_name='visa'
            )

    # 3. Inference Loop
    print(f"\n--- Evaluating Epoch {epoch_i} for {classes_to_test} ---")
    logging.info(f"\n--- Evaluating Epoch {epoch_i} for {classes_to_test} ---")
    r = 0.1
    
    for c_name in classes_to_test:
        p_pred, p_label, i_pred, i_label = [], [], [], []
        count = 0
        start_time = time.time()
        
        print(f"Processing Class: {c_name} ...")
        logging.info(f"Processing Class: {c_name} ...")
        
        if not file_paths[c_name]:
            print(f"  No test files found for {c_name}")
            logging.info(f"  No test files found for {c_name}")
            continue

        for file_path in file_paths[c_name]:
            count += 1
            
            # GT Mask Preparation
            is_normal = 'Normal' in file_path.split('/')[-2]
            if is_normal:
                img_mask = np.zeros((eval_size, eval_size))
            else:
                mask_path = file_path.replace('Images', 'Masks').replace('.JPG', '.png')
                if os.path.exists(mask_path):
                    img_mask = Image.open(mask_path).convert('L')
                    img_mask = mask_transform(img_mask)
                    threshold = img_mask.max() / 100
                    img_mask[img_mask > threshold] = 1
                    img_mask[img_mask <= threshold] = 0
                    img_mask = img_mask.squeeze().reshape(eval_size, eval_size).cpu().numpy()
                else:
                    img_mask = np.zeros((eval_size, eval_size))

            # Inference
            with torch.no_grad():
                pixel_output, cls_score = model.generate({
                    'prompt': describles.get(c_name, c_name),
                    'image_paths': [file_path],
                    'normal_img_paths': [],
                    'cached_text_features': cached_text_feat[c_name],
                    'cached_normal_patch_tokens': cached_normal_feats[c_name],
                    'r': r
                })

            anomaly_map = pixel_output.reshape(eval_size, eval_size).detach().cpu().numpy()
            
            p_label.append(img_mask)
            p_pred.append(anomaly_map)
            i_label.append(0 if is_normal else 1)
            
            # Image Score
            flat_matrix = anomaly_map.ravel()
            k_top = 30
            if len(flat_matrix) > k_top:
                score1 = np.mean(np.sort(flat_matrix)[-k_top:])
            else:
                score1 = np.max(flat_matrix)
            
            i_pred.append(r * cls_score.cpu().detach().numpy() + (1-r) * score1)

        # Metrics
        if len(p_label) > 0:
            p_label = np.array(p_label)
            p_pred = np.array(p_pred)
            i_label = np.array(i_label)
            i_pred = np.array(i_pred)

            p_auroc = round(roc_auc_score(p_label.ravel(), p_pred.ravel()) * 100, 2)
            i_auroc = round(roc_auc_score(i_label.ravel(), i_pred.ravel()) * 100, 2)
            p_pro = round(cal_pro_score(p_label, p_pred) * 100, 2)
            ap = round(average_precision_score(i_label, i_pred) * 100, 2)
            
            cost_time = time.time() - start_time
            print(f"  Finished {count} images in {cost_time:.2f}s")
            print(f"  {c_name} -> i_AUC: {i_auroc}, p_AUC: {p_auroc}, PRO: {p_pro}, AP: {ap}")
            logging.info(f"  Finished {count} images in {cost_time:.2f}s")
            logging.info(f"  {c_name} -> i_AUC: {i_auroc}, p_AUC: {p_auroc}, PRO: {p_pro}, AP: {ap}")
        else:
            print(f"  [Warning] No test images found (len(p_label)=0) for {c_name}")
            logging.warning(f"  [Warning] No test images found (len(p_label)=0) for {c_name}")


def main(**args):
    epochs_cli = args.get("epochs")   # 保存 CLI 传入的值（config_env 会覆盖）
    lambda_align_cli = args.get("lambda_align")
    lambda_gate_cli  = args.get("lambda_gate")
    config_env(args)
    if epochs_cli is not None:        # CLI 优先级高于 yaml
        args["epochs"] = epochs_cli
    if lambda_align_cli is not None:
        args["lambda_align"] = lambda_align_cli
    if lambda_gate_cli is not None:
        args["lambda_gate"] = lambda_gate_cli
    args['ds_config_path'] = f'dsconfig/{args["model"]}_stage_{args["stage"]}.json'
    dschf = HfDeepSpeedConfig(args['ds_config_path'])
    args['dschf'] = dschf

    # 设置数据集名称，用于缓存区分
    args['dataset_name'] = 'visa'

    build_directory(args['save_path'])
    build_directory(args['log_path'])

    if args['log_path']:
        logging.basicConfig(
            format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s', 
            level=logging.DEBUG,
            filename=f'{args["log_path"]}/train_{time.asctime()}.log',
            filemode='w'
        )
        # Suppress massive PIL/PngImagePlugin debug logs
        logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
        logging.getLogger("PIL.Image").setLevel(logging.WARNING)
    
    train_data, train_iter, sampler = load_visa_dataset(args)

    length = args['epochs'] * len(train_data) // args['world_size'] // dschf.config['train_micro_batch_size_per_gpu']
    total_steps = 2 * args['epochs'] * len(train_data) // dschf.config['train_batch_size']
    args['total_steps'] = total_steps
    agent = load_model(args)
    torch.distributed.barrier()

    # begin to train
    pbar = tqdm(total=length)    # maximum total number
    current_step = 0
    for epoch_i in tqdm(range(args['epochs'])):
        iter_every_epoch = 0
        for batch in train_iter:
            iter_every_epoch += 1
            agent.train_model(
                batch, 
                current_step=current_step, 
                pbar=pbar
            )
            del batch

            current_step += 1
            gc.collect()
            # torch.cuda.empty_cache()
            if iter_every_epoch % 1000 == 0:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    agent.save_model(args['save_path'], 0)
        # save at the end of the training
        torch.distributed.barrier()
        if not dist.is_initialized() or dist.get_rank() == 0:
            agent.save_model(args['save_path'], epoch_i, total_epochs=args['epochs'])
        
        # # Evaluate visa subset
        # if args['local_rank'] == 0 and epoch_i > 50:
        #     agent.ds_engine.module.eval()
        #     evaluate_visa_classes(agent.ds_engine.module, ['pcb2', 'pcb3'], epoch_i)
        #     agent.ds_engine.module.train()

if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    args['layers'] = [7,15,23,31]
    main(**args)
