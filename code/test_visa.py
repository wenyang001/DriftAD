import os
from model.openllama import OpenLLAMAPEFTModel
import torch
from torchvision import transforms
from sklearn.metrics import roc_auc_score,average_precision_score
from PIL import Image
import numpy as np
import csv
import argparse
import glob
from tqdm import tqdm
from metrics import cal_pro_score
# from visualization import visualizer


parser = argparse.ArgumentParser("AnomalyGPT", add_help=True)
parser.add_argument("--few_shot", type=bool, default=True)
parser.add_argument("--k_shot", type=int, default=1)
parser.add_argument("--round", type=int, default=14) # 1-shot:14, 2-shot:57, 4-shot:78
parser.add_argument("--ckpt", type=str, default=None, help="override anomalygpt_ckpt_path")
parser.add_argument("--support-mode", type=str, default='round', choices=['round', 'promptad_seed'])
parser.add_argument("--promptad-seed-dir", type=str,
                    default=os.environ.get('PROMPTAD_SEED_DIR', '../data/seeds_visa'))
parser.add_argument("--visa-pytorch-root", type=str,
                    default=os.environ.get('VISA_PYTORCH_ROOT', '../data/visa_pytorch/1cls'))
parser.add_argument("--classes", type=str, default=None,
                    help="comma-separated class subset for quick diagnostics")
parser.add_argument("--r", type=float, default=0.1)
parser.add_argument("--topk", type=int, default=30)

command_args = parser.parse_args()

describles = {
    'candle': 'candle',
    'capsules': 'capsule',
    'cashew': 'cashew',
    'chewinggum': 'chewinggom',
    'fryum': 'fryum',
    'macaroni1': 'macaroni',
    'macaroni2': 'macaroni',
    'pcb1': 'pcb',
    'pcb2': 'pcb',
    'pcb3': 'pcb',
    'pcb4': 'pcb',
    'pipe_fryum': 'pipe fryum'
}


FEW_SHOT = command_args.few_shot
EVAL_SIZE = 224

# init the model
args = {
    'model': 'openllama_peft',
    'imagebind_ckpt_path': '../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth',
    'anomalygpt_ckpt_path': command_args.ckpt or './ckpt/train_visa/train_on_visa.pt',
    'stage': 2,
    'features_list': [6, 12, 18, 24],
}


model = OpenLLAMAPEFTModel(**args)
delta_ckpt = torch.load(args['anomalygpt_ckpt_path'], map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
model = model.cuda()

def predict(
        input,
        image_path,
        normal_img_path,
        r,
        cached_text_features=None,
        cached_normal_patch_tokens=None,
):
    prompt_text = input
    generate_inputs = {
        'prompt': prompt_text,
        'image_paths': [image_path] if image_path else [],
        'normal_img_paths': normal_img_path if normal_img_path else [],
        'r': r
    }
    if cached_text_features is not None:
        generate_inputs['cached_text_features'] = cached_text_features
    if cached_normal_patch_tokens is not None:
        generate_inputs['cached_normal_patch_tokens'] = cached_normal_patch_tokens

    pixel_output, cls_score = model.generate(generate_inputs)

    return pixel_output, cls_score


root_dir = os.environ.get('VISA_ROOT', '../data/visa')

mask_transform = transforms.Compose([
    transforms.Resize(EVAL_SIZE),
    transforms.CenterCrop(EVAL_SIZE),
    transforms.ToTensor()
])

datas_csv_path = os.path.join(root_dir, 'split_csv', '1cls.csv')

CLASS_NAMES = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2',
               'pcb3', 'pcb4', 'pipe_fryum']

if command_args.classes:
    requested_classes = [c.strip() for c in command_args.classes.split(',') if c.strip()]
    CLASS_NAMES = [c for c in CLASS_NAMES if c in requested_classes]


def load_promptad_support_paths(class_name, k_shot):
    seed_file = os.path.join(command_args.promptad_seed_dir, class_name, 'selected_samples_per_run.txt')
    begin_str = f'#{k_shot}: '
    selected_indices = None
    with open(seed_file, 'r') as f:
        for line in f:
            if line.startswith(begin_str):
                selected_indices = [int(item) for item in line[len(begin_str):].strip().split()]
                break
    if selected_indices is None:
        raise ValueError(f'No PromptAD seed entry found for {class_name} with k_shot={k_shot}: {seed_file}')

    train_good_dir = os.path.join(command_args.visa_pytorch_root, class_name, 'train', 'good')
    train_paths = []
    for img_path in glob.glob(os.path.join(train_good_dir, '*.JPG')):
        train_paths.append(img_path)

    return [train_paths[index] for index in selected_indices]

file_paths = {}
normal_img_path = {}

for class_name in CLASS_NAMES:
    file_paths[class_name] = []
    normal_img_path[class_name] = []

with open(datas_csv_path, 'r') as file:
    reader = csv.reader(file)

    for row in reader:
        if row[1] == 'test' and row[0] in CLASS_NAMES:
            file_paths[row[0]].append(os.path.join(root_dir, row[3]))
        if row[0] in CLASS_NAMES and len(normal_img_path[row[0]]) < command_args.round * 4 + command_args.k_shot and \
                row[1] == 'train':
            normal_img_path[row[0]].append(os.path.join(root_dir, row[3]))

if FEW_SHOT:
    for i in CLASS_NAMES:
        if command_args.support_mode == 'promptad_seed':
            normal_img_path[i] = load_promptad_support_paths(i, command_args.k_shot)
        else:
            normal_img_path[i] = normal_img_path[i][command_args.round * 4:]
            # normal_img_path[i] = random.sample(normal_img_path[i], command_args.k_shot)
        print(f'{i} support: {normal_img_path[i]}')

r = command_args.r
p_auc_list = []
i_auc_list = []
p_pro_list = []
ap_list = []

cached_text_features = {}
cached_normal_patch_tokens = {}
print('[test_visa] building per-run in-memory caches')
for c_name in CLASS_NAMES:
    cached_text_features[c_name] = model._encode_text([describles[c_name]], use_disk_cache=False)
    cached_normal_patch_tokens[c_name] = None
    if FEW_SHOT and normal_img_path[c_name]:
        cached_normal_patch_tokens[c_name], _ = model.encode_image_for_one_shot(
                normal_img_path[c_name],
                use_cache=False,
            )

for c_name in CLASS_NAMES:
    p_pred = []
    p_label = []
    i_pred = []
    i_label = []
    for file_path in tqdm(file_paths[c_name]):
        if FEW_SHOT:
            model.eval()
            with torch.no_grad():
                anomaly_map, score = predict(
                    describles[c_name],
                    file_path,
                    normal_img_path[c_name],
                    r,
                    cached_text_features=cached_text_features[c_name],
                    cached_normal_patch_tokens=cached_normal_patch_tokens[c_name],
                )
        else:
            anomaly_map, score = predict(
                describles[c_name],
                file_path,
                None,
                1,
                cached_text_features=cached_text_features[c_name],
            )

        is_normal = 'Normal' in file_path.split('/')[-2]

        if is_normal:
            img_mask = Image.fromarray(np.zeros((EVAL_SIZE, EVAL_SIZE), dtype=np.uint8), mode='L')
        else:
            mask_path = file_path.replace('Images', 'Masks')
            mask_path = mask_path.replace('.JPG', '.png')
            img_mask = Image.open(mask_path).convert('L')

        img_mask = mask_transform(img_mask)
        threshold = img_mask.max() / 100
        img_mask[img_mask > threshold], img_mask[img_mask <= threshold] = 1, 0
        img_mask = img_mask.squeeze().reshape(EVAL_SIZE, EVAL_SIZE).cpu().numpy()

        anomaly_map = anomaly_map.reshape(EVAL_SIZE, EVAL_SIZE).detach().cpu().numpy()

        # save_path = '/mnt/sda/fenfangtao/AnomalyGPT-main/code/visualize/'
        # visualizer(file_path, anomaly_map, 224, save_path, c_name,'visa')

        p_label.append(img_mask)
        p_pred.append(anomaly_map)

        i_label.append(1 if not is_normal else 0)

        k = command_args.topk
        flat_matrix = anomaly_map.ravel()
        top_k_indices = np.argpartition(-flat_matrix, k)[:k]
        top_k_values = flat_matrix[top_k_indices]
        score1 = np.mean(top_k_values)
        i_pred.append(r * score.cpu().numpy() + (1-r) * score1)
        # i_pred.append(anomaly_map.max())

    p_pred = np.array(p_pred)
    p_label = np.array(p_label)

    i_pred = np.array(i_pred)
    i_label = np.array(i_label)

    p_auroc = round(roc_auc_score(p_label.ravel(), p_pred.ravel()) * 100, 2)
    i_auroc = round(roc_auc_score(i_label.ravel(), i_pred.ravel()) * 100, 2)
    p_pro = round(cal_pro_score(p_label, p_pred) * 100, 2)
    ap = round(average_precision_score(i_label, i_pred)* 100, 2)

    p_auc_list.append(p_auroc)
    i_auc_list.append(i_auroc)
    p_pro_list.append(p_pro)
    ap_list.append(ap)

    print(c_name, "i_AUROC:", i_auroc)
    print(c_name, "p_AUROC:", p_auroc)
    print(c_name, "p_pro:", p_pro)
    print(c_name, "ap:", ap)

print("i_AUROC:", torch.tensor(i_auc_list).mean())
print("p_AUROC:", torch.tensor(p_auc_list).mean())
print("p_PRO:", torch.tensor(p_pro_list).mean())
print('ap', torch.tensor(ap_list).mean())
