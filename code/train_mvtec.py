# train_mvtec.py

from header import *
from datasets import *
from model import *
from config import *

import os
import time
import logging
import random
import gc
import json
import types    # 用于兼容 args['dschf']

from pathlib import Path
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score
from metrics import cal_pro_score
from model.openllama import encode_text_with_prompt_ensemble

import numpy as np
import torch
import torch.distributed as dist
import deepspeed
from tqdm import tqdm

# Suppress massive PIL/PngImagePlugin debug logs
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
logging.getLogger("PIL.Image").setLevel(logging.WARNING)


def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--model', type=str)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--log_path', type=str)
    # model configurations
    parser.add_argument('--imagebind_ckpt_path', type=str)   # the path that stores the imagebind checkpoint
    parser.add_argument('--stage', type=int)                 # training stage
    parser.add_argument('--image_root_path', type=str)       # root path of images
    parser.add_argument('--epochs', type=int, default=None)       # override yaml epochs
    parser.add_argument('--lambda_align', type=float, default=None)  # Drift-Visual Alignment Loss 权重
    parser.add_argument('--lambda_gate', type=float, default=None)   # Gate supervision loss 权重
    parser.add_argument(
        "--features_list",
        type=int,
        nargs="+",
        default=[6, 12, 18, 24],
        help="features used"
    )
    return parser.parse_args()


def initialize_distributed(args: dict):
    """
    只在这里做一次全局分布式初始化。
    之后所有地方都假设 dist 已经 init 完成。
    """
    # 已经初始化过就不要重复 init
    if dist.is_available() and dist.is_initialized():
        args["world_size"] = dist.get_world_size()
        args["global_rank"] = dist.get_rank()
        args["local_rank"] = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(args["local_rank"])
        return

    # deepspeed launcher / torchrun 一般会设置 LOCAL_RANK
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    torch.cuda.set_device(local_rank)

    # 用 DeepSpeed 帮我们 init nccl backend
    deepspeed.init_distributed(dist_backend="nccl")

    world_size = dist.get_world_size()
    global_rank = dist.get_rank()

    args["world_size"] = world_size
    args["global_rank"] = global_rank
    args["local_rank"] = local_rank


def set_random_seed(seed: int):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def config_env(args: dict):
    """
    加载配置 + 初始化分布式 + 设置随机种子
    """
    args["root_dir"] = "../"
    args["mode"] = "train"

    cfg = load_config(args)   # 你原来的配置加载函数
    args.update(cfg)

    # ★ 只在这里 init_distributed 一次
    initialize_distributed(args)

    # 设置随机种子
    set_random_seed(args["seed"])


def build_directory(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def evaluate_screw(model, root_dir, epoch_i, k_shot=1, round_idx=195):
    # Force use of correct dataset path
    root_dir = os.environ.get('MVTEC_ROOT', '../data/mvtec')
    
    # 测试多个类别
    class_list = [
        ('screw', 'screw'),
        ('cable', 'cable'),
    ]
    
    for c_name, describle in class_list:
        evaluate_single_class(model, root_dir, epoch_i, c_name, describle, k_shot, round_idx)


def get_eval_size():
    return 224


def evaluate_single_class(model, root_dir, epoch_i, c_name, describle, k_shot=1, round_idx=195):
    """评估单个类别"""
    r = 0.1
    eval_size = get_eval_size()
    mask_transform = transforms.Compose([
        transforms.Resize((eval_size, eval_size)),
        transforms.ToTensor()
    ])
    
    base_path = os.path.join(root_dir, c_name, "train", "good")
    normal_img_paths = []
    if os.path.exists(base_path):
        for i in range(k_shot):
            round_number = round_idx + i
            file_path = os.path.join(base_path, f"{str(round_number).zfill(3)}.png")
            if os.path.isfile(file_path):
                normal_img_paths.append(file_path)
    if not normal_img_paths and os.path.exists(base_path):
        all_imgs = sorted([str(p) for p in Path(base_path).glob("*.png")])
        if all_imgs: normal_img_paths = all_imgs[-k_shot:]
            
    cached_text_feat = model._encode_text([describle], use_disk_cache=True)
    cached_normal_feats = None
    if normal_img_paths:
        cached_normal_feats = model.encode_image_for_one_shot_with_aug(
            normal_img_paths,
            use_cache=True,
            dataset_name='mvtec'
        )

    test_dir = os.path.join(root_dir, c_name, "test")
    if not os.path.exists(test_dir):
        print(f"Warning: Test dir not found: {test_dir}")
        return

    p_pred, p_label, i_pred, i_label = [], [], [], []
    count = 0
    start_time = time.time()
    print(f"Processing Class: {c_name} ...")
    logging.info(f"Processing Class: {c_name} ...")

    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if not file.endswith('.png'): continue
            file_path = os.path.join(root, file)
            count += 1
            
            is_normal = 'good' in root.split(os.sep)
            if is_normal:
                img_mask = np.zeros((eval_size, eval_size))
            else:
                mask_path = file_path.replace('test', 'ground_truth').replace('.png', '_mask.png')
                if os.path.exists(mask_path):
                    img_mask = Image.open(mask_path).convert('L')
                    img_mask = mask_transform(img_mask).squeeze().numpy()
                    img_mask[img_mask > 0.1] = 1
                    img_mask[img_mask <= 0.1] = 0
                else:
                    img_mask = np.zeros((eval_size, eval_size))
            
            with torch.no_grad():
                pixel_output, cls_score = model.generate({
                    'prompt': describle, 'image_paths': [file_path], 'normal_img_paths': [],
                    'cached_text_features': cached_text_feat,
                    'cached_normal_patch_tokens': cached_normal_feats,
                    'r': r, 'mask': img_mask
                })
            
            anomaly_map = pixel_output.reshape(eval_size, eval_size).detach().cpu().numpy()
            p_label.append(img_mask)
            p_pred.append(anomaly_map)
            i_label.append(0 if is_normal else 1)
            
            flat_matrix = anomaly_map.ravel()
            score1 = np.mean(np.sort(flat_matrix)[-30:]) if len(flat_matrix) > 30 else np.max(flat_matrix)
            i_pred.append(0.1 * cls_score.cpu().detach().numpy() + 0.9 * score1)

    if len(p_label) > 0:
        p_auroc = round(roc_auc_score(np.array(p_label).ravel(), np.array(p_pred).ravel()) * 100, 2)
        i_auroc = round(roc_auc_score(np.array(i_label).ravel(), np.array(i_pred).ravel()) * 100, 2)
        p_pro = round(cal_pro_score(np.array(p_label), np.array(p_pred)) * 100, 2)
        ap = round(average_precision_score(np.array(i_label), np.array(i_pred)) * 100, 2)
        
        cost_time = time.time() - start_time
        print(f"  Finished {count} images in {cost_time:.2f}s")
        print(f"  {c_name} -> i_AUC: {i_auroc}, p_AUC: {p_auroc}, PRO: {p_pro}, AP: {ap}")
        logging.info(f"  Finished {count} images in {cost_time:.2f}s")
        logging.info(f"  {c_name} -> i_AUC: {i_auroc}, p_AUC: {p_auroc}, PRO: {p_pro}, AP: {ap}")
    else:
        print(f"  [Warning] No test images found for {c_name} in {test_dir}")
        logging.warning(f"  [Warning] No test images found for {c_name} in {test_dir}")


def main(**args):
    # 1) 读取 config + 初始化分布式 + 随机种子
    epochs_cli       = args.get("epochs")        # 保存 CLI 传入的值（config_env 会覆盖）
    lambda_align_cli = args.get("lambda_align")  # 同上
    lambda_gate_cli  = args.get("lambda_gate")   # 同上
    config_env(args)
    if epochs_cli is not None:
        args["epochs"] = epochs_cli
    if lambda_align_cli is not None:
        args["lambda_align"] = lambda_align_cli
    if lambda_gate_cli is not None:
        args["lambda_gate"] = lambda_gate_cli

    # 2) 读取 DeepSpeed 的 json 配置
    args["ds_config_path"] = f'dsconfig/{args["model"]}_stage_{args["stage"]}.json'
    dschf = HfDeepSpeedConfig(args['ds_config_path'])
    args['dschf'] = dschf

    # 设置数据集名称，用于缓存区分
    args['dataset_name'] = 'mvtec'
    
    build_directory(args["save_path"])
    build_directory(args["log_path"])

    # 只在 rank0 写 log 文件，避免多进程抢文件
    if args["log_path"] and (not dist.is_initialized() or dist.get_rank() == 0):
        logging.basicConfig(
            format="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s",
            level=logging.DEBUG,
            filename=os.path.join(args["log_path"], f"train_{time.strftime('%Y%m%d_%H%M%S')}.log"),
            filemode="w",
        )
        # Suppress massive PIL/PngImagePlugin debug logs
        logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
        logging.getLogger("PIL.Image").setLevel(logging.WARNING)

    # 3) 构建数据集 & DataLoader
    # load_mvtec_dataset 内部如果用 args['dschf']，现在已经兼容了
    train_data, train_iter, sampler = load_mvtec_dataset(args)

    # 估算迭代总步数 (与 train_visa 相同写法)
    length = args["epochs"] * len(train_data) // args["world_size"] // dschf.config["train_micro_batch_size_per_gpu"]
    total_steps = 2 * args["epochs"] * len(train_data) // dschf.config["train_batch_size"]
    args["total_steps"] = total_steps

    # 4) 构建模型 + DeepSpeed engine
    # 注意：load_model 里要用 dist_init_required=False（下面会给参考）
    agent = load_model(args)

    # 同步一下，确保所有 rank 模型都 ready
    if dist.is_initialized():
        dist.barrier()

    # 5) 训练循环
    pbar = tqdm(
        total=length,
        disable=(dist.is_initialized() and dist.get_rank() != 0),  # 只在 rank0 显示进度条
    )
    current_step = 0

    for epoch_i in tqdm(range(args["epochs"]), disable=(dist.is_initialized() and dist.get_rank() != 0)):
        if sampler is not None and hasattr(sampler, "set_epoch"):
            # 分布式 sampler 每个 epoch 要设置一下，否则各 rank 样本顺序相同
            sampler.set_epoch(epoch_i)

        iter_every_epoch = 0

        for batch in train_iter:
            iter_every_epoch += 1

            agent.train_model(
                batch,
                current_step=current_step,
                pbar=pbar,
            )

            del batch
            current_step += 1

            gc.collect()
            torch.cuda.empty_cache()

        # 每个 epoch 结束同步
        if dist.is_initialized():
            dist.barrier()

        # 只在 rank0 保存模型
        if not dist.is_initialized() or dist.get_rank() == 0:
            agent.save_model(args["save_path"], epoch_i, total_epochs=args["epochs"])
            # Evaluate screw
            # agent.ds_engine.module.eval()
            # evaluate_screw(agent.ds_engine.module, args['image_root_path'], epoch_i)
            # agent.ds_engine.module.train()

    # 6) 训练结束可以销毁 PG（避免 PyTorch 2.4 的 NCCL warning）
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    # 你原来就有的设置
    args["layers"] = [7, 15, 23, 31]
    main(**args)
