import os
import cv2
from metrics import cal_pro_score
from model.openllama import OpenLLAMAPEFTModel
import torch
from torchvision import transforms
from sklearn.metrics import roc_auc_score,average_precision_score
from PIL import Image
import numpy as np
import argparse
from pathlib import Path
# from visualization import visualizer

parser = argparse.ArgumentParser("KAG_prompt", add_help=True)
# paths
parser.add_argument("--few_shot", type=bool, default=True)
parser.add_argument("--k_shot", type=int, default=1)
parser.add_argument("--round", type=int, default=0)  # support = first k training images per class
parser.add_argument("--ckpt", type=str, default=None, help="override anomalygpt_ckpt_path")

command_args = parser.parse_args()

describles = {
    'bottle': 'bottle',
    'cable': 'cable',
    'capsule': 'capsule',
    'carpet': 'carpet',
    'grid': 'grid',
    'hazelnut': 'hazelnut',
    'leather': 'leather',
    'metal_nut': 'metal nut',
    'pill': 'pill',
    'screw': 'screw',
    'tile': 'tile',
    'toothbrush': 'toothbrush',
    'transistor': 'transistor',
    'wood': 'wood',
    'zipper': 'zipper'
}

FEW_SHOT = command_args.few_shot
EVAL_SIZE = 224


# init the model
args = {
    'model': 'openllama_peft',
    'imagebind_ckpt_path': '../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth',
    'anomalygpt_ckpt_path': command_args.ckpt or './ckpt/train_mvtec/train_on_mvtec.pt',
    'stage': 2,
    'features_list': [6, 12, 18, 24],
}

model = OpenLLAMAPEFTModel(**args)
delta_ckpt = torch.load(args['anomalygpt_ckpt_path'], map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
model = model.cuda().eval()

p_auc_list = []
i_auc_list = []
p_pro_list =[]
ap_list = []
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


root_dir = os.environ.get('MVTEC_ROOT', '../data/mvtec')

mask_transform = transforms.Compose([
    transforms.Resize((EVAL_SIZE, EVAL_SIZE)),
    transforms.ToTensor()
])

CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

precision = []
r = 0.1
print('[test_mvtec] using per-run in-memory caches')
for c_name in CLASS_NAMES:
    base_path = os.path.join(root_dir, c_name, "train", "good") + "/"
    normal_img_paths = []
    for i in range(command_args.k_shot):
        round_number = command_args.round + i
        file_path = base_path + str(round_number).zfill(3) + ".png"

        if not Path(file_path).is_file():
            break

        normal_img_paths.append(file_path)

    if not normal_img_paths:
        all_image_paths = sorted(Path(base_path).glob("*.png"))
        normal_img_paths = all_image_paths[-command_args.k_shot:]

    cached_text_features = model._encode_text([describles[c_name]], use_disk_cache=False)
    cached_normal_patch_tokens = None
    if FEW_SHOT and normal_img_paths:
        cached_normal_patch_tokens = model.encode_image_for_one_shot_with_aug(
            normal_img_paths,
            use_cache=False,
        )

    p_pred = []
    p_label = []
    i_pred = []
    i_label = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if "test" in file_path and 'png' in file and c_name in file_path:
                if FEW_SHOT:
                    anomaly_map, score = predict(
                        describles[c_name],
                        file_path,
                        normal_img_paths,
                        r,
                        cached_text_features=cached_text_features,
                        cached_normal_patch_tokens=cached_normal_patch_tokens,
                    )
                else:
                    anomaly_map, score = predict(
                        describles[c_name],
                        file_path,
                        [],
                        r,
                        cached_text_features=cached_text_features,
                    )

                is_normal = 'good' in file_path.split('/')[-2]

                if is_normal:
                    img_mask = Image.fromarray(np.zeros((EVAL_SIZE, EVAL_SIZE), dtype=np.uint8), mode='L')
                else:
                    mask_path = file_path.replace('test', 'ground_truth')
                    mask_path = mask_path.replace('.png', '_mask.png')
                    img_mask = Image.open(mask_path).convert('L')

                img_mask = mask_transform(img_mask)
                img_mask[img_mask > 0.1], img_mask[img_mask <= 0.1] = 1, 0
                img_mask = img_mask.squeeze().reshape(EVAL_SIZE, EVAL_SIZE).cpu().numpy()

                anomaly_map = anomaly_map.reshape(EVAL_SIZE, EVAL_SIZE).detach().cpu().numpy()

                # save_path = '/mnt/sda/fenfangtao/AnomalyGPT-main/code/visualize/'
                # visualizer(file_path, anomaly_map, 224, save_path, c_name, 'mvtec')

                p_label.append(img_mask)
                p_pred.append(anomaly_map)

                i_label.append(1 if not is_normal else 0)

                # i_pred.append(anomaly_map.max())
                k = 30
                flat_matrix = anomaly_map.ravel()
                top_k_indices = np.argpartition(-flat_matrix, k)[:k]
                top_k_values = flat_matrix[top_k_indices]
                score1 = np.mean(top_k_values)
                i_pred.append(0.1 * score.cpu().detach().numpy() + 0.9 * score1)

    p_pred = np.array(p_pred)
    p_label = np.array(p_label)

    i_pred = np.array(i_pred)
    i_label = np.array(i_label)

    p_auroc = round(roc_auc_score(p_label.ravel(), p_pred.ravel()) * 100, 2)
    i_auroc = round(roc_auc_score(i_label.ravel(), i_pred.ravel()) * 100, 2)
    p_pro = round(cal_pro_score(p_label, p_pred) * 100, 2)
    ap = round(average_precision_score(i_label, i_pred) * 100, 2)

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
print("ap:", torch.tensor(ap_list).mean())
