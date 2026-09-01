import torch.nn as nn
import os
import logging
import random
import hashlib
import torchvision.transforms as transforms
import torch.nn.functional as F
import kornia as K
import torch
import torch.distributed as dist
import wandb
from pathlib import Path
from PIL import Image

from .ImageBind import *
from .ImageBind import data
from .AnomalyGPT_models import LinearLayer, Adapter, MMCI
from utils.loss import FocalLoss, BinaryDiceLoss

from .siamese_model_conf_gnn import GNNNet, compute_comprehensive_loss

logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


# =============================================================================
# Text Features Disk Cache
# =============================================================================
class TextFeatureCache:
    """
    磁盘缓存 CLIP text embeddings
    避免每次训练都重新计算冻结的 text features
    """
    def __init__(self, cache_dir='./cache/text_features'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}  # 内存缓存，避免重复磁盘IO
        self._load_all_cached()

    def _get_cache_path(self, class_name):
        """生成缓存文件路径"""
        # 使用类名的 hash 作为文件名，避免特殊字符问题
        safe_name = class_name.replace(' ', '_').replace('/', '_')
        return self.cache_dir / f"{safe_name}.pt"

    def _load_all_cached(self):
        """启动时加载所有已缓存的 features"""
        if not self.cache_dir.exists():
            return

        for cache_file in self.cache_dir.glob("*.pt"):
            try:
                data = torch.load(cache_file, map_location='cpu')
                class_name = data['class_name']
                self.memory_cache[class_name] = data['features']
                print(f"[Cache] Loaded text features for '{class_name}'")
            except Exception as e:
                print(f"[Cache] Failed to load {cache_file}: {e}")

    def get(self, class_name):
        """获取缓存的 features，如果不存在返回 None"""
        return self.memory_cache.get(class_name)

    def set(self, class_name, features):
        """保存 features 到内存和磁盘"""
        # 保存到内存
        self.memory_cache[class_name] = features.cpu()

        # 保存到磁盘
        cache_path = self._get_cache_path(class_name)
        torch.save({
            'class_name': class_name,
            'features': features.cpu()
        }, cache_path)
        print(f"[Cache] Saved text features for '{class_name}' to {cache_path}")

    def has(self, class_name):
        """检查是否有缓存"""
        return class_name in self.memory_cache

    def clear(self):
        """清除所有缓存"""
        self.memory_cache.clear()
        for cache_file in self.cache_dir.glob("*.pt"):
            cache_file.unlink()
        print("[Cache] Cleared all text feature caches")


class ImageFeatureCache:
    """
    磁盘缓存 One-shot 图像特征
    """
    def __init__(self, cache_dir='./cache/image_features'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}
        self._load_all_cached()

    def _get_cache_path(self, key):
        """生成缓存文件路径"""
        key_hash = hashlib.md5(key.encode('utf-8')).hexdigest()
        return self.cache_dir / f"{key_hash}.pt"

    def _load_all_cached(self):
        """启动时加载所有已缓存的 features"""
        if not self.cache_dir.exists():
            return

        for cache_file in self.cache_dir.glob("*.pt"):
            try:
                data = torch.load(cache_file, map_location='cpu')
                key = data['key']
                self.memory_cache[key] = data['features']
            except Exception as e:
                print(f"[Cache] Failed to load {cache_file}: {e}")

    def get(self, key):
        return self.memory_cache.get(key)

    def set(self, key, features):
        self.memory_cache[key] = features
        cache_path = self._get_cache_path(key)
        torch.save({'key': key, 'features': features}, cache_path)

    def has(self, key):
        return key in self.memory_cache


# 全局缓存实例 - 支持多个数据集的缓存
_text_feature_cache = {}
_image_feature_cache = {}

def get_text_feature_cache(cache_dir='./cache/text_features', dataset_name=None):
    """获取全局缓存实例（支持按数据集名称分离缓存）"""
    global _text_feature_cache
    
    # 如果指定了数据集名称，使用数据集特定的缓存目录
    if dataset_name:
        cache_dir = f'./cache/{dataset_name}'
    
    if cache_dir not in _text_feature_cache:
        _text_feature_cache[cache_dir] = TextFeatureCache(cache_dir)
    return _text_feature_cache[cache_dir]


def get_image_feature_cache(cache_dir='./cache/image_features', dataset_name=None):
    """获取图像特征缓存实例（支持按数据集名称分离缓存）"""
    global _image_feature_cache

    if dataset_name:
        cache_dir = f'./cache/{dataset_name}/image_features'

    if cache_dir not in _image_feature_cache:
        _image_feature_cache[cache_dir] = ImageFeatureCache(cache_dir)
    return _image_feature_cache[cache_dir]


def _build_image_cache_key(image_paths, aug=False, namespace=None):
    """构建 One-shot 图像特征缓存 key"""
    if isinstance(image_paths, (list, tuple)):
        key = '|'.join([str(p) for p in image_paths])
    else:
        key = str(image_paths)
    prefix = 'aug' if aug else 'plain'
    if namespace:
        prefix = f'{namespace}::{prefix}'
    return f"{prefix}::{key}"


def _features_to_device(features, device):
    """将缓存的特征转到指定设备"""
    if isinstance(features, (list, tuple)):
        return [f.to(device) for f in features]
    return features.to(device)

# =============================================================================
# Global Variables & Helper Functions
# =============================================================================
CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper', 'object',
               'candle', 'cashew', 'chewinggum', 'fryum', 'macaroni', 'pcb', 'pipe fryum']

prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect',
                 '{} without damage']
prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']

prompt_state = [prompt_normal, prompt_abnormal]
prompt_templates = ['a photo of a {}.', 'a photo of the {}.']

objs = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal nut', 'pill', 'screw', 'tile',
        'toothbrush', 'transistor', 'wood', 'zipper', 'object',
        'candle', 'cashew', 'chewinggum', 'fryum', 'macaroni', 'pcb', 'pipe fryum', 'macaroni1', 'macaroni2', 'pcb1',
        'pcb2', 'pcb3', 'pcb4', 'capsules']

prompt_sentences_cache = None
text_embeddings_cache = {}

def get_prompt_sentences(device):
    global prompt_sentences_cache
    if prompt_sentences_cache is not None:
        first_key = list(prompt_sentences_cache.keys())[0]
        if prompt_sentences_cache[first_key][0].device == device:
            return prompt_sentences_cache

    cache = {}
    for obj in objs:
        prompt_sentence_obj = []
        for i in range(len(prompt_state)):
            prompted_state = [state.format(obj) for state in prompt_state[i]]
            prompted_sentence = []
            for s in prompted_state:
                for template in prompt_templates:
                    prompted_sentence.append(template.format(s))
            prompted_sentence = data.load_and_transform_text(prompted_sentence, device)
            prompt_sentence_obj.append(prompted_sentence)
        cache[obj] = prompt_sentence_obj

    prompt_sentences_cache = cache
    return cache

def _compute_single_class_embedding(model, obj_name, device):
    prompt_sentences = get_prompt_sentences(device)
    key_name = obj_name.replace('_', ' ')
    if key_name not in prompt_sentences:
        key_name = 'object'

    sentence = prompt_sentences[key_name]
    normal_sentences = sentence[0].to(device)
    abnormal_sentences = sentence[1].to(device)

    with torch.no_grad():
        class_embeddings_normal = model({ModalityType.TEXT: normal_sentences})[ModalityType.TEXT]
        class_embeddings_abnormal = model({ModalityType.TEXT: abnormal_sentences})[ModalityType.TEXT]

    if isinstance(class_embeddings_normal, (tuple, list)):
        class_embeddings_normal = class_embeddings_normal[0]
    if isinstance(class_embeddings_abnormal, (tuple, list)):
        class_embeddings_abnormal = class_embeddings_abnormal[0]

    class_embeddings_normal = class_embeddings_normal.reshape(
        (1, len(prompt_templates) * len(prompt_normal), 1024))
    class_embeddings_normal = class_embeddings_normal.mean(dim=1, keepdim=True)
    class_embeddings_normal = class_embeddings_normal / class_embeddings_normal.norm(dim=-1, keepdim=True)

    class_embeddings_abnormal = class_embeddings_abnormal.reshape(
        (1, len(prompt_templates) * len(prompt_abnormal), 1024))
    class_embeddings_abnormal = class_embeddings_abnormal.mean(dim=1, keepdim=True)
    class_embeddings_abnormal = class_embeddings_abnormal / class_embeddings_abnormal.norm(dim=-1, keepdim=True)

    text_features = torch.cat([class_embeddings_normal, class_embeddings_abnormal], dim=1)
    return text_features

def encode_text_with_prompt_ensemble(model, obj, device, use_disk_cache=False, dataset_name=None):
    """
    编码 text features，支持磁盘缓存

    Args:
        model: CLIP visual encoder
        obj: 类别名称 (str 或 list)
        device: 目标设备
        use_disk_cache: 是否使用磁盘缓存 (默认 True)
        dataset_name: 数据集名称 (可选，如 'mvtec', 'visa')，用于分离缓存目录

    Returns:
        text_features: [B, 2, 1024] (Normal, Abnormal)
    """
    global text_embeddings_cache

    if isinstance(obj, str):
        obj = [obj]

    # 获取磁盘缓存
    disk_cache = get_text_feature_cache(dataset_name=dataset_name) if use_disk_cache else None

    embeddings_list = []
    for class_name in obj:
        key_name = class_name.replace('_', ' ')
        cache_key = (key_name, str(device))

        # 1. 先查内存缓存
        if cache_key in text_embeddings_cache:
            embeddings_list.append(text_embeddings_cache[cache_key])
            continue

        # 2. 再查磁盘缓存
        if disk_cache and disk_cache.has(key_name):
            embedding = disk_cache.get(key_name).to(device)
            text_embeddings_cache[cache_key] = embedding
            embeddings_list.append(embedding)
            continue

        # 3. 都没有，计算并缓存
        embedding = _compute_single_class_embedding(model, class_name, device)
        text_embeddings_cache[cache_key] = embedding

        # 保存到磁盘
        if disk_cache:
            disk_cache.set(key_name, embedding)

        embeddings_list.append(embedding)

    text_features = torch.cat(embeddings_list, dim=0)
    return text_features


class OpenLLAMAPEFTModel(nn.Module):
    def __init__(self, **args):
        super(OpenLLAMAPEFTModel, self).__init__()
        self.args = args
        imagebind_ckpt_path = args.get('imagebind_ckpt_path', '')
        print(f'Initializing visual encoder from {imagebind_ckpt_path} ...')

        # Frozen ImageBind-Huge visual encoder (OpenCLIP ViT-H/14), joint dim 1024
        self.visual_encoder, self.visual_hidden_size = imagebind_model.imagebind_huge(args)
        if imagebind_ckpt_path:
            imagebind_ckpt = torch.load(imagebind_ckpt_path, map_location=torch.device('cpu'))
            self.visual_encoder.load_state_dict(imagebind_ckpt, strict=True)
        self.feat_dim = 1024
        self.patch_in = 1280
        self.num_patches = 256

        self.image_decoder = LinearLayer(self.patch_in, self.feat_dim, 4)
        self.adapter = Adapter(self.feat_dim, self.feat_dim)
        self.aGNN = GNNNet(all_channel=self.feat_dim)
        self.mmci = MMCI(self.feat_dim, self.feat_dim)  # 多尺度卷积交叉融合
        self.loss_focal = FocalLoss()
        self.loss_dice = BinaryDiceLoss()

        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder.eval()
        self.device = torch.cuda.current_device()

        self.iter = 0
        self.is_main_process = (not dist.is_initialized()) or (dist.get_rank() == 0)
        # 数据集名称，用于缓存区分
        self.dataset_name = args.get('dataset_name', 'mvtec')

        # Fusion weight (fixed) - 70% Drift, 30% CLIP
        self.drift_weight = 0.7
        # Drift-Visual Alignment Loss 权重 (0.0 = 关闭，等价于 asa_drift)
        # Drift separation loss weight (L_drift); effective weight is value * 10.
        _lambda_align = args.get('lambda_align', 1.0)
        self.lambda_align = 1.0 if _lambda_align is None else float(_lambda_align)
        # Gate supervision loss weight (L_gate). 0.0 disables it.
        _lambda_gate = args.get('lambda_gate', 0.2)
        self.lambda_gate = 0.2 if _lambda_gate is None else float(_lambda_gate)

        if self.is_main_process:
            wandb.init(
                project="AnomalyGPT-KAG-PEFT",
                config=args,
                name=f"patch_align_v1_{random.randint(0, 1000)}",
                resume="allow"
            )

    def _encode_text(self, obj, use_disk_cache=False):
        return encode_text_with_prompt_ensemble(
            self.visual_encoder, obj, self.device,
            use_disk_cache=use_disk_cache, dataset_name=self.dataset_name)

    def rot90_img(self, x, k):
        degreesarr = [0., 90., 180., 270., 360]
        degrees = torch.tensor(degreesarr[k]).to(torch.float32).to(self.device)
        x = K.geometry.transform.rotate(x, angle=degrees, padding_mode='reflection')
        return x

    def _image_cache_namespace(self):
        return 'imagebind-224'

    def _output_spatial_size(self):
        return 224

    def _load_vision_data(self, image_paths):
        return data.load_and_transform_vision_data(image_paths, self.device)

    def encode_image(self, image_paths):
        image_tensors = self._load_vision_data(image_paths)
        inputs = {ModalityType.VISION: image_tensors}
        inputs = {key: inputs[key].to(torch.float32) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            image_embeds = embeddings['vision'][0]
            patch_features = embeddings['vision'][1]

        image_embeds = self.adapter(image_embeds)
        patch_tokens = self.image_decoder(patch_features)

        return image_embeds, patch_tokens

    def encode_image_for_one_shot(self, image_paths, use_cache=False, dataset_name=None):
        cache = get_image_feature_cache(dataset_name=dataset_name) if use_cache else None
        cache_key = _build_image_cache_key(
            image_paths, aug=False, namespace=self._image_cache_namespace()
        ) if cache else None

        if cache and cache.has(cache_key):
            cached_features = cache.get(cache_key)
            return _features_to_device(cached_features, self.device), None

        inputs = {ModalityType.VISION: self._load_vision_data(image_paths)}
        inputs = {key: inputs[key].to(torch.float32) for key in inputs}
        for key in inputs:
            images = inputs[key]
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            patch_features = embeddings['vision'][1]
            for i in range(len(patch_features)):
                patch_features[i] = patch_features[i].transpose(0, 1)[:, 1:, :]

        if cache:
            cache.set(cache_key, [pf.detach().cpu() for pf in patch_features])

        return patch_features, images

    def encode_image_for_one_shot_with_aug(self, image_paths, use_cache=False, dataset_name=None):
        cache = get_image_feature_cache(dataset_name=dataset_name) if use_cache else None
        cache_key = _build_image_cache_key(
            image_paths, aug=True, namespace=self._image_cache_namespace()
        ) if cache else None

        if cache and cache.has(cache_key):
            cached_features = cache.get(cache_key)
            return _features_to_device(cached_features, self.device)

        image_tensors = self._load_vision_data(image_paths).to(torch.float32)
        B, C, H, W = image_tensors.shape
        rotated_images = torch.zeros((4, B, C, H, W)).to(torch.float32).to(self.device)
        for j, degree in enumerate([0, 1, 2, 3]):
            rotated_img = self.rot90_img(image_tensors, degree)
            rotated_images[j] = rotated_img
        image_tensors = rotated_images.transpose(0, 1).reshape(B * 4, C, H, W)
        inputs = {ModalityType.VISION: image_tensors}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            patch_features = embeddings['vision'][1]
            for i in range(len(patch_features)):
                patch_features[i] = patch_features[i].transpose(0, 1)[:, 1:, :].reshape(B, 4, self.num_patches, self.patch_in).reshape(B, 4 * self.num_patches, self.patch_in)

        if cache:
            cache.set(cache_key, [pf.detach().cpu() for pf in patch_features])

        return patch_features

    def encode_image_from_tensor(self, image_tensors):
        if not isinstance(image_tensors, list):
            image_tensors = [image_tensors]
        stacked_images = torch.stack(image_tensors, dim=0).to(self.device)
        stacked_images = self._prepare_vision_tensor(stacked_images)
        inputs = {ModalityType.VISION: stacked_images}
        inputs = {key: inputs[key].to(torch.float32) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            image_embeds = embeddings['vision'][0]
            patch_features = embeddings['vision'][1]
        image_embeds = self.adapter(image_embeds)
        patch_tokens = self.image_decoder(patch_features)
        return image_embeds, patch_tokens

    # -------------------------------------------------------------------------
    # Training Forward
    # -------------------------------------------------------------------------
    def forward(self, inputs):
        if 'masks' in inputs:
            image_paths = inputs['images']
            class_name = inputs['class_names']

            # 使用缓存加速训练
            feats_text_tensor = self._encode_text(class_name, use_disk_cache=True)
            image_embeds, patch_tokens = self.encode_image_from_tensor(image_paths)

            B = patch_tokens[0].size(0)
            if feats_text_tensor.size(0) == 1 and B > 1:
                feats_text_tensor = feats_text_tensor.expand(B, -1, -1)

            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            image_map = image_embeds.unsqueeze(1) @ feats_text_tensor.transpose(-2, -1)
            image_map = torch.squeeze(image_map, dim=1)

            # GNN Prep
            B = patch_tokens[0].size(0)
            patch_tokens = [patch_tokens[i].transpose(1, 2).view(B, self.feat_dim, 16, 16) for i in range(4)]
            patch_tokens = self.mmci(patch_tokens)

            # Clone MMCI output for Clean branch (保留CLIP原始对齐)
            clean_visuals = [p.clone() for p in patch_tokens]
            clean_text = F.normalize(feats_text_tensor, dim=-1)

            # GNN Forward
            refined_visuals, drifted_normal_list, drifted_abnormal_list, mod_info = self.aGNN(
                patch_tokens[0],
                patch_tokens[1],
                patch_tokens[2],
                patch_tokens[3],
                feats_text_tensor,
            )

            # Fuse Clean + Drift at logit level, then upsample + softmax
            fused_maps = []
            score_normal_maps = []    # 存储各层 score，用于 align loss
            score_abnormal_maps = []

            # Ground Truth
            gt = torch.stack(inputs['masks'], dim=0).to(self.device)
            if gt.dim() == 4:
                gt = gt.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
            gt[gt > 0.3], gt[gt <= 0.3] = 1, 0
            output_size = gt.shape[-2:]

            for i in range(len(refined_visuals)):
                B, C, H, W = refined_visuals[i].shape

                # --- Branch A: Clean (MMCI Visual <-> Clean Text) → logits ---
                mmci_feat = F.normalize(clean_visuals[i], dim=1)
                mmci_flat = mmci_feat.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
                clean_logits = 100.0 * mmci_flat @ clean_text.transpose(-2, -1)  # [B, HW, 2]
                clean_logits = clean_logits.permute(0, 2, 1).view(B, 2, H, W)

                # --- Branch B: Drift (GNN Visual <-> Drifted Normal + Abnormal Text) → logits ---
                gnn_feat = F.normalize(refined_visuals[i], dim=1)
                drift_normal = F.normalize(drifted_normal_list[i], dim=1)
                drift_abnormal = F.normalize(drifted_abnormal_list[i], dim=1)
                score_normal = (gnn_feat * drift_normal).sum(dim=1, keepdim=True)
                score_abnormal = (gnn_feat * drift_abnormal).sum(dim=1, keepdim=True)
                drift_logits = torch.cat([score_normal, score_abnormal], dim=1) * 100.0

                score_normal_maps.append(score_normal.squeeze(1))    # [B, H, W]
                score_abnormal_maps.append(score_abnormal.squeeze(1))

                # --- Fuse logits → upsample → softmax ---
                fused_logits = (1 - self.drift_weight) * clean_logits + self.drift_weight * drift_logits
                fused_map = F.interpolate(fused_logits, size=output_size, mode='bilinear', align_corners=True)
                fused_map = torch.softmax(fused_map, dim=1)
                fused_maps.append(fused_map)

            # Classification Loss
            label, _ = torch.max(gt.view(gt.size(0), -1), dim=1, keepdim=True)
            label = F.one_hot(label.squeeze(1).long(), num_classes=2)
            criterion = nn.BCELoss()
            cls_loss = criterion(torch.sigmoid(image_map.float()), label.float())

            # Pixel Loss (Focal + Dice) on 4 fused maps
            loss_pixel = 0
            for fused_map in fused_maps:
                f_loss = self.loss_focal(fused_map, gt)
                d_loss = (self.loss_dice(fused_map[:, 1, :, :], gt) +
                          self.loss_dice(fused_map[:, 0, :, :], 1 - gt))
                loss_pixel += 1.0 * f_loss + 0.3 * d_loss

            # Compute Accuracy - Average 4 fused maps
            fused_abnormal = [m[:, 1, :, :] for m in fused_maps]
            anomaly_map_all = torch.mean(torch.stack(fused_abnormal, dim=0), dim=0).unsqueeze(1)
            anomaly_map_all_squeezed = torch.squeeze(anomaly_map_all, dim=1)
            anomaly_map_all_squeezed = (anomaly_map_all_squeezed > 0.5).float()
            matches = anomaly_map_all_squeezed == gt
            gen_acc = matches.sum().item() / matches.numel()

            # Comprehensive Loss (Drift + Aux Segmentation)
            # 准备gt_mask
            if gt.dim() == 3:
                gt_mask = gt.unsqueeze(1)
            else:
                gt_mask = gt

            # Warmup策略
            drift_warmup_steps = 300
            drift_lambda = min(1.0, float(self.iter) / float(drift_warmup_steps))

            # --- Drift Separation Loss (L_drift, paper formula 24) ---
            # 防止 drifted normal/abnormal 文本塌陷（比原始冻结 embedding 更相似）
            # L_drift = Σ_l ||max(<T^{l,+}, T^{l,-}> - (<t^+,t^-> - ε), 0)||^2
            combined_loss, drift_details = compute_comprehensive_loss(
                drift_info=mod_info,
                gt_mask=gt_mask if self.lambda_gate > 0 else None,
                lambda_drift=self.lambda_align * drift_lambda,
                lambda_seg=self.lambda_gate,
            )
            align_loss = combined_loss
            drift_loss = drift_details.get('drift_loss', 0.0)
            aux_seg_loss = drift_details.get('aux_seg_loss', 0.0)

            # Total Loss
            total_loss = cls_loss + loss_pixel + align_loss

            if self.is_main_process and self.training:
                step = self.iter
                log_dict = {
                    "Loss/cls_loss": cls_loss.item(),
                    "Loss/pixel_loss": loss_pixel.item(),
                    "Loss/drift_loss": drift_loss,
                    "Loss/gating_loss": aux_seg_loss,
                    "Loss/total_loss": total_loss.item(),
                    "Metrics/gen_acc": gen_acc,
                }

                # 记录 TextDrift 缩放系数
                if 'learned_scale' in mod_info:
                    log_dict["TextDrift/scale_param"] = mod_info['learned_scale']

                wandb.log(log_dict, step=step)
                self.iter += 1

            return total_loss, gen_acc
        else:
            return torch.tensor(0.0).to(self.device), 0.0

    # -------------------------------------------------------------------------
    # Inference / One-Shot
    # -------------------------------------------------------------------------
    def extract_multimodal_feature(self, inputs, web_demo):
        # 支持复用预编码的 CLIP 图像特征（加速多 epoch 评估）
        if 'cached_image_embeds' in inputs and 'cached_patch_tokens' in inputs:
            # 使用预缓存的 CLIP 编码，跳过 encode_image
            image_embeds = inputs['cached_image_embeds']
            patch_tokens = inputs['cached_patch_tokens']
            feats_text_tensor = inputs.get('cached_text_features', inputs.get('feats_text_tensor'))
        elif 'cached_text_features' in inputs:
            feats_text_tensor = inputs['cached_text_features']
            if not web_demo:
                image_embeds, patch_tokens = self.encode_image(inputs['image_paths'])
        else:
            if inputs['image_paths']:
                prompt = inputs['prompt']
                c_name = 'object'
                for name in CLASS_NAMES:
                    if name in prompt:
                        c_name = name
                        break
                if not web_demo:
                    image_embeds, patch_tokens = self.encode_image(inputs['image_paths'])
                    feats_text_tensor = self._encode_text([c_name])

        B = patch_tokens[0].size(0)
        if feats_text_tensor.size(0) == 1 and B > 1:
            feats_text_tensor = feats_text_tensor.expand(B, -1, -1)

        patch_tokens = [patch_tokens[i].transpose(1, 2).view(B, self.feat_dim, 16, 16) for i in range(4)]
        patch_tokens = self.mmci(patch_tokens)

        # Clone MMCI output for Clean branch
        clean_visuals = [p.clone() for p in patch_tokens]
        clean_text = F.normalize(feats_text_tensor, dim=-1)

        # GNN Forward
        refined_visuals, drifted_normal_list, drifted_abnormal_list, _ = self.aGNN(
            patch_tokens[0],
            patch_tokens[1],
            patch_tokens[2],
            patch_tokens[3],
            feats_text_tensor,
        )

        # Fuse Clean + Drift at logit level (与训练一致)
        fused_maps = []
        output_size = self._output_spatial_size()

        for i in range(len(refined_visuals)):
            B, C, H, W = refined_visuals[i].shape

            # --- Branch A: Clean (MMCI Visual <-> Clean Text) → logits ---
            mmci_feat = F.normalize(clean_visuals[i], dim=1)
            mmci_flat = mmci_feat.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
            clean_logits = 100.0 * mmci_flat @ clean_text.transpose(-2, -1)  # [B, HW, 2]
            clean_logits = clean_logits.permute(0, 2, 1).view(B, 2, H, W)

            # --- Branch B: Drift (GNN Visual <-> Drifted Normal + Abnormal Text) → logits ---
            gnn_feat = F.normalize(refined_visuals[i], dim=1)
            drift_normal = F.normalize(drifted_normal_list[i], dim=1)
            drift_abnormal = F.normalize(drifted_abnormal_list[i], dim=1)
            score_normal = (gnn_feat * drift_normal).sum(dim=1, keepdim=True)
            score_abnormal = (gnn_feat * drift_abnormal).sum(dim=1, keepdim=True)
            drift_logits = torch.cat([score_normal, score_abnormal], dim=1) * 100.0

            # --- Fuse logits → upsample → softmax ---
            fused_logits = (1 - self.drift_weight) * clean_logits + self.drift_weight * drift_logits
            fused_map = F.interpolate(fused_logits, size=output_size, mode='bilinear', align_corners=True)
            fused_map = torch.softmax(fused_map, dim=1)
            fused_maps.append(fused_map[:, 1, :, :])  # 取abnormal channel

        # 4个fused map跨层平均
        anomaly_map_ret = torch.mean(torch.stack(fused_maps, dim=0), dim=0).unsqueeze(1)

        # One-Shot Support Set Logic (Unchanged)
        has_cache = 'cached_normal_patch_tokens' in inputs
        has_paths = inputs.get('normal_img_paths')

        if has_cache or has_paths:
            # 优先使用缓存的 query_patch_tokens（来自 encode_image_for_one_shot）
            if 'query_patch_tokens' in inputs:
                query_patch_tokens = inputs['query_patch_tokens']
            elif 'image_paths' in inputs:
                query_patch_tokens, _ = self.encode_image_for_one_shot(inputs['image_paths'])
            else:
                query_patch_tokens = None

            if has_cache:
                normal_patch_tokens = inputs['cached_normal_patch_tokens']
            else:
                is_mvtec = False
                normal_paths = inputs['normal_img_paths']
                if isinstance(normal_paths, list) and len(normal_paths) > 0 and 'mvtec' in str(normal_paths[0]):
                    is_mvtec = True
                elif isinstance(normal_paths, str) and 'mvtec' in normal_paths:
                    is_mvtec = True

                if is_mvtec:
                    normal_patch_tokens = self.encode_image_for_one_shot_with_aug(inputs['normal_img_paths'])
                else:
                    normal_patch_tokens, normal_images = self.encode_image_for_one_shot(inputs['normal_img_paths'])

            if query_patch_tokens is not None:
                sims = []
                for i in range(len(query_patch_tokens)):
                    # 恢复旧的 cosine_similarity 方式
                    query_patch_tokens_reshaped = query_patch_tokens[i].view(self.num_patches, 1, self.patch_in)
                    normal_tokens_reshaped = normal_patch_tokens[i].reshape(1, -1, self.patch_in)
                    cosine_similarity_matrix = F.cosine_similarity(query_patch_tokens_reshaped, normal_tokens_reshaped, dim=2)
                    sim_max, _ = torch.max(cosine_similarity_matrix, dim=1)
                    sims.append(sim_max)

                sim = torch.mean(torch.stack(sims, dim=0), dim=0).reshape(1, 1, 16, 16)
                sim = F.interpolate(sim, size=output_size, mode='bilinear', align_corners=True)
                anomaly_map = 1 - sim
                anomaly_map = torch.cat([sim, anomaly_map], dim=1)
                anomaly_map = torch.softmax(anomaly_map, dim=1)
                r = inputs.get('r', 0.5)
                # print(f'One-shot fusion with r={r}')
                anomaly_map_ret = r * anomaly_map_ret + (1 - r) * anomaly_map[:, 1:2, :, :]

        # Gaussian blur post-processing to smooth pixel-level anomaly map
        anomaly_map_ret = K.filters.gaussian_blur2d(anomaly_map_ret, kernel_size=(33, 33), sigma=(4.0, 4.0))

        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        image_map = image_embeds.unsqueeze(1) @ feats_text_tensor.transpose(-2, -1)
        image_score = image_map[0, 0, 1]

        return anomaly_map_ret, image_score

    def generate(self, inputs, web_demo=False):
        pixel_output, image_score = self.extract_multimodal_feature(inputs, web_demo)
        return pixel_output, image_score
