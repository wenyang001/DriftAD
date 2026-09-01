import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 0. 基础配置
# ==========================================
def get_norm(dim):
    return nn.BatchNorm2d(dim)

# 兼容性导入
try:
    from . import ConvGRU2 as ConvGRU
except ImportError:
    class ConvGRU:
        class ConvGRUCell(nn.Module):
            def __init__(self, *args, **kwargs): super().__init__()
            def forward(self, x, h): return x
    import sys
    sys.modules['ConvGRU2'] = ConvGRU


# ==========================================
# Module 1: Dual-Scale Drift (The Guide)
# ==========================================
class SharedTopDownDrift(nn.Module):
    def __init__(self, dim, num_layers=4, layer_grid_sizes=[4, 8, 8, 16]):
        """
        层级自适应尺度Grid Drift

        Args:
            dim: 特征维度
            num_layers: 层数 (默认4)
            layer_grid_sizes: 每层的Grid尺度 (默认[4, 8, 8, 16])
                - Layer 0 (浅层): 4×4 → 语义弱，共享drift即可
                - Layer 1 (中浅层): 8×8 → 区域特征
                - Layer 2 (中深层): 8×8 → 区域特征 (稳定)
                - Layer 3 (深层): 16×16 → 语义强，需要更细微的drift
        """
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.layer_grid_sizes = layer_grid_sizes  # 每层独立的Grid尺度

        assert len(layer_grid_sizes) == num_layers, \
            f"layer_grid_sizes长度({len(layer_grid_sizes)})必须等于num_layers({num_layers})"

        # 可学习的漂移缩放系数
        self.scale_param = nn.Parameter(torch.tensor(0.05))

        # 共享的视觉投影 (参数复用，节省内存)
        self.visual_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim * 2, dim, 1),
                get_norm(dim),
                nn.ReLU(),
                nn.Conv2d(dim, dim, 1)
            ) for _ in range(num_layers)
        ])

        self.context_fusion = nn.ModuleList([
            nn.Conv2d(dim * 2, dim, 1) for _ in range(num_layers - 1)
        ])

        # 【关键改进】每层独立的delta生成器 (层级自适应)
        # 输出 2*C: 前C给normal delta, 后C给abnormal delta
        self.delta_gens = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 1), nn.ReLU(),
                nn.Conv2d(dim, dim * 2, 1)
            ) for _ in range(num_layers)
        ])

    def forward(self, text_features, visual_features_list):
        B, K, C = text_features.shape
        text_base = text_features.view(B, K, C, 1, 1)
        active_scale = torch.clamp(self.scale_param, 0.01, 0.2)

        # 【层级自适应处理】每层使用最适合的Grid尺度
        drifted_texts = []
        drift_info_list = []

        # === 步骤1: 提取每层的视觉上下文 (各自的Grid尺度) ===
        raw_contexts = []
        for i in range(self.num_layers):
            vis_feat = visual_features_list[i]
            grid_size = self.layer_grid_sizes[i]  # 每层独立的Grid尺度

            ctx_avg = F.adaptive_avg_pool2d(vis_feat, (grid_size, grid_size))
            ctx_max = F.adaptive_max_pool2d(vis_feat, (grid_size, grid_size))
            ctx_dual = torch.cat([ctx_avg, ctx_max], dim=1).contiguous()
            raw_contexts.append(ctx_dual)

        # === 步骤2: 自顶向下融合 (升级! 无损下采样) ===
        fused_contexts = [None] * self.num_layers
        current_high = self.visual_proj[-1](raw_contexts[-1])
        fused_contexts[-1] = current_high

        for i in range(self.num_layers - 2, -1, -1):
            low_proj = self.visual_proj[i](raw_contexts[i])

            # Bilinear插值对齐空间维度 (upsample或downsample)
            if current_high.shape[2:] != low_proj.shape[2:]:
                current_high = F.interpolate(
                    current_high,
                    size=low_proj.shape[2:],
                    mode='bilinear',
                    align_corners=True
                )

            cat_ctx = torch.cat([low_proj, current_high], dim=1).contiguous()
            fused_ctx = self.context_fusion[i](cat_ctx)
            fused_contexts[i] = fused_ctx
            current_high = fused_ctx

        # === 步骤3: 每层独立生成漂移 (normal/abnormal 各自独立delta) ===
        for i in range(self.num_layers):
            grid_size = self.layer_grid_sizes[i]
            visual_ctx = fused_contexts[i]

            # 生成 2*C delta, split为 normal 和 abnormal
            delta_all = self.delta_gens[i](visual_ctx)  # [B, 2*C, grid, grid]
            delta_normal = F.normalize(delta_all[:, :C], dim=1) * active_scale
            delta_abnormal = F.normalize(delta_all[:, C:], dim=1) * active_scale

            # 拼接为 [B, 2, C, grid, grid]
            delta_per_class = torch.stack([delta_normal, delta_abnormal], dim=1)

            # 加到 text_base 上
            drifted_grid = text_base + delta_per_class
            drifted_grid = F.normalize(drifted_grid, dim=2)

            # 上采样到目标分辨率 (与visual特征对齐)，使用双线性插值
            H, W = visual_features_list[i].shape[2:]
            drifted_upsampled = F.interpolate(
                drifted_grid.view(B*K, C, grid_size, grid_size),
                size=(H, W), mode='bilinear', align_corners=True
            ).view(B, K, C, H, W)

            drifted_texts.append(drifted_upsampled)

            # 记录信息
            drift_info_list.append({
                f'layer{i}_grid_size': grid_size,
                f'layer{i}_delta_mag_normal': delta_normal.norm(dim=1).mean().item(),
                f'layer{i}_delta_mag_abnormal': delta_abnormal.norm(dim=1).mean().item()
            })

        drift_info = {
            'drifted_texts': drifted_texts,
            'drift_metadata': drift_info_list,
            'original_text': text_features,
            'learned_scale': active_scale.item(),
            'layer_grid_sizes': self.layer_grid_sizes
        }

        return drifted_texts, None, drift_info

class FrequencyAwareAmplifier(nn.Module):
    """
    双分支差异放大器

    核心思想:
    - 空间域: 多尺度背景对比度，捕捉局部异常
    - 频率域: FFT分析，捕捉全局纹理异常
    - 双分支简单融合

    参数增量: ~0.05M
    """
    def __init__(self, channels):
        super(FrequencyAwareAmplifier, self).__init__()

        # ===== 空间域分支 =====
        self.local_pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.global_pool = nn.AvgPool2d(9, stride=1, padding=4)

        self.spatial_fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            get_norm(channels),
            nn.ReLU(inplace=True)
        )

        self.spatial_amplifier = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

        self.amp_scale = nn.Parameter(torch.tensor(1.5))

        # ===== 频率域分支 =====
        self.freq_importance = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )

        self.low_freq_weight = nn.Parameter(torch.tensor(0.7))
        self.high_freq_weight = nn.Parameter(torch.tensor(0.3))

        self.freq_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1)
        )

        # ===== 双分支融合权重 =====
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # x: [B, C, H, W], 来自ImageBind (14×14)
        B, C, H, W = x.shape

        # ===== 分支1: 空间域增强 (保留原逻辑) =====
        bg_local = self.local_pool(x)
        bg_global = self.global_pool(x)

        diff_local = torch.abs(x - bg_local)
        diff_global = torch.abs(x - bg_global)

        spatial_combined = torch.cat([x, diff_local, diff_global], dim=1).contiguous()
        spatial_fused = self.spatial_fusion(spatial_combined)

        spatial_weight = self.spatial_amplifier(diff_local + diff_global)
        active_scale = torch.clamp(self.amp_scale, 0.5, 3.0)
        spatial_out = x + (x * spatial_weight * active_scale)

        # ===== 分支2: 频率域增强 (新增) =====

        # Step 1: 2D FFT (实数输入 → 复数频谱)
        x_freq = torch.fft.rfft2(x, norm='ortho')  # [B, C, H, W//2+1]
        x_mag = torch.abs(x_freq)  # 幅度谱
        x_phase = torch.angle(x_freq)  # 相位谱 (保留以重建)

        # Step 2: 高频/低频分离
        # 策略: 中心区域为低频 (DC及周围), 边缘为高频
        # 对于14×14输入 → FFT shape [B, C, 14, 8]
        h_center, w_center = H // 2, (W // 2 + 1) // 2
        radius_low = max(2, min(H, W) // 6)  # 低频半径

        # 创建低频mask (中心圆形区域)
        mask_low = torch.zeros_like(x_mag)
        for h in range(H):
            for w in range(W // 2 + 1):
                dist = ((h - h_center) ** 2 + (w - w_center) ** 2) ** 0.5
                if dist <= radius_low:
                    mask_low[:, :, h, w] = 1.0

        mask_high = 1.0 - mask_low

        # Step 3: 分离并加权
        mag_low = x_mag * mask_low
        mag_high = x_mag * mask_high

        # 学习到的频率权重 (数据集自适应)
        lp_w = torch.clamp(self.low_freq_weight, 0.0, 1.0)
        hp_w = torch.clamp(self.high_freq_weight, 0.0, 1.0)

        # 加权组合 (纹理异常会学到更高的hp_w)
        mag_weighted = lp_w * mag_low + hp_w * mag_high

        # Step 4: 通道级频率重要性调制
        # 不同channel对频率的敏感度不同
        freq_importance = self.freq_importance(x)  # [B, C, 1, 1]
        # 调整shape以匹配FFT输出: [B, C, 1, 1] → [B, C, 1, 1] → broadcast到[B, C, H, W//2+1]
        mag_modulated = mag_weighted * freq_importance  # broadcast自动处理

        # Step 5: 逆FFT重建
        x_freq_new = mag_modulated * torch.exp(1j * x_phase)
        freq_reconstructed = torch.fft.irfft2(x_freq_new, s=(H, W), norm='ortho')

        # Step 6: 频率域特征提取
        freq_diff = torch.abs(x - freq_reconstructed)
        freq_combined = torch.cat([x, freq_diff], dim=1)
        freq_out = self.freq_fusion(freq_combined)

        # ===== 双分支融合: 空间域 + 频率域 =====
        fusion_w = torch.clamp(self.fusion_weight, 0.0, 1.0)
        out = fusion_w * spatial_out + (1 - fusion_w) * freq_out

        return out


# ==========================================
# Module 3: Multi-Head Similarity Gated Attention (The Locator)
# [功能] 多头注意力定位，更精细的异常检测
# ==========================================
class SimilarityGatedAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, f"dim={dim} must be divisible by num_heads={num_heads}"

        # Multi-head投影层
        self.visual_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.text_proj = nn.Conv2d(dim, dim, 1, bias=False)

        # 【改进】每个head独立的可学习温度
        # 不同head可以关注不同尺度/模式的异常
        # 初始化: [0.08, 0.10, 0.12, 0.14] (从尖锐到平滑)
        init_temps = torch.linspace(0.08, 0.14, num_heads)
        self.temperatures = nn.Parameter(init_temps)

        # 融合多头Gate的轻量网络
        self.head_fusion = nn.Sequential(
            nn.Conv2d(num_heads, num_heads // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_heads // 2, 1, 1),
        )

        # 输出调整层
        self.out_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            get_norm(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, visual, text_feat):
        """
        visual: [B, C, H, W] (Amplified Visual Features)
        text_feat: [B, C, H, W] (Drifted Anomaly Text Features)

        Returns:
            out: [B, C, H, W] (Gated Features)
            attn_gate: [B, 1, H, W] (Fused Attention Gate)
        """
        B, C, H, W = visual.shape

        # 1. 多头投影
        q = self.visual_proj(visual)   # [B, C, H, W]
        k = self.text_proj(text_feat)  # [B, C, H, W]

        # Reshape为multi-head: [B, num_heads, head_dim, H, W]
        q = q.view(B, self.num_heads, self.head_dim, H, W)
        k = k.view(B, self.num_heads, self.head_dim, H, W)

        # 2. 归一化（每个head独立）
        q = F.normalize(q, dim=2).contiguous()
        k = F.normalize(k, dim=2).contiguous()

        # 3. 计算多头相似度
        # [B, num_heads, head_dim, H, W] -> [B, num_heads, H, W]
        multi_sim = (q * k).sum(dim=2).contiguous()

        # 4. 每个head使用独立的temperature生成gate
        # temperatures: [num_heads] -> [1, num_heads, 1, 1]
        temps = torch.clamp(self.temperatures, 0.05, 0.20).view(1, -1, 1, 1)
        multi_gates = torch.sigmoid(multi_sim / temps).contiguous()  # [B, num_heads, H, W]

        # 5. 融合多头gates
        # 使用轻量卷积网络学习最优融合方式
        attn_gate = torch.sigmoid(self.head_fusion(multi_gates)).contiguous()  # [B, 1, H, W]

        # 6. 应用门控增强
        out = (visual * (1.0 + attn_gate)).contiguous()

        return self.out_conv(out), attn_gate


# ==========================================
# Main Model: CoattentionModel (Simplified)
# [架构] 黄金三角：Guide -> Amplify -> Locate
# ==========================================
class CoattentionModel(nn.Module):
    def __init__(self, all_channel=1024, layer_grid_sizes=[4, 8, 8, 16]):
        super(CoattentionModel, self).__init__()

        # 最终残差融合层
        self.res_fusion = nn.Sequential(
            nn.Conv2d(all_channel * 2, all_channel, 3, padding=1, bias=False),
            get_norm(all_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(all_channel, all_channel, 3, padding=1, bias=False),
            get_norm(all_channel)
        )
        self.fusion_act = nn.ReLU(inplace=True)

        # 1. Text Drift (Guide) - 层级自适应语义微调
        self.text_drift = SharedTopDownDrift(all_channel, layer_grid_sizes=layer_grid_sizes)

        # 2. Frequency-Aware Amplifier (Enhance)
        self.geo_layers = nn.ModuleList([FrequencyAwareAmplifier(all_channel) for _ in range(4)])

        # 3. Similarity Gating (Locate)
        self.att_layers = nn.ModuleList([SimilarityGatedAttention(all_channel) for _ in range(4)])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None: nn.init.constant_(m.weight, 1)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, inputs1, inputs2, inputs3, inputs4, text_features):
        visual_inputs = [inputs1, inputs2, inputs3, inputs4]

        # --- Stage 1: ASA Amplify (纯放大，无融合) ---
        feat_amplified = []
        for i in range(4):
            feat_amplified.append(self.geo_layers[i](visual_inputs[i]))

        # --- Stage 2: TextDrift guided by ASA features ---
        _, _, drift_info = self.text_drift(text_features, feat_amplified)
        drifted_anomaly_features = drift_info['drifted_texts']

        # --- Stage 3: SpatialGating + my_fcn(gated, ASA) ---
        attn_gates = []
        final_outputs = []
        for i in range(4):
            feat_selected, gate = self.att_layers[i](
                feat_amplified[i], drifted_anomaly_features[i][:, 1])
            out = self.my_fcn(feat_selected, feat_amplified[i])  # skip from ASA
            final_outputs.append(out)
            attn_gates.append(gate)

        drift_info['attn_gates'] = attn_gates
        drifted_normal_list   = [d[:, 0] for d in drifted_anomaly_features]
        drifted_abnormal_list = [d[:, 1] for d in drifted_anomaly_features]

        return tuple(final_outputs), drifted_normal_list, drifted_abnormal_list, drift_info

    def my_fcn(self, filtered, original):
        # [FIX] 显存对齐
        x = torch.cat([filtered, original], dim=1).contiguous()
        residual = self.res_fusion(x)
        return self.fusion_act(original + residual)


def GNNNet(all_channel=1024, layer_grid_sizes=[4, 8, 8, 16]):
    # GNNNet 返回含 SpatialGating 的完整架构（CoattentionModel）
    # full_arch / full_gate / full 消融均依赖此函数
    return CoattentionModel(all_channel=all_channel, layer_grid_sizes=layer_grid_sizes)


# ==========================================
# Comprehensive Loss Function
# [功能] 同时计算 Drift Loss 和 Deep Supervision
# ==========================================
def compute_comprehensive_loss(drift_info, gt_mask=None, lambda_drift=1.0, lambda_seg=0.5):
    loss_dict = {}
    # 获取设备: 如果有gt_mask用其设备，否则从drifted_texts推断
    if gt_mask is not None:
        device = gt_mask.device
    else:
        drifted = drift_info.get('drifted_texts', [])
        device = drifted[0].device if len(drifted) > 0 else 'cpu'
    total_loss = torch.tensor(0.0, device=device)

    # --- Part 1: Drift Separation Loss ---
    # v1 (backup): mean over layers + *10 scale factor
    # L_drift_eff = lambda_drift * 10.0 * mean_l( mean_hw( max(<T^+,T^-> - (<t^+,t^->-ε), 0)^2 ) )
    drifted_texts = drift_info.get('drifted_texts', [])
    original_text  = drift_info.get('original_text', None)  # [B, 2, C]

    if lambda_drift > 0 and original_text is not None and len(drifted_texts) > 0:
        base_sim = F.cosine_similarity(original_text[:, 0], original_text[:, 1], dim=-1)
        base_sim = base_sim.view(-1, 1, 1).detach()   # [B, 1, 1]

        sep_loss = 0
        current_sim_avg = 0
        for drifted in drifted_texts:
            # drifted: [B, 2, C, H, W]  → cosine along C dim (dim=1 after indexing)
            curr_sim = F.cosine_similarity(drifted[:, 0], drifted[:, 1], dim=1)  # [B, H, W]
            current_sim_avg += curr_sim.mean()
            penalty = torch.clamp(curr_sim - (base_sim - 0.05), min=0.0)
            sep_loss += (penalty ** 2).mean()

        sep_loss = sep_loss / len(drifted_texts)          # mean over layers
        current_sim_avg = current_sim_avg / len(drifted_texts)

        drift_loss = (lambda_drift * 10.0) * sep_loss    # *10 scale factor
        total_loss = total_loss + drift_loss
        loss_dict['drift_loss'] = drift_loss.item()
        loss_dict['curr_sim']   = current_sim_avg.item()

    # --- Part 2: Auxiliary Segmentation Loss (Deep Supervision) ---
    attn_gates = drift_info.get('attn_gates', [])
    
    if gt_mask is not None and len(attn_gates) > 0:
        seg_loss_total = 0
        for i, gate in enumerate(attn_gates):
            # 自动适配尺寸
            if gate.shape[2:] != gt_mask.shape[2:]:
                target = F.interpolate(gt_mask, size=gate.shape[2:], mode='nearest')
            else:
                target = gt_mask
            
            # BCE Loss 监督每一层的 Gate
            bce_loss = F.binary_cross_entropy(gate, target)
            seg_loss_total += bce_loss
            loss_dict[f'aux_seg_L{i}'] = bce_loss.item()
            
        avg_seg_loss = seg_loss_total / len(attn_gates)
        total_loss += avg_seg_loss * lambda_seg
        loss_dict['aux_seg_loss'] = avg_seg_loss.item()

    if 'learned_scale' in drift_info:
        loss_dict['learned_scale'] = drift_info['learned_scale']

    return total_loss, loss_dict
