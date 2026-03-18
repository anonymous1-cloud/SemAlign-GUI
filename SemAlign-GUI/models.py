import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Tuple, Dict


class TemporalRelationModule(nn.Module):
    """
    TRM 模块：处理 Patch 级别的时序交互。
    对应论文描述：4层 Transformer Encoder, 8个注意头, 可学习时间嵌入。
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, num_layers: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 1. 可学习的时间嵌入 E_t=0 和 E_t=1 (对应论文 3.2 节)
        # Shape: [2, 1, hidden_dim]，自动广播到 [B, N, D]
        self.temporal_embeddings = nn.Parameter(torch.randn(2, 1, hidden_dim))

        # 2. 交叉帧自注意力 (MHSA) 序列建模
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, ref_patches: torch.Tensor, tar_patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ref_patches: [B, N, D] (N=196, D=768)
            tar_patches: [B, N, D]
        Returns:
            diff_patches: [B, N, D] 经过对齐交互后的差异特征
        """
        # 注入时间箭头 (Arrow of Time)
        ref_encoded = ref_patches + self.temporal_embeddings[0]
        tar_encoded = tar_patches + self.temporal_embeddings[1]

        # 拼接为联合序列 [B, 2*N, D]
        # 让 ref 和 tar 的 patch 在注意力机制中寻找语义对应点
        temporal_seq = torch.cat([ref_encoded, tar_encoded], dim=1)

        # 核心：跨帧交互 (Global Cross-frame Interactions)
        enhanced_seq = self.transformer_encoder(temporal_seq)

        # 拆分回原状态
        N = ref_patches.shape[1]
        ref_enhanced = enhanced_seq[:, :N, :]
        tar_enhanced = enhanced_seq[:, N:, :]

        # 潜空间语义相减 (Latent Space Subtraction)
        # 此时的相减是基于对齐后的特征，能有效过滤渲染抖动
        diff_patches = tar_enhanced - ref_enhanced
        return diff_patches


class VisualFeatureExtractor(nn.Module):
    """
    视觉特征提取器：负责从 ViT 提取 Patch 特征并生成 Dense Mask。
    """

    def __init__(self, model_path: str, config):
        super().__init__()
        self.config = config

        # 加载预训练 ViT (ViT-B/16)
        vit_config = AutoConfig.from_pretrained(model_path)
        self.vit = AutoModel.from_pretrained(model_path, config=vit_config)

        # 投影层：将 ViT 输出映射到隐藏维度
        self.projection = nn.Linear(vit_config.hidden_size, config.hidden_dim)

        # 时序关系模块 (TRM)
        self.temporal_module = TemporalRelationModule(
            hidden_dim=config.hidden_dim,
            num_heads=8,
            num_layers=4
        )

        # 密集预测解码器 (对应论文中的 Transposed Convolutional Layers)
        # 从 14x14 逐步上采样回 224x224
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(config.hidden_dim, 256, kernel_size=4, stride=2, padding=1),  # 14 -> 28
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 28 -> 56
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 56 -> 112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)  # 112 -> 224
        )

    def forward(self, ref_image: torch.Tensor, tar_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = ref_image.shape[0]
        images = torch.cat([ref_image, tar_image], dim=0)

        # 提取 ViT 特征 [2*B, 197, 768] (包含 CLS)
        outputs = self.vit(pixel_values=images)
        # 丢弃 CLS token，保留全量 Patch 特征 [2*B, 196, 768]
        all_patch_features = outputs.last_hidden_state[:, 1:, :]

        # 分离并投影到 hidden_dim
        all_proj = self.projection(all_patch_features)
        ref_patches = all_proj[:batch_size]
        tar_patches = all_proj[batch_size:]

        # --- TRM 核心流程 ---
        diff_patches = self.temporal_module(ref_patches, tar_patches)

        # 空间特征重建：[B, 196, D] -> [B, D, 14, 14]
        B, N, D = diff_patches.shape
        spatial_features = diff_patches.transpose(1, 2).view(B, D, 14, 14)

        # 生成密集变化 mask (M_pred)
        pred_mask_logits = self.decoder(spatial_features)

        # 全局差异向量 (用于后续 AGF 模块和全局分类)
        # 对 Patch 取平均以获得全局表征
        global_diff = diff_patches.mean(dim=1)

        return global_diff, pred_mask_logits.squeeze(1)


class Stage1VisualModel(nn.Module):
    """
    第一阶段完整模型：整合视觉提取与多任务输出。
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        model_path = f"{config.model_root}/{config.image_model}"

        self.visual_encoder = VisualFeatureExtractor(model_path, config)

        # 全局变化分类器 (p_change)
        self.change_classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, ref_image: torch.Tensor, tar_image: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 1. 提取视觉差异特征与密集 Mask 预测
        diff_features, pred_mask_logits = self.visual_encoder(ref_image, tar_image)

        # 2. 预测全局变化概率 (p_change)
        change_logits = self.change_classifier(diff_features)

        return {
            'diff_features': diff_features,  # 用于 AGF 模块
            'pred_mask_logits': pred_mask_logits,  # 用于 L_mask (BCE Loss)
            'change_logits': change_logits  # 用于 L_cls (Classification Loss)
        }
