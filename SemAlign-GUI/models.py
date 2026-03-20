import torch
import torch.nn as nn
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

        # 可学习的时间嵌入 E_t=0 和 E_t=1
        self.temporal_embeddings = nn.Parameter(torch.randn(2, 1, hidden_dim))

        # 交叉帧自注意力序列建模
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
        # 注入时间箭头
        ref_encoded = ref_patches + self.temporal_embeddings[0]
        tar_encoded = tar_patches + self.temporal_embeddings[1]

        # 拼接为联合序列 [B, 2*N, D]
        temporal_seq = torch.cat([ref_encoded, tar_encoded], dim=1)

        # 跨帧交互
        enhanced_seq = self.transformer_encoder(temporal_seq)

        # 拆分回原状态并相减，提取对齐后的差异
        N = ref_patches.shape[1]
        diff_patches = enhanced_seq[:, N:, :] - enhanced_seq[:, :N, :]
        return diff_patches


class VisualFeatureExtractor(nn.Module):
    """视觉特征提取器：负责从 ViT 提取 Patch 特征并生成 Dense Mask。"""
    def __init__(self, model_path: str, config):
        super().__init__()
        self.config = config

        # 加载预训练 ViT (这里默认用 google/vit-base-patch16-224-in21k，可替换为你的本地路径)
        try:
            self.vit = AutoModel.from_pretrained(model_path)
            vit_hidden_size = self.vit.config.hidden_size
        except:
            print(f"[*] 未找到本地模型 {model_path}，回退到在线默认配置...")
            vit_config = AutoConfig.from_pretrained("google/vit-base-patch16-224-in21k")
            self.vit = AutoModel.from_config(vit_config)
            vit_hidden_size = vit_config.hidden_size

        # 投影层：将 ViT 输出映射到隐藏维度
        self.projection = nn.Linear(vit_hidden_size, config.hidden_dim)

        # 时序关系模块 (TRM)
        self.temporal_module = TemporalRelationModule(
            hidden_dim=config.hidden_dim,
            num_heads=8,
            num_layers=4
        )

        # 密集预测解码器 (从 14x14 逐步上采样回 224x224)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(config.hidden_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, ref_image: torch.Tensor, tar_image: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = ref_image.shape[0]
        images = torch.cat([ref_image, tar_image], dim=0)

        # 提取 ViT 特征 [2*B, 197, 768]
        outputs = self.vit(pixel_values=images)
        # 丢弃 CLS token，保留全量 Patch 特征 [2*B, 196, 768]
        all_patch_features = outputs.last_hidden_state[:, 1:, :]

        all_proj = self.projection(all_patch_features)
        ref_patches = all_proj[:batch_size]
        tar_patches = all_proj[batch_size:]

        # --- TRM 核心流程 --- [B, 196, 768]
        diff_patches = self.temporal_module(ref_patches, tar_patches)

        # 空间特征重建：[B, 196, D] -> [B, D, 14, 14]
        B, N, D = diff_patches.shape
        spatial_features = diff_patches.transpose(1, 2).view(B, D, 14, 14)

        # 生成密集变化 mask (M_pred) [B, 224, 224]
        pred_mask_logits = self.decoder(spatial_features).squeeze(1)

        # 全局差异向量 [B, 768]
        global_diff = diff_patches.mean(dim=1)

        return {
            'global_diff': global_diff,
            'diff_patches': diff_patches,       # 为 Stage 3 预留！
            'pred_mask_logits': pred_mask_logits
        }


class Stage1VisualModel(nn.Module):
    """第一阶段完整模型：整合视觉提取与多任务输出。"""
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
        vis_outputs = self.visual_encoder(ref_image, tar_image)
        change_logits = self.change_classifier(vis_outputs['global_diff'])

        # 返回字典的 Key 严格对齐 train1.py 中的 compute_loss
        return {
            'diff_features': vis_outputs['global_diff'],   # Stage 2 需要
            'diff_patches': vis_outputs['diff_patches'],   # Stage 3 需要
            'pred_logits': vis_outputs['pred_mask_logits'],# Trainer 需要
            'change_logits': change_logits                 # Trainer 需要
        }
