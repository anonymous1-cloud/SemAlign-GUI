import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Tuple, Optional
import math


class TemporalRelationModule(nn.Module):
    """时序关系模块：捕捉GUI变化的时间模式"""

    def __init__(self, hidden_dim: int, num_heads: int = 8, temporal_window: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.temporal_window = temporal_window

        # 时序自注意力
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )

        # 位置编码（时序）
        self.temporal_pos_encoding = nn.Parameter(
            torch.randn(1, temporal_window, hidden_dim) * 0.02
        )

        # 时序卷积捕获局部模式
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )

        # 门控机制融合
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        # 归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(0.1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, D] 当前时刻的特征
        Returns:
            enhanced: [B, D] 增强后的特征
        """
        batch_size = features.shape[0]

        # 扩展为时序序列（模拟多时刻）
        seq_length = self.temporal_window

        # 创建时序序列
        if seq_length > 1:
            # 通过不同线性变换模拟多个时刻
            temporal_seq = []
            for i in range(seq_length):
                weight = torch.eye(self.hidden_dim, device=features.device) + \
                         torch.randn_like(torch.eye(self.hidden_dim, device=features.device)) * 0.01
                temporal_seq.append(torch.matmul(features, weight))
            temporal_seq = torch.stack(temporal_seq, dim=1)  # [B, T, D]
        else:
            temporal_seq = features.unsqueeze(1)  # [B, 1, D]

        # 添加位置编码
        temporal_seq = temporal_seq + self.temporal_pos_encoding[:, :temporal_seq.size(1), :]

        # 时序自注意力
        attn_output, attn_weights = self.temporal_attention(
            query=temporal_seq,
            key=temporal_seq,
            value=temporal_seq
        )

        # 残差连接 + 归一化
        temporal_seq = self.norm1(temporal_seq + attn_output)

        # 时序卷积
        conv_input = temporal_seq.transpose(1, 2)  # [B, D, T]
        conv_output = self.temporal_conv(conv_input)  # [B, D, T]
        conv_output = conv_output.transpose(1, 2)  # [B, T, D]

        # 门控融合
        gate_input = torch.cat([temporal_seq, conv_output], dim=-1)
        gate = self.gate(gate_input)
        fused = gate * temporal_seq + (1 - gate) * conv_output

        # FFN
        ffn_output = self.ffn(fused)
        fused = self.norm2(fused + ffn_output)

        # 提取当前时刻特征（最后一个时间步）
        if fused.size(1) > 1:
            current_feat = fused[:, -1, :]  # 取最后一个作为当前时刻
        else:
            current_feat = fused.squeeze(1)

        return current_feat


class VisualFeatureExtractor(nn.Module):
    """视觉特征提取器（集成时序关系模块）"""

    def __init__(self, model_path: str, config):
        super().__init__()
        self.config = config

        # 加载预训练ViT
        vit_config = AutoConfig.from_pretrained(
            model_path,
            hidden_size=1024,
            num_attention_heads=16,
            num_hidden_layers=24,
            intermediate_size=4096
        )

        self.vit = AutoModel.from_pretrained(
            model_path,
            config=vit_config,
            ignore_mismatched_sizes=True
        )

        if config.gradient_checkpointing:
            self.vit.gradient_checkpointing_enable()

        # 投影到统一维度
        self.projection = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # 时序关系模块
        self.temporal_module = TemporalRelationModule(
            hidden_dim=config.hidden_dim,
            num_heads=8,
            temporal_window=3
        )

        # 变化检测头（logits输出）
        self.change_detector = nn.Sequential(
            nn.Conv2d(config.hidden_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, ref_image: torch.Tensor, tar_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = ref_image.shape[0]

        # 提取特征
        images = torch.cat([ref_image, tar_image], dim=0)

        with torch.amp.autocast('cuda', enabled=self.training):
            vit_output = self.vit(pixel_values=images)
            features = vit_output.last_hidden_state[:, 0]  # [CLS] token

        # 分离参考和目标特征
        ref_features = features[:batch_size]
        tar_features = features[batch_size:]

        # 投影到统一维度
        ref_proj = self.projection(ref_features)
        tar_proj = self.projection(tar_features)

        # 时序关系处理
        ref_temporal = self.temporal_module(ref_proj)
        tar_temporal = self.temporal_module(tar_proj)

        # 计算差异特征
        diff_features = tar_temporal - ref_temporal

        # 重建空间特征用于变化检测
        spatial_diff = diff_features.unsqueeze(-1).unsqueeze(-1)  # [B, D, 1, 1]
        spatial_diff = spatial_diff.repeat(1, 1, 14, 14)  # ViT的patch数

        # 生成变化mask（logits）
        change_logits = self.change_detector(spatial_diff)
        change_logits = F.interpolate(change_logits, size=(224, 224), mode='bilinear')

        return diff_features, change_logits.squeeze(1)


class Stage1VisualModel(nn.Module):
    """第一阶段：纯视觉模型（集成时序关系）"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 视觉编码器（含时序关系）
        model_path = f"{config.model_root}/{config.image_model}"
        self.visual_encoder = VisualFeatureExtractor(model_path, config)

        # 变化分类器（logits输出）
        self.change_classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, ref_image: torch.Tensor, tar_image: torch.Tensor):
        # 提取视觉差异特征
        diff_features, pred_logits = self.visual_encoder(ref_image, tar_image)

        # 预测是否有变化（logits）
        change_logits = self.change_classifier(diff_features)

        return {
            'diff_features': diff_features,
            'pred_logits': pred_logits,
            'change_logits': change_logits
        }