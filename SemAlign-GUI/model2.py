import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# 导入第一阶段模型以进行级联封装
from models import Stage1VisualModel


class ComponentChangeEncoder(nn.Module):
    """组件变化编码器：处理界面结构图信息"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.type_embedding = nn.Embedding(20, 32)

        self.component_encoder = nn.Sequential(
            nn.Linear(40, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, config.hidden_dim // 2)
        )

        match_dim = 2 * (config.hidden_dim // 2) + 1

        self.attention_pool = nn.Sequential(
            nn.Linear(match_dim, 1),
            nn.Softmax(dim=1)
        )

        self.change_projection = nn.Sequential(
            nn.Linear(match_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def encode_components(self, components):
        if components is None:
            return None
        comp_types = components[:, :, 0].long()
        comp_types = torch.clamp(comp_types, 0, 19)
        type_emb = self.type_embedding(comp_types)

        comp_features = components[:, :, 1:9].clone()
        comp_features[:, :, 4:7] = comp_features[:, :, 4:7] / 255.0

        comp_input = torch.cat([type_emb, comp_features], dim=-1)
        encoded = self.component_encoder(comp_input)

        mask = (comp_types > 0).float().unsqueeze(-1)
        return encoded * mask

    def compute_component_changes(self, ref_encoded, tar_encoded):
        batch_size, num_components, _ = ref_encoded.shape
        ref_norm = F.normalize(ref_encoded, dim=-1)
        tar_norm = F.normalize(tar_encoded, dim=-1)
        similarity = torch.matmul(ref_norm, tar_norm.transpose(1, 2))

        match_scores, match_indices = similarity.max(dim=-1)

        match_features_list = []
        for b in range(batch_size):
            sample_features = []
            for i in range(num_components):
                j = match_indices[b, i].item()
                if j < num_components and match_scores[b, i] > 0.1:
                    combined = torch.cat([ref_encoded[b, i], tar_encoded[b, j], match_scores[b, i].unsqueeze(0)])
                else:
                    combined = torch.cat([ref_encoded[b, i], torch.zeros_like(ref_encoded[b, i]),
                                          torch.tensor([0.0], device=ref_encoded.device)])
                sample_features.append(combined)
            match_features_list.append(torch.stack(sample_features, dim=0))

        match_features = torch.stack(match_features_list, dim=0)
        attn_weights = self.attention_pool(match_features)
        pooled_features = torch.sum(match_features * attn_weights, dim=1)

        return pooled_features, similarity

    def forward(self, ref_components, tar_components):
        if ref_components is None or tar_components is None:
            # Fallback for empty batch
            device = ref_components.device if ref_components is not None else torch.device('cuda')
            B = ref_components.shape[0] if ref_components is not None else 1
            return {
                'change_features': torch.zeros(B, self.config.hidden_dim, device=device),
                'similarity_matrix': torch.zeros(B, 14, 14, device=device)
            }

        ref_encoded = self.encode_components(ref_components)
        tar_encoded = self.encode_components(tar_components)

        change_features, similarity = self.compute_component_changes(ref_encoded, tar_encoded)
        projected_features = self.change_projection(change_features)

        return {
            'change_features': projected_features,
            'similarity_matrix': similarity
        }


class TextChangeEncoder(nn.Module):
    """文本变化编码器：处理语言模态"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        # 简单 Embedding 层模拟 LLM 特征提取 (实际训练时可对接 Qwen 的特征)
        self.vocab_size = 50000
        self.embedding = nn.Embedding(self.vocab_size, config.hidden_dim)
        self.post_process = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(0.1)
        )

    def forward(self, text_tokens):
        emb = self.embedding(text_tokens)
        # 简化处理：对 seq_len 维度取均值
        text_features = emb.mean(dim=1)
        return self.post_process(text_features)


class MultiModalFusion(nn.Module):
    """自适应门控融合模块 (AGF)"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim, num_heads=8, batch_first=True, dropout=0.1
        )
        self.gate_network = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, 3),
            nn.Sigmoid()
        )
        self.final_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim * 2),
            nn.LayerNorm(config.hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim)
        )

    def forward(self, visual_features, text_features, component_features):
        sequence = torch.stack([visual_features, text_features, component_features], dim=1)
        aligned_seq, _ = self.cross_attention(sequence, sequence, sequence)

        v_align = aligned_seq[:, 0, :]
        t_align = aligned_seq[:, 1, :]
        c_align = aligned_seq[:, 2, :]

        H = torch.cat([v_align, t_align, c_align], dim=-1)
        raw_gates = self.gate_network(H)

        gate_sum = raw_gates.sum(dim=1, keepdim=True) + 1e-8
        normalized_gates = raw_gates / gate_sum

        g_v, g_t, g_c = normalized_gates[:, 0:1], normalized_gates[:, 1:2], normalized_gates[:, 2:3]

        v_weighted = g_v * v_align
        t_weighted = g_t * t_align
        c_weighted = g_c * c_align

        fused = torch.cat([v_weighted, t_weighted, c_weighted], dim=-1)
        unified_representation = self.final_mlp(fused)

        return {
            'fused_features': unified_representation,
            'gate_values': normalized_gates
        }


class Stage2AlignmentModel(nn.Module):
    """第二阶段：完整对齐模型"""

    def __init__(self, stage1_checkpoint, config, use_components=True):
        super().__init__()
        self.config = config

        # 1. 内部实例化并冻结 Stage 1 视觉模型
        self.stage1 = Stage1VisualModel(config)
        if stage1_checkpoint and os.path.exists(stage1_checkpoint):
            state_dict = torch.load(stage1_checkpoint, map_location='cpu')['model_state_dict']
            self.stage1.load_state_dict(state_dict)
        for param in self.stage1.parameters():
            param.requires_grad = False

        # 2. 视觉特征重投影
        self.visual_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(inplace=True)
        )

        # 3. 各模态编码器与融合模块
        self.text_encoder = TextChangeEncoder(config)
        self.component_encoder = ComponentChangeEncoder(config)
        self.fusion_module = MultiModalFusion(config)

        # 4. 对齐头部与对比学习投影
        self.contrastive_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, 256),
            nn.LayerNorm(256)
        )
        self.alignment_head = nn.Linear(config.hidden_dim, 1)
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

    def forward(self, ref_image, tar_image, text_tokens, ref_components, tar_components):
        # 1. 视觉特征流
        with torch.no_grad():
            stage1_outputs = self.stage1(ref_image, tar_image)
            visual_diff_features = stage1_outputs['diff_features']

        v_feat = self.visual_projection(visual_diff_features)

        # 2. 文本特征流
        t_feat = self.text_encoder(text_tokens)

        # 3. 组件特征流
        c_outputs = self.component_encoder(ref_components, tar_components)
        c_feat = c_outputs['change_features']

        # 4. AGF 融合
        fusion_outputs = self.fusion_module(v_feat, t_feat, c_feat)
        fused_features = fusion_outputs['fused_features']

        # 5. 返回字典必须严丝合缝对齐 Stage 2 Trainer 的需求
        return {
            'visual_features': v_feat,  # 用于算 v-t 相似度
            'text_features': t_feat,  # 用于算 v-t 相似度
            'component_outputs': c_outputs,  # 用于算 c-v, c-t 相似度
            'fusion_outputs': fusion_outputs,  # 用于算 gate_entropy 损失
            'fused_features': fused_features,
            'contrastive_features': self.contrastive_proj(fused_features),
            'alignment_logits': self.alignment_head(fused_features),
            'temperature': self.temperature
        }
