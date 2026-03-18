import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModelForCausalLM


# 假设你有一个 config 对象包含 hidden_dim (例如 768)
# from models import Stage1VisualModel # 记得导入你的 Stage1 模型

class ComponentChangeEncoder(nn.Module):
    """
    组件变化编码器 (优化版)
    完美适配：类别ID(1) + 坐标(4) + RGB(3) + 权重(1)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. 组件类型嵌入：14种组件，设20防越界
        self.type_embedding = nn.Embedding(20, 32)

        # 2. 组件特征提取
        # 输入维度: 32(类别) + 4(坐标) + 3(RGB) + 1(权重) = 40
        self.component_encoder = nn.Sequential(
            nn.Linear(40, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, config.hidden_dim // 2)  # 输出 D/2 维
        )

        # 匹配特征的维度：ref(D/2) + tar(D/2) + 相似度得分(1)
        match_dim = 2 * (config.hidden_dim // 2) + 1

        # 3. 引入 Attention Pooling 替代 Mean Pooling (保留结构重点)
        self.attention_pool = nn.Sequential(
            nn.Linear(match_dim, 1),
            nn.Softmax(dim=1)
        )

        # 4. 组件变化特征投影到 hidden_dim
        self.change_projection = nn.Sequential(
            nn.Linear(match_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def encode_components(self, components):
        """
        components shape: [Batch, Num_components (14), 9]
        索引 0: 类别ID
        索引 1-4: (x, y, w, h)
        索引 5-7: RGB
        索引 8: Weight
        """
        # 分离类别
        comp_types = components[:, :, 0].long()
        comp_types = torch.clamp(comp_types, 0, 19)
        type_emb = self.type_embedding(comp_types)  # [B, N, 32]

        # 分离连续特征 [B, N, 8]
        comp_features = components[:, :, 1:9].clone()  # 避免原地修改报错

        # 归一化 RGB (特征切片中的索引 4,5,6 对应原始的 5,6,7)
        comp_features[:, :, 4:7] = comp_features[:, :, 4:7] / 255.0

        # 组合特征: [B, N, 40]
        comp_input = torch.cat([type_emb, comp_features], dim=-1)

        # 编码: [B, N, D/2]
        encoded = self.component_encoder(comp_input)

        # 掩码无效组件 (假设类别 0 是 padding)
        mask = (comp_types > 0).float().unsqueeze(-1)
        encoded = encoded * mask

        return encoded

    def compute_component_changes(self, ref_encoded, tar_encoded):
        """计算并聚合组件变化特征"""
        batch_size, num_components, feat_dim = ref_encoded.shape

        # 计算余弦相似度矩阵寻找对应组件
        ref_norm = F.normalize(ref_encoded, dim=-1)
        tar_norm = F.normalize(tar_encoded, dim=-1)
        similarity = torch.matmul(ref_norm, tar_norm.transpose(1, 2))

        # 找到最佳匹配
        match_scores, match_indices = similarity.max(dim=-1)

        match_features_list = []
        for b in range(batch_size):
            sample_features = []
            for i in range(num_components):
                j = match_indices[b, i].item()
                if j < num_components and match_scores[b, i] > 0.1:
                    # 有效匹配
                    combined = torch.cat([
                        ref_encoded[b, i],
                        tar_encoded[b, j],
                        match_scores[b, i].unsqueeze(0)
                    ])
                else:
                    # 无匹配 (代表可能被删除了)
                    combined = torch.cat([
                        ref_encoded[b, i],
                        torch.zeros_like(ref_encoded[b, i]),
                        torch.tensor([0.0], device=ref_encoded.device)
                    ])
                sample_features.append(combined)

            # [N, match_dim]
            sample_tensor = torch.stack(sample_features, dim=0)
            match_features_list.append(sample_tensor)

        # 堆叠所有样本: [B, N, match_dim]
        match_features = torch.stack(match_features_list, dim=0)

        # 使用 Attention Pooling 聚合，而不是无脑 Mean
        attn_weights = self.attention_pool(match_features)  # [B, N, 1]
        pooled_features = torch.sum(match_features * attn_weights, dim=1)  # [B, match_dim]

        return pooled_features, similarity

    def forward(self, ref_components, tar_components):
        # 1. 编码单帧组件
        ref_encoded = self.encode_components(ref_components)
        tar_encoded = self.encode_components(tar_components)

        # 2. 跨帧匹配与注意力聚合
        change_features, similarity = self.compute_component_changes(ref_encoded, tar_encoded)

        # 3. 投影到统一维度
        projected_features = self.change_projection(change_features)

        return {
            'change_features': projected_features,  # [B, hidden_dim]
            'similarity_matrix': similarity
        }


class TextChangeEncoder(nn.Module):
    """文本编码器: 冻结主干提取语义特征"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 假设通过某种方式加载了 Qwen，这里简写其结构
        # 实际使用中保留你之前的 AutoModelForCausalLM 逻辑即可
        self.projection = nn.Linear(1536, config.hidden_dim)  # 假设 Qwen 特征维度是 1536
        self.post_process = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(0.1)
        )

    def forward(self, text_features_from_qwen):
        # 这里简化了 Qwen 的前向传播，直接处理池化后的输出
        text_features = self.projection(text_features_from_qwen)
        text_features = self.post_process(text_features)
        return {'text_features': text_features}


class MultiModalFusion(nn.Module):
    """
    自适应门控融合模块 (AGF)
    完全对应论文公式: g_m = sigmoid(W_m * H) / sum(...)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 跨模态注意力对齐 (Shared cross-attention layer)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )

        # 论文中的 W_m H 门控生成网络 (映射 3D -> 1 计算置信度)
        self.gate_network = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, 3),  # 输出3个门控值对应 v, t, c
            nn.Sigmoid()
        )

        # MLP 融合统一表示
        self.final_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim * 2),
            nn.LayerNorm(config.hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim)
        )

    def forward(self, visual_features, text_features, component_features):
        # 序列化三种模态进行交叉注意力 [B, 3, D]
        sequence = torch.stack([visual_features, text_features, component_features], dim=1)

        # 软对齐特征
        aligned_seq, _ = self.cross_attention(sequence, sequence, sequence)

        # 取出对齐后的特征
        v_align = aligned_seq[:, 0, :]
        t_align = aligned_seq[:, 1, :]
        c_align = aligned_seq[:, 2, :]

        # 级联特征 H = [F_diff || F_text || F_comp] -> [B, 3D]
        H = torch.cat([v_align, t_align, c_align], dim=-1)

        # 计算归一化门控值
        raw_gates = self.gate_network(H)  # [B, 3]
        gate_sum = raw_gates.sum(dim=1, keepdim=True) + 1e-8
        normalized_gates = raw_gates / gate_sum  # [B, 3]

        g_v, g_t, g_c = normalized_gates[:, 0:1], normalized_gates[:, 1:2], normalized_gates[:, 2:3]

        # 门控加权
        v_weighted = g_v * v_align
        t_weighted = g_t * t_align
        c_weighted = g_c * c_align

        # 投影融合
        fused = torch.cat([v_weighted, t_weighted, c_weighted], dim=-1)
        unified_representation = self.final_mlp(fused)

        return {
            'fused_features': unified_representation,
            'gate_values': normalized_gates
        }


class Stage2AlignmentModel(nn.Module):
    """
    第二阶段：完整对齐模型
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. 实例化各个编码器
        self.text_encoder = TextChangeEncoder(config)
        self.component_encoder = ComponentChangeEncoder(config)
        self.fusion_module = MultiModalFusion(config)

        # 假设这里传入的是 Stage1 出来的全局池化特征 (通常是 [B, D])
        # 如果 Stage1 输出维度不是 config.hidden_dim，需要加一个投影层
        self.visual_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(inplace=True)
        )

        # 2. 对比学习特征投影头 (优化空间，对应公式 margin loss)
        self.contrastive_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, 256),
            nn.LayerNorm(256)
        )

        # 3. 可学习温度参数 (论文中提到初始化为 0.07)
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

    def forward(self, visual_diff_features, raw_text_features, ref_components, tar_components):
        # 1. 视觉特征投影
        v_feat = self.visual_projection(visual_diff_features)

        # 2. 文本特征提取 (冻结主干)
        t_outputs = self.text_encoder(raw_text_features)
        t_feat = t_outputs['text_features']

        # 3. 结构组件提取
        c_outputs = self.component_encoder(ref_components, tar_components)
        c_feat = c_outputs['change_features']

        # 4. AGF 三模态自适应门控融合
        fusion_outputs = self.fusion_module(v_feat, t_feat, c_feat)
        fused_features = fusion_outputs['fused_features']

        # 5. 用于 Triplet Margin Loss 计算的对齐特征
        contrastive_features = self.contrastive_proj(fused_features)

        return {
            'fused_features': fused_features,
            'contrastive_features': contrastive_features,
            'gate_values': fusion_outputs['gate_values'],
            'temperature': self.temperature
        }
