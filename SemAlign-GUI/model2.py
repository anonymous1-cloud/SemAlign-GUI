import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional
import os
from models import Stage1VisualModel
from transformers import AutoTokenizer, AutoModelForCausalLM


class ComponentChangeEncoder(nn.Module):
    """组件变化编码器 - 修复索引越界问题"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 组件类型嵌入
        self.type_embedding = nn.Embedding(20, 32)

        # 组件特征提取 - 输出维度匹配hidden_dim的一半
        self.component_encoder = nn.Sequential(
            nn.Linear(32 + 12, 256),  # 类型(32) + 特征(12) = 44
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, config.hidden_dim // 2)  # 输出384维
        )

        # 组件变化特征投影到hidden_dim
        self.change_projection = nn.Sequential(
            nn.Linear(2 * (config.hidden_dim // 2) + 1, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 变化类型预测
        self.change_type_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # 4种变化类型
        )

        print(f"ComponentChangeEncoder初始化:")
        print(f"  组件特征输出: {config.hidden_dim // 2}")
        print(f"  变化特征投影: {2 * (config.hidden_dim // 2) + 1} -> {config.hidden_dim}")

    def encode_components(self, components):
        """编码组件特征"""
        batch_size, num_components, _ = components.shape

        # 分离特征
        comp_types = components[:, :, 0].long()
        comp_types = torch.clamp(comp_types, 0, 19)  # 限制在[0, 19]范围内
        comp_features = components[:, :, 1:13]

        # 类型嵌入
        type_emb = self.type_embedding(comp_types)

        # 组合特征
        comp_input = torch.cat([type_emb, comp_features], dim=-1)

        # 编码
        encoded = self.component_encoder(comp_input)

        # 掩码无效组件
        mask = (comp_types > 0).float().unsqueeze(-1)
        encoded = encoded * mask

        return encoded

    def compute_component_changes(self, ref_encoded, tar_encoded):
        """计算组件变化特征"""
        batch_size, num_components, feat_dim = ref_encoded.shape

        # 计算相似度矩阵
        ref_norm = F.normalize(ref_encoded, dim=-1)
        tar_norm = F.normalize(tar_encoded, dim=-1)
        similarity = torch.matmul(ref_norm, tar_norm.transpose(1, 2))

        # 找到最佳匹配
        match_scores, match_indices = similarity.max(dim=-1)

        # 创建匹配特征
        match_features_list = []

        for b in range(batch_size):
            sample_features = []
            for i in range(num_components):
                j = match_indices[b, i].item()
                if j < num_components and match_scores[b, i] > 0.1:
                    # 有有效匹配
                    combined = torch.cat([
                        ref_encoded[b, i],
                        tar_encoded[b, j],
                        match_scores[b, i].unsqueeze(0)
                    ])
                else:
                    # 无匹配
                    combined = torch.cat([
                        ref_encoded[b, i],
                        torch.zeros_like(ref_encoded[b, i]),
                        torch.tensor([0.0], device=ref_encoded.device)
                    ])
                sample_features.append(combined)

            # 聚合每个样本的特征
            if sample_features:
                sample_tensor = torch.stack(sample_features, dim=0)
                sample_pooled = sample_tensor.mean(dim=0)
            else:
                sample_pooled = torch.cat([
                    torch.zeros(feat_dim, device=ref_encoded.device),
                    torch.zeros(feat_dim, device=ref_encoded.device),
                    torch.tensor([0.0], device=ref_encoded.device)
                ])

            match_features_list.append(sample_pooled)

        # 堆叠所有样本
        match_features = torch.stack(match_features_list, dim=0)

        return match_features, similarity

    def forward(self, ref_components, tar_components):
        """前向传播"""
        try:
            # 编码组件
            ref_encoded = self.encode_components(ref_components)
            tar_encoded = self.encode_components(tar_components)

            # 计算变化特征
            change_features, similarity = self.compute_component_changes(ref_encoded, tar_encoded)

            # 投影到hidden_dim
            projected_features = self.change_projection(change_features)

            # 预测变化类型
            change_type_logits = self.change_type_predictor(projected_features)

            return {
                'change_features': projected_features,
                'similarity_matrix': similarity,
                'change_type_logits': change_type_logits,
                'ref_encoded': ref_encoded,
                'tar_encoded': tar_encoded
            }
        except Exception as e:
            print(f"组件编码器前向传播错误: {e}")
            batch_size = ref_components.shape[0]
            return {
                'change_features': torch.zeros(batch_size, self.config.hidden_dim,
                                               device=ref_components.device),
                'similarity_matrix': torch.zeros(batch_size, 20, 20,
                                                 device=ref_components.device),
                'change_type_logits': torch.zeros(batch_size, 4,
                                                  device=ref_components.device),
                'ref_encoded': torch.zeros(batch_size, 20, self.config.hidden_dim // 2,
                                           device=ref_components.device),
                'tar_encoded': torch.zeros(batch_size, 20, self.config.hidden_dim // 2,
                                           device=ref_components.device)
            }


class TextChangeEncoder(nn.Module):
    """使用Qwen2.5-1.5B的文本编码器"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 模型路径配置
        model_root = "/home/common-dir/models"
        text_model = "Qwen2.5-1.5B"
        model_path = os.path.join(model_root, text_model)

        print(f"加载Qwen2.5-1.5B模型: {model_path}")
        print(f"目标维度: {config.hidden_dim}")

        try:
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )

            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 加载模型
            self.qwen_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,  # 使用float32提高稳定性
                device_map="auto",
                trust_remote_code=True,
                output_hidden_states=True
            )

            print(f"✅ Qwen2.5-1.5B加载成功")
            print(f"   模型隐藏维度: {self.qwen_model.config.hidden_size}")

            # Qwen2.5的隐藏维度
            qwen_hidden_size = self.qwen_model.config.hidden_size

            # 投影层：将Qwen2.5的输出投影到config.hidden_dim
            self.projection = nn.Linear(qwen_hidden_size, config.hidden_dim)

            # 后处理层
            self.post_process = nn.Sequential(
                nn.LayerNorm(config.hidden_dim),
                nn.Dropout(0.1)
            )

            self.use_qwen = True

        except Exception as e:
            print(f"❌ 加载Qwen2.5失败: {e}")
            print("使用简单的文本编码器作为回退")

            # 回退方案：简单的Embedding编码器
            vocab_size = 1500
            self.embedding = nn.Embedding(vocab_size, 256)
            self.text_encoder = nn.Sequential(
                nn.Linear(256, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.ReLU()
            )
            self.use_qwen = False

        print(f"TextChangeEncoder初始化完成:")
        print(f"  使用Qwen2.5: {self.use_qwen}")
        if self.use_qwen:
            print(f"  投影层: {qwen_hidden_size} -> {config.hidden_dim}")

    def forward(self, text_tokens):
        """编码文本"""
        batch_size = text_tokens.shape[0]

        # 回退方案
        if not self.use_qwen:
            # 确保token索引在有效范围内
            text_tokens = torch.clamp(text_tokens, 0, self.embedding.num_embeddings - 1)
            embedded = self.embedding(text_tokens)
            pooled = embedded.mean(dim=1)
            text_features = self.text_encoder(pooled)

            return {
                'text_features': text_features,
                'text_embeddings': embedded
            }

        # Qwen2.5方案
        try:
            # 创建attention mask
            attention_mask = (text_tokens != self.tokenizer.pad_token_id).long()

            # 确保设备一致
            device = next(self.projection.parameters()).device
            text_tokens = text_tokens.to(device)
            attention_mask = attention_mask.to(device)

            # 获取Qwen2.5的隐藏状态
            with torch.no_grad():
                outputs = self.qwen_model(
                    input_ids=text_tokens,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )

            # 获取最后一层的隐藏状态
            last_hidden_state = outputs.hidden_states[-1]

            # 使用attention mask进行加权平均池化
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask

            # 投影到目标维度
            text_features = self.projection(pooled_output)
            text_features = self.post_process(text_features)

            return {
                'text_features': text_features,
                'qwen_hidden_states': outputs.hidden_states,
                'attention_mask': attention_mask
            }

        except Exception as e:
            print(f"Qwen2.5前向传播错误: {e}")
            device = next(self.projection.parameters()).device
            return {
                'text_features': torch.zeros(batch_size, self.config.hidden_dim, device=device),
                'qwen_hidden_states': None,
                'attention_mask': None
            }


class MultiModalFusion(nn.Module):
    """多模态融合模块"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 特征投影层 - 所有模态投影到hidden_dim
        self.visual_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.text_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.component_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 跨模态注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )

        # 门控机制
        self.gate_visual = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, 1),
            nn.Sigmoid()
        )

        self.gate_text = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, 1),
            nn.Sigmoid()
        )

        self.gate_component = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, 1),
            nn.Sigmoid()
        )

        # 最终融合
        self.final_fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim * 2),
            nn.LayerNorm(config.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim)
        )

        print(f"MultiModalFusion初始化:")
        print(f"  注意力头数: 8")
        print(f"  融合输出: {config.hidden_dim}")

    def forward(self, visual_features, text_features, component_features):
        """多模态融合"""
        batch_size = visual_features.shape[0]

        # 投影特征
        v_proj = self.visual_projection(visual_features)
        t_proj = self.text_projection(text_features)
        c_proj = self.component_projection(component_features)

        # 准备序列用于跨模态注意力
        sequence = torch.stack([v_proj, t_proj, c_proj], dim=1)

        # 跨模态注意力
        attended, attn_weights = self.cross_attention(
            sequence, sequence, sequence
        )

        # 组合特征用于门控
        combined_features = torch.cat([v_proj, t_proj, c_proj], dim=-1)

        # 计算每个模态的门控权重
        gate_v = self.gate_visual(combined_features)  # [B, 1]
        gate_t = self.gate_text(combined_features)  # [B, 1]
        gate_c = self.gate_component(combined_features)  # [B, 1]

        # 归一化门控权重
        gate_sum = gate_v + gate_t + gate_c + 1e-8
        gate_v = gate_v / gate_sum
        gate_t = gate_t / gate_sum
        gate_c = gate_c / gate_sum

        # 加权融合
        v_weighted = gate_v * v_proj
        t_weighted = gate_t * t_proj
        c_weighted = gate_c * c_proj

        # 组合加权特征
        fused = torch.cat([v_weighted, t_weighted, c_weighted], dim=-1)

        # 最终融合
        output = self.final_fusion(fused)

        return {
            'fused_features': output,
            'attention_weights': attn_weights,
            'gate_values': torch.cat([gate_v, gate_t, gate_c], dim=1),
            'visual_features': v_proj,
            'text_features': t_proj,
            'component_features': c_proj
        }


class Stage2AlignmentModel(nn.Module):
    """第二阶段：视觉-文本-组件对齐模型"""

    def __init__(self, stage1_checkpoint: str, config, use_components: bool = True):
        super().__init__()
        self.config = config
        self.use_components = use_components

        print(f"加载Stage1检查点: {stage1_checkpoint}")

        # ============ 1. 视觉编码器 ============
        self.visual_encoder = Stage1VisualModel(config)

        # 尝试加载权重
        try:
            if os.path.exists(stage1_checkpoint):
                checkpoint = torch.load(stage1_checkpoint, map_location='cpu', weights_only=False)

                # 提取状态字典
                state_dict = checkpoint.get('model_state_dict',
                                            checkpoint.get('state_dict', checkpoint))

                # 清理状态字典
                cleaned_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('visual_encoder.'):
                        cleaned_state_dict[k.replace('visual_encoder.', '', 1)] = v
                    elif k.startswith('module.'):
                        cleaned_state_dict[k.replace('module.', '', 1)] = v
                    else:
                        cleaned_state_dict[k] = v

                # 加载模型
                missing_keys, unexpected_keys = self.visual_encoder.load_state_dict(
                    cleaned_state_dict, strict=False
                )

                print("✅ Stage1视觉编码器加载成功")
                if missing_keys:
                    print(f"   缺失的键: {len(missing_keys)}个")
                if unexpected_keys:
                    print(f"   意外的键: {len(unexpected_keys)}个")

            else:
                print(f"⚠️ 检查点文件不存在: {stage1_checkpoint}")
                print("使用随机初始化的视觉编码器")

        except Exception as e:
            print(f"⚠️ 加载Stage1模型失败: {e}")
            print("使用随机初始化的视觉编码器")

        # 冻结视觉编码器
        for param in self.visual_encoder.parameters():
            param.requires_grad = False

        # ============ 2. 文本编码器 ============
        self.text_encoder = TextChangeEncoder(config)

        # ============ 3. 组件编码器 ============
        if use_components:
            self.component_encoder = ComponentChangeEncoder(config)
        else:
            self.component_encoder = None

        # ============ 4. 多模态融合 ============
        self.fusion_module = MultiModalFusion(config)

        # ============ 5. 对齐头 ============
        self.alignment_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # ============ 6. 对比学习投影头 ============
        self.contrastive_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128)
        )

        # ============ 7. 视觉投影层 ============
        # 关键修复：从Stage1输出维度投影到hidden_dim
        # 我们需要先检查Stage1的实际输出维度
        print(f"配置视觉投影层...")
        print(f"  config.visual_dim: {config.visual_dim}")
        print(f"  config.hidden_dim: {config.hidden_dim}")

        # 尝试获取Stage1的输出维度
        test_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            test_output = self.visual_encoder(test_input, test_input)
            visual_output_dim = test_output.get('diff_features', test_input).shape[-1]
            print(f"  Stage1实际输出维度: {visual_output_dim}")

        # 使用实际检测到的维度
        self.visual_projection = nn.Sequential(
            nn.Linear(visual_output_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        print(f"  视觉投影: {visual_output_dim} -> {config.hidden_dim}")

        # ============ 8. 温度参数 ============
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

        print(f"\n✅ Stage2AlignmentModel初始化完成:")
        print(f"   视觉特征: {visual_output_dim} -> {config.hidden_dim}")
        print(f"   文本特征: Qwen2.5 -> {config.hidden_dim}")
        print(f"   组件特征: -> {config.hidden_dim}")
        print(f"   最终输出: -> {config.hidden_dim}")
        print(f"   使用组件: {use_components}")

    def forward(self, ref_image, tar_image, text_tokens, ref_components=None, tar_components=None):
        """前向传播"""
        try:
            batch_size = ref_image.shape[0]

            # ============ 1. 视觉特征提取 ============
            original_mode = self.visual_encoder.training
            self.visual_encoder.eval()

            with torch.no_grad():
                visual_outputs = self.visual_encoder(ref_image, tar_image)

                # 调试：检查视觉输出
                print(f"\n调试信息 - Step {self.training}:")


                # 提取视觉特征
                if 'diff_features' in visual_outputs:
                    visual_features = visual_outputs['diff_features']

                else:
                    # 尝试其他可能的键
                    for key in visual_outputs:
                        if isinstance(visual_outputs[key], torch.Tensor):
                            print(f"  {key}形状: {visual_outputs[key].shape}")

                    # 使用第一个张量作为视觉特征
                    visual_features = None
                    for key in visual_outputs:
                        if isinstance(visual_outputs[key], torch.Tensor) and visual_outputs[key].dim() == 2:
                            visual_features = visual_outputs[key]
                            print(f"  使用 {key} 作为视觉特征")
                            break

                    if visual_features is None:
                        # 创建虚拟特征
                        visual_features = torch.randn(batch_size, 1024, device=ref_image.device)
                        print(f"  创建虚拟视觉特征: {visual_features.shape}")

            if original_mode:
                self.visual_encoder.train()

            # 投影视觉特征

            visual_features = self.visual_projection(visual_features)


            # ============ 2. 文本特征提取 ============
            text_outputs = self.text_encoder(text_tokens)
            text_features = text_outputs['text_features']


            # ============ 3. 组件特征提取 ============
            component_features = None
            component_outputs = None

            if self.use_components and ref_components is not None and tar_components is not None:
                component_outputs = self.component_encoder(ref_components, tar_components)
                component_features = component_outputs['change_features']


            # ============ 4. 多模态融合 ============
            if component_features is not None:
                fusion_inputs = (visual_features, text_features, component_features)
            else:
                dummy_component = torch.zeros(
                    batch_size, self.config.hidden_dim,
                    device=visual_features.device
                )
                fusion_inputs = (visual_features, text_features, dummy_component)
                print(f"  使用虚拟组件特征: {dummy_component.shape}")

            fusion_outputs = self.fusion_module(*fusion_inputs)
            fused_features = fusion_outputs['fused_features']


            # ============ 5. 对齐预测 ============
            alignment_logits = self.alignment_head(fused_features)
            alignment_scores = torch.sigmoid(alignment_logits)


            # ============ 6. 对比学习特征 ============
            contrastive_features = self.contrastive_proj(fused_features)


            # ============ 7. 构建输出 ============
            outputs = {
                'visual_features': visual_features,
                'text_features': text_features,
                'fused_features': fused_features,
                'contrastive_features': contrastive_features,
                'alignment_logits': alignment_logits,
                'alignment_scores': alignment_scores,
                'fusion_outputs': fusion_outputs,
                'text_outputs': text_outputs,
                'visual_outputs': visual_outputs,
                'temperature': self.temperature
            }

            if component_outputs is not None:
                outputs['component_outputs'] = component_outputs

            return outputs

        except Exception as e:
            print(f"模型前向传播错误: {e}")
            import traceback
            traceback.print_exc()

            batch_size = ref_image.shape[0]
            return {
                'visual_features': torch.zeros(batch_size, self.config.hidden_dim,
                                               device=ref_image.device),
                'text_features': torch.zeros(batch_size, self.config.hidden_dim,
                                             device=ref_image.device),
                'fused_features': torch.zeros(batch_size, self.config.hidden_dim,
                                              device=ref_image.device),
                'contrastive_features': torch.zeros(batch_size, 128,
                                                    device=ref_image.device),
                'alignment_scores': torch.zeros(batch_size, 1,
                                                device=ref_image.device),
                'fusion_outputs': {},
                'text_outputs': {},
                'visual_outputs': {},
                'temperature': self.temperature
            }


# 导出所有类
__all__ = [
    'ComponentChangeEncoder',
    'TextChangeEncoder',
    'MultiModalFusion',
    'Stage2AlignmentModel'
]