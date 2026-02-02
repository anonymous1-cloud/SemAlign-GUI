"""
第三阶段：短语级对比学习模型
建立短语-token ⟷ 图像-patch的双向细粒度对应
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import math
from typing import List, Dict, Tuple, Optional, Any
from model2 import Stage2AlignmentModel


class PhraseParser:
    """从differ_text解析短语结构"""

    def __init__(self):
        self.component_keywords = [
            'TextView', 'ImageView', 'Button', 'EditText', 'WebView',
            'View', 'CheckBox', 'RadioButton', 'Switch', 'ToggleButton',
            'Widget', 'SwitchMain', 'SwitchSlider'
        ]

    def parse(self, text: str) -> List[Dict]:
        """
        解析differ_text为结构化短语
        示例: 'Removed TextView from position (0, 9, 144, 239)'
        """
        if not text:
            return []

        phrases = []
        # 按分号分割不同变化
        segments = [s.strip() for s in text.split(';') if s.strip()]

        for segment in segments:
            phrase = self._parse_segment(segment)
            if phrase:
                phrases.append(phrase)

        return phrases

    def _parse_segment(self, segment: str) -> Optional[Dict]:
        """解析单个变化片段"""
        try:
            # 检查变化类型
            if segment.startswith('Added'):
                change_type = 'addition'
                # 提取: Added TextView at position (6, 136, 138, 239)
                pattern = r'Added (\w+) at position \(([\d., ]+)\)'
            elif segment.startswith('Removed'):
                change_type = 'removal'
                # 提取: Removed TextView from position (0, 9, 144, 239)
                pattern = r'Removed (\w+) from position \(([\d., ]+)\)'
            elif ' to ' in segment and ' from ' in segment:
                change_type = 'movement'
                # 提取: TextView from (0, 9, 144, 239) to (23, 116, 121, 141)
                pattern = r'(\w+) from \(([\d., ]+)\) to \(([\d., ]+)\)'
            else:
                return None

            match = re.search(pattern, segment)
            if not match:
                return None

            if change_type in ['addition', 'removal']:
                component_type = match.group(1)
                bbox_str = match.group(2)
                bbox = [float(x.strip()) for x in bbox_str.split(',')]

                return {
                    'type': change_type,
                    'component': component_type,
                    'bbox': bbox,
                    'text': segment,
                    'is_movement': False
                }
            else:  # movement
                component_type = match.group(1)
                from_bbox = [float(x.strip()) for x in match.group(2).split(',')]
                to_bbox = [float(x.strip()) for x in match.group(3).split(',')]

                return {
                    'type': change_type,
                    'component': component_type,
                    'from_bbox': from_bbox,
                    'to_bbox': to_bbox,
                    'text': segment,
                    'is_movement': True
                }

        except Exception as e:
            print(f"解析短语失败: {segment}, 错误: {e}")
            return None

    def batch_parse(self, texts: List[str]) -> List[List[Dict]]:
        """批量解析"""
        return [self.parse(text) for text in texts]


class PhraseEncoder(nn.Module):
    """短语编码器 - 修复版本：输出768维特征"""

    def __init__(self, hidden_dim: int = 768, dropout: float = 0.1):
        super().__init__()

        # 组件类型嵌入
        self.component_embedding = nn.Embedding(20, 64)  # 13种类型 + 余量

        # 变化类型嵌入
        self.change_type_embedding = nn.Embedding(4, 32)  # addition, removal, movement, unknown

        # 位置编码 (归一化坐标)
        self.pos_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 64)
        )

        # 短语融合网络
        self.fusion = nn.Sequential(
            nn.Linear(64 + 32 + 64, 256),  # 组件 + 变化类型 + 位置 = 160 -> 256
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),  # 256 -> hidden_dim
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # 短语投影头 - 输出到隐藏维度
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_dim),  # 输出隐藏维度
            nn.LayerNorm(hidden_dim)
        )

        print(f"PhraseEncoder初始化: 输出维度 = {hidden_dim}")

    def forward(self, phrases_batch: List[List[Dict]], device: str = 'cuda'):
        """
        编码短语列表

        Args:
            phrases_batch: 批次化的短语列表 [B, variable_num_phrases]

        Returns:
            phrase_features: 短语特征列表 [total_phrases, hidden_dim]
            batch_indices: 每个短语对应的批次索引
        """
        batch_size = len(phrases_batch)
        all_features = []
        batch_indices = []

        # 组件类型映射
        component_map = {
            'TextView': 1, 'ImageView': 2, 'Button': 3,
            'EditText': 4, 'WebView': 5, 'View': 6,
            'CheckBox': 7, 'RadioButton': 8, 'Switch': 9,
            'ToggleButton': 10, 'Widget': 11, 'SwitchMain': 12,
            'SwitchSlider': 13, 'unknown': 0
        }

        # 变化类型映射
        change_type_map = {'addition': 0, 'removal': 1, 'movement': 2, 'unknown': 3}

        for batch_idx, phrases in enumerate(phrases_batch):
            for phrase in phrases:
                # 组件类型编码
                comp_type = phrase.get('component', 'unknown')
                comp_id = component_map.get(comp_type, 0)
                comp_emb = self.component_embedding(torch.tensor([comp_id], device=device))

                # 变化类型编码
                change_type = phrase.get('type', 'unknown')
                change_id = change_type_map.get(change_type, 3)
                change_emb = self.change_type_embedding(torch.tensor([change_id], device=device))

                # 位置编码
                if phrase.get('is_movement', False):
                    # 对于移动，使用目标位置
                    bbox = phrase.get('to_bbox', [0, 0, 0, 0])
                else:
                    bbox = phrase.get('bbox', [0, 0, 0, 0])

                # 确保bbox是4个值
                if len(bbox) != 4:
                    bbox = [0, 0, 0, 0]

                # 归一化到[0, 1]
                bbox_norm = [max(0.0, min(1.0, x)) for x in bbox]
                pos_emb = self.pos_encoder(torch.tensor(bbox_norm, device=device).unsqueeze(0))

                # 融合特征
                fused = self.fusion(torch.cat([comp_emb, change_emb, pos_emb], dim=-1))

                # 投影到目标维度
                phrase_feat = self.projection(fused)

                all_features.append(phrase_feat.squeeze(0))
                batch_indices.append(batch_idx)

        if not all_features:
            # 返回空张量
            return torch.zeros((0, 768), device=device), torch.zeros(0, device=device, dtype=torch.long)

        return torch.stack(all_features, dim=0), torch.tensor(batch_indices, device=device)


class PatchEncoder(nn.Module):
    """图像Patch编码器"""

    def __init__(self, visual_dim: int = 768, hidden_dim: int = 128):
        super().__init__()

        # ViT的patch特征维度是768，grid是14x14
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(visual_dim, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )

        # Patch位置编码
        self.patch_pos_encoding = nn.Parameter(
            torch.randn(1, hidden_dim, 14, 14) * 0.02
        )

        # 自注意力增强
        self.patch_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        编码视觉patch特征

        Args:
            visual_features: [B, 768, 14, 14]  ViT patch特征

        Returns:
            patch_features: [B, 196, 128]  展平的patch特征
        """
        batch_size = visual_features.shape[0]

        # 编码patch特征
        patch_feat = self.patch_encoder(visual_features)  # [B, 128, 14, 14]

        # 添加位置编码
        patch_feat = patch_feat + self.patch_pos_encoding

        # 展平为序列
        patch_seq = patch_feat.flatten(2).permute(0, 2, 1)  # [B, 196, 128]

        # 自注意力增强
        attended, _ = self.patch_attention(patch_seq, patch_seq, patch_seq)
        patch_seq = patch_seq + 0.1 * attended

        return patch_seq


class PhrasePatchContrastive(nn.Module):
    """短语-Patch对比学习模块"""

    def __init__(self, hidden_dim: int = 128, temperature: float = 0.07):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.temperature = nn.Parameter(torch.tensor([temperature]))

        # 对比学习投影头
        self.phrase_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.patch_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 短语-Patch匹配网络
        self.matching_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def compute_contrastive_loss(self, features1: torch.Tensor, features2: torch.Tensor,
                                 batch_indices: torch.Tensor) -> torch.Tensor:
        """
        计算对比学习损失 - 修复版本

        Args:
            features1: [N, D] 特征1（通常是短语特征）
            features2: [M, D] 特征2（通常是patch特征）
            batch_indices: [N] 特征1对应的批次索引

        Returns:
            对比损失
        """
        # 归一化特征
        feat1_norm = F.normalize(features1, dim=-1)
        feat2_norm = F.normalize(features2, dim=-1)

        # 计算相似度矩阵
        sim_matrix = torch.matmul(feat1_norm, feat2_norm.T) / self.temperature  # [N, M]

        # 创建正样本标签
        batch_size = batch_indices.max().item() + 1 if len(batch_indices) > 0 else 1

        # 获取每个批次中的样本数量（对于patch特征，每个批次有196个patch）
        # 假设每个批次有固定数量的patch（196个）
        patches_per_batch = features2.shape[0] // batch_size if batch_size > 0 else features2.shape[0]

        # 创建正样本掩码
        pos_mask = torch.zeros((len(features1), len(features2)),
                               device=features1.device, dtype=torch.bool)

        for i, batch_idx in enumerate(batch_indices):
            # 对于第i个短语，同一批次的所有patch都是正样本
            start_idx = batch_idx * patches_per_batch
            end_idx = (batch_idx + 1) * patches_per_batch

            # 确保索引在范围内
            if start_idx < len(features2) and end_idx <= len(features2):
                pos_mask[i, start_idx:end_idx] = True

        # 计算InfoNCE损失
        pos_sim = sim_matrix[pos_mask]
        neg_sim = sim_matrix[~pos_mask]

        if len(pos_sim) == 0:
            return torch.tensor(0.0, device=features1.device)

        # 对每个query计算损失
        losses = []
        for i in range(len(features1)):
            # 正样本相似度
            pos_mask_i = pos_mask[i]
            if pos_mask_i.sum() == 0:
                continue

            pos_i = sim_matrix[i][pos_mask_i]

            # 负样本相似度（包括其他样本和batch内的负样本）
            neg_i = sim_matrix[i][~pos_mask_i]

            # 计算logits
            logits = torch.cat([pos_i.unsqueeze(0), neg_i.unsqueeze(0)], dim=1)

            # 标签：第一个是正样本
            labels = torch.zeros(1, dtype=torch.long, device=features1.device)

            # 交叉熵损失
            loss_i = F.cross_entropy(logits, labels)
            losses.append(loss_i)

        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=features1.device)

    def compute_correspondence(self, phrase_features: torch.Tensor,
                               patch_features: torch.Tensor,
                               batch_indices: torch.Tensor) -> List[Dict]:
        """
        计算短语-Patch对应关系

        Returns:
            每个短语对应的top-k匹配patch
        """
        batch_size = patch_features.shape[0]
        phrase_features_norm = F.normalize(phrase_features, dim=-1)

        correspondences = []

        for i in range(len(phrase_features)):
            batch_idx = batch_indices[i].item()
            if batch_idx >= batch_size:
                continue

            # 获取对应批次的patch特征
            batch_patches = patch_features[batch_idx]  # [196, D]
            batch_patches_norm = F.normalize(batch_patches, dim=-1)

            # 计算相似度
            sim_scores = torch.matmul(phrase_features_norm[i:i + 1], batch_patches_norm.T)  # [1, 196]

            # 获取top-k匹配
            top_k = 10
            top_scores, top_indices = torch.topk(sim_scores.squeeze(0), k=min(top_k, len(sim_scores)))

            # 转换为patch坐标 (14x14网格)
            patch_coords = []
            for idx in top_indices:
                patch_idx = idx.item()
                h = patch_idx // 14
                w = patch_idx % 14
                # 转换为像素坐标 (224x224)
                x1 = w * 16  # 224/14 = 16
                y1 = h * 16
                x2 = x1 + 16
                y2 = y1 + 16
                patch_coords.append({
                    'patch_idx': patch_idx,
                    'bbox': [x1, y1, x2, y2],
                    'score': top_scores[idx == top_indices].item()
                })

            correspondences.append({
                'phrase_idx': i,
                'batch_idx': batch_idx,
                'top_patches': patch_coords,
                'max_score': top_scores.max().item()
            })

        return correspondences

    def forward(self, phrase_features: torch.Tensor, patch_features: torch.Tensor,
                batch_indices: torch.Tensor):
        """
        前向传播

        Returns:
            losses: 各种损失
            correspondences: 短语-Patch对应关系
        """
        # 投影特征
        phrase_proj = self.phrase_proj(phrase_features)
        patch_proj = self.patch_proj(patch_features)

        # 获取patch特征的批次索引（每个批次有196个patch）
        batch_size = patch_features.shape[0]
        patches_per_batch = 196

        # 创建patch的批次索引
        patch_batch_indices = torch.arange(batch_size, device=patch_features.device)
        patch_batch_indices = patch_batch_indices.repeat_interleave(patches_per_batch)

        # 展平patch特征
        patch_features_flat = patch_proj.flatten(0, 1)  # [B*196, D]

        # 计算短语到patch的对比损失
        loss_phrase_to_patch = self.compute_contrastive_loss(
            phrase_proj, patch_features_flat, batch_indices
        )

        # 计算patch到短语的对比损失
        loss_patch_to_phrase = self.compute_contrastive_loss(
            patch_features_flat, phrase_proj, patch_batch_indices
        )

        # 计算短语-Patch匹配分数
        correspondences = self.compute_correspondence(
            phrase_proj, patch_proj, batch_indices
        )

        return {
            'loss_phrase_to_patch': loss_phrase_to_patch,
            'loss_patch_to_phrase': loss_patch_to_phrase,
            'total_contrastive_loss': loss_phrase_to_patch + loss_patch_to_phrase,
            'correspondences': correspondences,
            'phrase_features': phrase_proj,
            'patch_features': patch_proj
        }

class Stage3PhraseContrastiveModel(nn.Module):
    """第三阶段：短语级对比学习模型"""

    def __init__(self, stage2_checkpoint: str, config):
        super().__init__()
        self.config = config

        print(f"\n初始化第三阶段模型...")
        print(f"加载Stage2检查点: {stage2_checkpoint}")

        # ============ 1. 加载Stage2模型 ============
        self.stage2_model = Stage2AlignmentModel(
            stage1_checkpoint="",  # 不需要stage1
            config=config,  # 使用传入的config
            use_components=True
        )

        try:
            if stage2_checkpoint and os.path.exists(stage2_checkpoint):
                checkpoint = torch.load(stage2_checkpoint, map_location='cpu', weights_only=False)
                state_dict = checkpoint.get('model_state_dict', checkpoint)

                # 先加载模型结构，然后过滤不匹配的键
                model_dict = self.stage2_model.state_dict()

                # 过滤掉形状不匹配的键
                filtered_state_dict = {}
                for k, v in state_dict.items():
                    if k in model_dict and model_dict[k].shape == v.shape:
                        filtered_state_dict[k] = v
                    elif k in model_dict:
                        print(f"跳过形状不匹配的键: {k}, 检查点形状: {v.shape}, 模型形状: {model_dict[k].shape}")
                    else:
                        print(f"跳过不存在的键: {k}")

                # 加载过滤后的权重
                model_dict.update(filtered_state_dict)
                self.stage2_model.load_state_dict(model_dict, strict=False)

                print(f"✅ Stage2模型加载成功")
                print(f"   加载了 {len(filtered_state_dict)}/{len(state_dict)} 个参数")

            else:
                print(f"⚠️ Stage2检查点不存在，使用随机初始化")
        except Exception as e:
            print(f"⚠️ 加载Stage2模型失败: {e}")
            import traceback
            traceback.print_exc()

        # 冻结Stage2模型
        for param in self.stage2_model.parameters():
            param.requires_grad = False

        # ============ 2. 短语解析器 ============
        self.phrase_parser = PhraseParser()

        # ============ 3. 短语编码器 ============
        # 使用Stage2的隐藏维度
        self.hidden_dim = config.hidden_dim
        print(f"使用隐藏维度: {self.hidden_dim}")
        self.phrase_encoder = PhraseEncoder(hidden_dim=self.hidden_dim, dropout=0.1)

        # ============ 4. Patch编码器 ============
        self.patch_encoder = PatchEncoder(visual_dim=768, hidden_dim=128)

        # ============ 5. 短语-Patch对比学习 ============
        self.contrastive_module = PhrasePatchContrastive(hidden_dim=128, temperature=0.07)

        # ============ 6. 短语特征投影 ============
        # 将短语特征从hidden_dim投影到128维对比学习空间
        self.phrase_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),  # 输入hidden_dim（768）维
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),  # 输出128维对比学习空间
            nn.LayerNorm(128)
        )

        print(f"\n✅ 第三阶段模型初始化完成:")
        print(f"   短语编码维度: {self.hidden_dim} -> 128 (对比学习空间)")
        print(f"   Patch编码维度: 768 -> 128")
        print(f"   对比学习温度: 0.07")
        print(f"   可训练参数: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def extract_patch_features(self, ref_image: torch.Tensor, tar_image: torch.Tensor):
        """提取真正的ViT patch特征"""
        batch_size = ref_image.shape[0]

        with torch.no_grad():
            try:
                # 正确的访问路径：Stage1VisualModel -> VisualFeatureExtractor -> ViT
                visual_extractor = self.stage2_model.visual_encoder.visual_encoder

                if hasattr(visual_extractor, 'vit'):
                    images = torch.cat([ref_image, tar_image], dim=0)

                    # 获取ViT输出
                    vit_outputs = visual_extractor.vit(pixel_values=images)

                    # 获取所有token特征（包括CLS和patch tokens）
                    all_tokens = vit_outputs.last_hidden_state  # [2B, 197, 1024]

                    # 分离参考和目标，去除CLS token（第一个token）
                    ref_patches = all_tokens[:batch_size, 1:, :]  # [B, 196, 1024]
                    tar_patches = all_tokens[batch_size:, 1:, :]  # [B, 196, 1024]

                    # 计算patch级别的差异
                    diff_patches = tar_patches - ref_patches  # [B, 196, 1024]

                    # 重塑为 [B, 1024, 14, 14]
                    patch_features = diff_patches.permute(0, 2, 1).reshape(batch_size, 1024, 14, 14)

                    # 降维到768（适配PatchEncoder）
                    # 使用平均池化
                    patch_features = F.adaptive_avg_pool2d(patch_features, (14, 14))

                    # 确保维度为768
                    if patch_features.shape[1] > 768:
                        patch_features = patch_features[:, :768, :, :]
                    elif patch_features.shape[1] < 768:
                        padding = torch.zeros(batch_size, 768 - patch_features.shape[1], 14, 14,
                                              device=patch_features.device)
                        patch_features = torch.cat([patch_features, padding], dim=1)

                    return patch_features
                else:
                    raise AttributeError("VisualFeatureExtractor没有vit属性")

            except Exception as e:
                print(f"❌ 提取ViT patch特征失败: {e}")

                # 备用方案：使用Stage2的diff_features
                visual_outputs = self.stage2_model.visual_encoder(ref_image, tar_image)
                diff_features = visual_outputs.get('diff_features')

                batch_size, hidden_dim = diff_features.shape

                # 创建patch网格特征
                patch_features = diff_features.unsqueeze(-1).unsqueeze(-1)
                patch_features = patch_features.repeat(1, 1, 14, 14)

                # 确保维度为768
                if hidden_dim != 768:
                    if hidden_dim > 768:
                        patch_features = patch_features[:, :768, :, :]
                    else:
                        padding = torch.zeros(batch_size, 768 - hidden_dim, 14, 14,
                                              device=patch_features.device)
                        patch_features = torch.cat([patch_features, padding], dim=1)

                return patch_features

    def forward(self, ref_image: torch.Tensor, tar_image: torch.Tensor,
                text_tokens: torch.Tensor, ref_components: torch.Tensor,
                tar_components: torch.Tensor, differ_texts: List[str]):
        """
        前向传播
        """
        try:
            batch_size = ref_image.shape[0]
            device = ref_image.device
            device = ref_image.device
            tar_image = tar_image.to(device)
            text_tokens = text_tokens.to(device)
            ref_components = ref_components.to(device)
            tar_components = tar_components.to(device)

            # ============ 1. 解析短语 ============
            phrases_batch = self.phrase_parser.batch_parse(differ_texts)

            # ============ 2. 提取patch特征 ============
            patch_features = self.extract_patch_features(ref_image, tar_image)

            # ============ 3. 编码短语 ============
            phrase_features, phrase_batch_indices = self.phrase_encoder(
                phrases_batch, device=device
            )

            # ============ 4. 投影短语特征到对比学习空间 ============
            phrase_features_proj = self.phrase_projection(phrase_features)

            # ============ 5. 编码patch ============
            encoded_patches = self.patch_encoder(patch_features)

            # ============ 6. 短语-Patch对比学习 ============
            if len(phrase_features_proj) > 0:
                contrastive_outputs = self.contrastive_module(
                    phrase_features_proj, encoded_patches, phrase_batch_indices
                )
            else:
                contrastive_outputs = {
                    'loss_phrase_to_patch': torch.tensor(0.0, device=device),
                    'loss_patch_to_phrase': torch.tensor(0.0, device=device),
                    'total_contrastive_loss': torch.tensor(0.0, device=device),
                    'correspondences': [],
                    'phrase_features': phrase_features_proj,
                    'patch_features': encoded_patches
                }

            # ============ 7. 构建输出 ============
            outputs = {
                'patch_features': patch_features,
                'encoded_patches': encoded_patches,
                'phrase_features': phrase_features_proj,
                'phrase_batch_indices': phrase_batch_indices,
                'original_phrase_features': phrase_features,
                **contrastive_outputs,
                'parsed_phrases': phrases_batch,
                'num_phrases': len(phrase_features_proj),
                'stage2_features': None
            }

            # 可选：获取Stage2特征
            with torch.no_grad():
                try:
                    stage2_outputs = self.stage2_model(
                        ref_image, tar_image, text_tokens, ref_components, tar_components
                    )
                    outputs['stage2_features'] = {
                        'fused_features': stage2_outputs.get('fused_features'),
                        'alignment_scores': stage2_outputs.get('alignment_scores')
                    }
                except Exception as e:
                    outputs['stage2_features'] = None

            return outputs

        except Exception as e:
            print(f"❌ 第三阶段模型前向传播错误: {e}")
            import traceback
            traceback.print_exc()

            device = ref_image.device
            return {
                'patch_features': torch.zeros(batch_size, 768, 14, 14, device=device),
                'encoded_patches': torch.zeros(batch_size, 196, 128, device=device),
                'phrase_features': torch.zeros(0, 128, device=device),
                'phrase_batch_indices': torch.zeros(0, device=device, dtype=torch.long),
                'loss_phrase_to_patch': torch.tensor(0.0, device=device),
                'loss_patch_to_phrase': torch.tensor(0.0, device=device),
                'total_contrastive_loss': torch.tensor(0.0, device=device),
                'correspondences': [],
                'parsed_phrases': [],
                'num_phrases': 0,
                'stage2_features': None
            }


# 导出模型
__all__ = [
    'PhraseParser',
    'PhraseEncoder',
    'PatchEncoder',
    'PhrasePatchContrastive',
    'Stage3PhraseContrastiveModel'
]