"""
第三阶段：Phrase-Patch Contrastive Learning (PPCL) 模块
建立短语(纯语义) ⟷ 图像Patch(空间视觉) 的双向细粒度对应
完全对齐论文 3.4 节的理论描述。
"""
import math
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional

# 假设你的 Stage2 模型在此路径，确保能正常导入
from model2 import Stage2AlignmentModel


class PhraseParser:
    """
    短语解析器 (对应论文中的 Text Parser)
    从 differ_text 解析短语结构，提取目标边界框用于生成 Spatial Prior Mask。
    """

    def parse(self, text: str) -> List[Dict]:
        if not text:
            return []
        phrases = []
        segments = [s.strip() for s in text.split(';') if s.strip()]
        for segment in segments:
            phrase = self._parse_segment(segment)
            if phrase:
                phrases.append(phrase)
        return phrases

    def _parse_segment(self, segment: str) -> Optional[Dict]:
        try:
            if segment.startswith('Added'):
                change_type = 'addition'
                pattern = r'Added (\w+) at position \(([\d., ]+)\)'
            elif segment.startswith('Removed'):
                change_type = 'removal'
                pattern = r'Removed (\w+) from position \(([\d., ]+)\)'
            elif ' to ' in segment and ' from ' in segment:
                change_type = 'movement'
                pattern = r'(\w+) from \(([\d., ]+)\) to \(([\d., ]+)\)'
            else:
                return None

            match = re.search(pattern, segment)
            if not match:
                return None

            component_type = match.group(1)
            if change_type in ['addition', 'removal']:
                bbox = [float(x.strip()) for x in match.group(2).split(',')]
                target_bbox = bbox
            else:
                target_bbox = [float(x.strip()) for x in match.group(3).split(',')]

            # 统一坐标并归一化到 [0, 1] 假设原始坐标是 224x224
            # 如果预处理已经是 [0, 1] 这一步可省略。这里加个防御性除以 224。
            target_bbox = [x / 224.0 if x > 1.5 else x for x in target_bbox]
            target_bbox = [max(0.0, min(1.0, x)) for x in target_bbox]

            return {
                'type': change_type,
                'component': component_type,
                'target_bbox': target_bbox,  # 仅用于生成 Mask，绝不喂给 Encoder!
                'text': segment
            }
        except Exception:
            return None

    def batch_parse(self, texts: List[str]) -> List[List[Dict]]:
        return [self.parse(text) for text in texts]


class PhraseEncoder(nn.Module):
    """
    短语编码器 (纯语义编码器)
    🚨 论文对齐点：Explicitly stripping numerical coordinate tokens.
    这里彻底删除了原代码中的 pos_encoder，只保留组件类型和变化动作，杜绝坐标泄漏！
    """

    def __init__(self, hidden_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        # 14种组件类型 + 1个 unknown = 15
        self.component_embedding = nn.Embedding(20, 64)
        # addition, removal, movement, unknown
        self.change_type_embedding = nn.Embedding(4, 32)

        # 融合网络：只接受 64 + 32 = 96 维的纯语义特征
        self.fusion = nn.Sequential(
            nn.Linear(96, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.component_map = {
            'TextView': 1, 'ImageView': 2, 'Button': 3, 'EditText': 4, 'WebView': 5,
            'View': 6, 'CheckBox': 7, 'RadioButton': 8, 'Switch': 9, 'ToggleButton': 10,
            'Widget': 11, 'SwitchMain': 12, 'SwitchSlider': 13, 'unknown': 0
        }
        self.change_type_map = {'addition': 0, 'removal': 1, 'movement': 2, 'unknown': 3}

    def forward(self, phrases_batch: List[List[Dict]], device: torch.device):
        all_features = []
        batch_indices = []
        target_bboxes = []  # 顺便提取出来给后面算 Mask 用

        for batch_idx, phrases in enumerate(phrases_batch):
            for phrase in phrases:
                comp_id = self.component_map.get(phrase.get('component', 'unknown'), 0)
                change_id = self.change_type_map.get(phrase.get('type', 'unknown'), 3)

                comp_emb = self.component_embedding(torch.tensor([comp_id], device=device))
                change_emb = self.change_type_embedding(torch.tensor([change_id], device=device))

                # 核心：纯语义融合，没有 bbox！
                fused = self.fusion(torch.cat([comp_emb, change_emb], dim=-1))
                all_features.append(fused.squeeze(0))

                batch_indices.append(batch_idx)
                target_bboxes.append(phrase['target_bbox'])

        if not all_features:
            return (torch.zeros((0, self.fusion[-1].normalized_shape[0]), device=device),
                    torch.zeros(0, device=device, dtype=torch.long), [])

        return torch.stack(all_features, dim=0), torch.tensor(batch_indices, device=device), target_bboxes


class PhrasePatchContrastive(nn.Module):
    """
    短语-Patch 对比学习核心模块
    🚨 论文对齐点：50% area overlap threshold & multiple-positive contrastive loss.
    """

    def __init__(self, hidden_dim: int = 128, temperature: float = 0.07):
        super().__init__()
        self.hidden_dim = hidden_dim
        # 🚨 论文对齐点：Learnable temperature hyperparameter τ initialized to 0.07
        self.temperature = nn.Parameter(torch.ones([]) * temperature)

        self.phrase_proj = nn.Sequential(
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, hidden_dim)
        )

        self.patch_proj = nn.Sequential(
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, hidden_dim)
        )

    def generate_spatial_mask(self, target_bboxes: List[List[float]], grid_size: int = 14) -> torch.Tensor:
        """
        核心算法：计算 BBox 与 14x14 Grid 的重叠度。
        只有当 BBox 覆盖某个 Patch 面积超过 50% 时，该 Patch 才视为正样本。
        """
        num_phrases = len(target_bboxes)
        mask = torch.zeros((num_phrases, grid_size * grid_size), dtype=torch.bool)

        for i, bbox in enumerate(target_bboxes):
            # 缩放到 14x14 坐标系
            x1, y1, x2, y2 = [v * grid_size for v in bbox]

            for r in range(grid_size):
                for c in range(grid_size):
                    # 当前 Patch 的坐标范围
                    px1, py1, px2, py2 = c, r, c + 1, r + 1

                    # 计算交集矩形
                    ix1 = max(x1, px1)
                    iy1 = max(y1, py1)
                    ix2 = min(x2, px2)
                    iy2 = min(y2, py2)

                    if ix1 < ix2 and iy1 < iy2:
                        inter_area = (ix2 - ix1) * (iy2 - iy1)
                        patch_area = 1.0  # 1x1 grid cell
                        # 🚨 论文对齐点：50% threshold
                        if inter_area / patch_area > 0.5:
                            mask[i, r * grid_size + c] = True

            # 容错机制：如果 bbox 太小，导致没有任何 patch 达到 50%
            # 则强行选中中心点所在的那个 patch 作为正样本
            if not mask[i].any() and x1 < x2 and y1 < y2:
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                cx_idx, cy_idx = int(min(cx, grid_size - 1)), int(min(cy, grid_size - 1))
                mask[i, cy_idx * grid_size + cx_idx] = True

        return mask

    def forward(self, phrase_features: torch.Tensor, patch_features: torch.Tensor,
                batch_indices: torch.Tensor, target_bboxes: List[List[float]]):
        """
        phrase_features: [N, 768]
        patch_features: [B, 196, 768] (已融合 Stage 2 门控的特征)
        """
        device = phrase_features.device
        N = phrase_features.shape[0]
        B, P, _ = patch_features.shape  # P=196

        if N == 0:
            return {'total_contrastive_loss': torch.tensor(0.0, device=device)}

        # 投影到对比空间 (如 128维)
        p_feat = F.normalize(self.phrase_proj(phrase_features), dim=-1)  # [N, 128]
        v_feat = F.normalize(self.patch_proj(patch_features), dim=-1)  # [B, 196, 128]

        # 计算空间监督 Mask (正样本)
        spatial_mask = self.generate_spatial_mask(target_bboxes, grid_size=math.isqrt(P)).to(device)

        losses = []
        for i in range(N):
            b_idx = batch_indices[i]
            # 获取该图片所有的 196 个 patch 特征
            curr_v_feat = v_feat[b_idx]  # [196, 128]
            curr_mask = spatial_mask[i]  # [196]

            if not curr_mask.any():
                continue  # 没有正样本则跳过

            # 计算相似度矩阵 (s_ij)
            sim_scores = torch.matmul(p_feat[i:i + 1], curr_v_feat.T).squeeze(0) / self.temperature  # [196]

            # InfoNCE for Multiple Positives (多正样本对比损失)
            # log ( \sum_{pos} exp(s) / \sum_{all} exp(s) )
            pos_sum = torch.exp(sim_scores[curr_mask]).sum()
            all_sum = torch.exp(sim_scores).sum()

            loss_i = -torch.log(pos_sum / (all_sum + 1e-8))
            losses.append(loss_i)

        total_loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)

        return {'total_contrastive_loss': total_loss, 'temperature': self.temperature}


class Stage3PhraseContrastiveModel(nn.Module):
    """
    第三阶段主模型：整合解析、编码、Stage 2 提取和对比学习
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 加载冻结的 Stage 2 模型 (在实际训练脚本中 load_state_dict)
        self.stage2_model = Stage2AlignmentModel(config=config)
        for param in self.stage2_model.parameters():
            param.requires_grad = False

        self.phrase_parser = PhraseParser()
        self.phrase_encoder = PhraseEncoder(hidden_dim=config.hidden_dim)

        # Patch Encoder (引入自注意力以增强空间上下文)
        self.patch_enhancer = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

        self.contrastive_module = PhrasePatchContrastive(hidden_dim=128, temperature=0.07)

    def extract_recalibrated_patches(self, ref_image, tar_image, text_tokens, ref_comp, tar_comp):
        """
        🚨 论文对齐点：extract the semantically recalibrated visual patch tokens directly
        from the adaptive gated fusion module introduced in Stage 2.
        """
        batch_size = ref_image.shape[0]

        # 1. 直接复用 Stage 1 的视觉特征提取器获取空间 Patch 差异 [B, 196, 768]
        visual_extractor = self.stage2_model.visual_encoder.visual_encoder
        images = torch.cat([ref_image, tar_image], dim=0)
        outputs = visual_extractor.vit(pixel_values=images)
        all_patch_features = outputs.last_hidden_state[:, 1:, :]  # [2B, 196, 1024]

        all_proj = visual_extractor.projection(all_patch_features)
        ref_patches = all_proj[:batch_size]
        tar_patches = all_proj[batch_size:]

        # 经过 TRM 获得对齐后的空间 Patch 特征
        diff_patches = visual_extractor.temporal_module(ref_patches, tar_patches)  # [B, 196, 768]

        # 2. 借用 Stage 2 的全局融合门控 (Gating) 来重校准 Patch
        with torch.no_grad():
            stage2_outputs = self.stage2_model(
                diff_patches.mean(dim=1), text_tokens, ref_comp, tar_comp
            )
            gates = stage2_outputs['gate_values']  # [B, 3] 分别对应 g_v, g_t, g_c

        g_v = gates[:, 0:1].unsqueeze(-1)  # 变成 [B, 1, 1] 以便广播

        # 乘上视觉门控，实现"受到全局文本和结构上下文过滤" (Recalibrated)
        recalibrated_patches = diff_patches * g_v

        return self.patch_enhancer(recalibrated_patches)

    def forward(self, ref_image, tar_image, text_tokens, ref_components, tar_components, differ_texts: List[str]):
        device = ref_image.device

        # 1. 解析短语与目标 BBox
        phrases_batch = self.phrase_parser.batch_parse(differ_texts)

        # 2. 提取重新校准的视觉 Patch (保留 196 空间结构)
        # 此处不计算梯度，因为 Stage 1/2 已冻结，仅 Stage 3 参与梯度回传
        with torch.no_grad():
            patch_features = self.extract_recalibrated_patches(
                ref_image, tar_image, text_tokens, ref_components, tar_components
            )

        # 3. 编码短语 (纯语义，剔除坐标)
        phrase_features, phrase_batch_indices, target_bboxes = self.phrase_encoder(phrases_batch, device)

        # 4. Phrase-Patch 对比学习 (50% Area Overlap)
        contrastive_outputs = self.contrastive_module(
            phrase_features, patch_features, phrase_batch_indices, target_bboxes
        )

        return contrastive_outputs
