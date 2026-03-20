import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
import os

# 导入第二阶段模型 (Stage2内部已经封装了Stage1)
from model2 import Stage2AlignmentModel


class PhraseParser:
    """短语解析器：从 differ_text 提取纯语义信息和空间先验 (BBox)"""

    def parse(self, text: str) -> List[Dict]:
        if not text: return []
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
            if not match: return None

            component_type = match.group(1)
            if change_type in ['addition', 'removal']:
                target_bbox = [float(x.strip()) for x in match.group(2).split(',')]
            else:
                target_bbox = [float(x.strip()) for x in match.group(3).split(',')]

            # 归一化到 [0, 1] 范围
            target_bbox = [x / 224.0 if x > 1.5 else x for x in target_bbox]
            target_bbox = [max(0.0, min(1.0, x)) for x in target_bbox]

            return {
                'type': change_type,
                'component': component_type,
                'target_bbox': target_bbox,
                'text': segment
            }
        except Exception:
            return None

    def batch_parse(self, texts: List[str]) -> List[List[Dict]]:
        return [self.parse(text) for text in texts]


class PhraseEncoder(nn.Module):
    """短语编码器：剔除坐标信息，只编码纯语义类型和动作"""

    def __init__(self, hidden_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.component_embedding = nn.Embedding(20, 64)
        self.change_type_embedding = nn.Embedding(4, 32)

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
        all_features, batch_indices, target_bboxes = [], [], []

        for batch_idx, phrases in enumerate(phrases_batch):
            for phrase in phrases:
                comp_id = self.component_map.get(phrase.get('component', 'unknown'), 0)
                change_id = self.change_type_map.get(phrase.get('type', 'unknown'), 3)

                comp_emb = self.component_embedding(torch.tensor([comp_id], device=device))
                change_emb = self.change_type_embedding(torch.tensor([change_id], device=device))

                fused = self.fusion(torch.cat([comp_emb, change_emb], dim=-1))
                all_features.append(fused.squeeze(0))

                batch_indices.append(batch_idx)
                target_bboxes.append(phrase['target_bbox'])

        if not all_features:
            return (torch.zeros((0, self.fusion[-1].normalized_shape[0]), device=device),
                    torch.zeros(0, device=device, dtype=torch.long), [])

        return torch.stack(all_features, dim=0), torch.tensor(batch_indices, device=device), target_bboxes


class PhrasePatchContrastive(nn.Module):
    """短语-Patch 双向对比学习模块 (PPCL)"""

    def __init__(self, hidden_dim: int = 128, temperature: float = 0.07):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = nn.Parameter(torch.ones([]) * temperature)

        self.phrase_proj = nn.Sequential(
            nn.Linear(768, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, hidden_dim)
        )
        self.patch_proj = nn.Sequential(
            nn.Linear(768, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, hidden_dim)
        )

    def generate_spatial_mask(self, target_bboxes: List[List[float]], grid_size: int = 14) -> torch.Tensor:
        num_phrases = len(target_bboxes)
        mask = torch.zeros((num_phrases, grid_size * grid_size), dtype=torch.bool)

        for i, bbox in enumerate(target_bboxes):
            x1, y1, x2, y2 = [v * grid_size for v in bbox]
            for r in range(grid_size):
                for c in range(grid_size):
                    px1, py1, px2, py2 = c, r, c + 1, r + 1
                    ix1, iy1 = max(x1, px1), max(y1, py1)
                    ix2, iy2 = min(x2, px2), min(y2, py2)

                    if ix1 < ix2 and iy1 < iy2:
                        inter_area = (ix2 - ix1) * (iy2 - iy1)
                        if inter_area / 1.0 > 0.5:  # 50% 面积重叠阈值
                            mask[i, r * grid_size + c] = True

            if not mask[i].any() and x1 < x2 and y1 < y2:
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                cx_idx, cy_idx = int(min(cx, grid_size - 1)), int(min(cy, grid_size - 1))
                mask[i, cy_idx * grid_size + cx_idx] = True

        return mask

    def forward(self, phrase_features, patch_features, batch_indices, target_bboxes):
        device = phrase_features.device
        N = phrase_features.shape[0]
        B, P, _ = patch_features.shape

        if N == 0:
            return {'total_contrastive_loss': torch.tensor(0.0, device=device)}

        p_feat = F.normalize(self.phrase_proj(phrase_features), dim=-1)
        v_feat = F.normalize(self.patch_proj(patch_features), dim=-1)
        spatial_mask = self.generate_spatial_mask(target_bboxes, grid_size=math.isqrt(P)).to(device)

        loss_p2v, loss_v2p, correspondences = [], [], []

        for i in range(N):
            b_idx = batch_indices[i]
            curr_v_feat = v_feat[b_idx]
            curr_mask = spatial_mask[i]

            if not curr_mask.any(): continue

            # [1, 128] x [128, 196] -> [196]
            sim_scores = torch.matmul(p_feat[i:i + 1], curr_v_feat.T).squeeze(0) / self.temperature

            pos_sum = torch.exp(sim_scores[curr_mask]).sum()
            all_sum = torch.exp(sim_scores).sum()

            # Phrase -> Patch
            loss_p2v.append(-torch.log(pos_sum / (all_sum + 1e-8)))
            # Patch -> Phrase
            loss_v2p.append(-torch.log(torch.exp(sim_scores[curr_mask]).mean() / (all_sum + 1e-8)))

            max_score, max_idx = torch.max(sim_scores, dim=0)
            correspondences.append({
                'batch_idx': b_idx.item(), 'phrase_idx': i,
                'max_score': max_score.item() * self.temperature.item(),
                'top_patches': [{'bbox': [0, 0, 0, 0]}]
            })

        l_p2v = torch.stack(loss_p2v).mean() if loss_p2v else torch.tensor(0.0, device=device)
        l_v2p = torch.stack(loss_v2p).mean() if loss_v2p else torch.tensor(0.0, device=device)

        return {
            'total_contrastive_loss': l_p2v + l_v2p,
            'loss_phrase_to_patch': l_p2v,
            'loss_patch_to_phrase': l_v2p,
            'correspondences': correspondences
        }


class Stage3PhraseContrastiveModel(nn.Module):
    """第三阶段：完整的主模型。"""

    def __init__(self, stage2_checkpoint, config):
        super().__init__()
        self.config = config

        # 内部封装并彻底冻结 Stage 2 (含 Stage 1)
        self.stage2_model = Stage2AlignmentModel(stage1_checkpoint=None, config=config)
        if stage2_checkpoint and os.path.exists(stage2_checkpoint):
            state_dict = torch.load(stage2_checkpoint, map_location='cpu')['model_state_dict']
            self.stage2_model.load_state_dict(state_dict)
        for param in self.stage2_model.parameters():
            param.requires_grad = False

        self.phrase_parser = PhraseParser()
        self.phrase_encoder = PhraseEncoder(hidden_dim=config.hidden_dim)

        self.patch_enhancer = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        self.contrastive_module = PhrasePatchContrastive(hidden_dim=128, temperature=0.07)

    def forward(self, ref_image, tar_image, text_tokens, ref_components, tar_components, differ_texts: List[str]):
        device = ref_image.device

        # 1. 文本解析
        phrases_batch = self.phrase_parser.batch_parse(differ_texts)
        num_phrases = sum(len(p) for p in phrases_batch)

        # 2. 跨阶段获取特征 (冻结前向传播)
        with torch.no_grad():
            stage1_out = self.stage2_model.stage1(ref_image, tar_image)
            diff_patches = stage1_out['diff_patches']  # 从 Stage1 直接拿到 196 个 Patch [B, 196, 768]

            stage2_out = self.stage2_model(ref_image, tar_image, text_tokens, ref_components, tar_components)
            g_v = stage2_out['gate_values'][:, 0:1].unsqueeze(-1)  # 拿到 Stage2 的视觉门控 [B, 1, 1]

        # 乘以视觉置信度，得到重校准后的特征
        recalibrated_patches = self.patch_enhancer(diff_patches * g_v)

        # 3. 语义编码
        phrase_features, batch_indices, target_bboxes = self.phrase_encoder(phrases_batch, device)

        # 4. PPCL 双向对比
        c_out = self.contrastive_module(phrase_features, recalibrated_patches, batch_indices, target_bboxes)

        # 5. 返回字典必须严丝合缝对齐 Stage 3 Trainer 提取需求
        return {
            'total_contrastive_loss': c_out['total_contrastive_loss'],
            'loss_phrase_to_patch': c_out['loss_phrase_to_patch'],
            'loss_patch_to_phrase': c_out['loss_patch_to_phrase'],
            'num_phrases': num_phrases,
            'parsed_phrases': phrases_batch,
            'correspondences': c_out['correspondences'],
            'stage2_features': {'alignment_scores': stage2_out['alignment_logits']}  # 为了可选的对齐损失
        }
