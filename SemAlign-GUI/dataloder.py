import h5py
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import gc
import warnings
import random


class EnhancedGUIDataset(Dataset):
    """增强的GUI数据集加载器 - 专门为第二阶段设计"""

    def __init__(self,
                 split: str,
                 config,
                 is_stage1: bool = True,
                 augment: bool = False):
        self.config = config
        self.split = split
        self.is_stage1 = is_stage1
        self.augment = augment and split == 'train'

        # 加载元数据
        meta_path = Path(config.data_root) / f"{split}.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"元数据文件不存在: {meta_path}")

        print(f"加载 {split} 集元数据...")
        with open(meta_path, 'r') as f:
            self.metadata = json.load(f)

        # 打开HDF5文件
        self.images_h5_path = Path(config.data_root) / "images.h5"
        self.masks_h5_path = Path(config.data_root) / "masks.h5"

        if not self.images_h5_path.exists() or not self.masks_h5_path.exists():
            raise FileNotFoundError("HDF5文件不存在")

        # 使用lazy加载
        self._images_h5 = None
        self._masks_h5 = None

        # 预计算有效样本
        print(f"预过滤 {split} 集样本...")
        self.valid_indices = self._prefilter_samples()

        if len(self.valid_indices) == 0:
            warnings.warn(f"{split} 集没有有效样本！")

        print(f"加载 {split} 集完成: {len(self.valid_indices)} 个有效样本")

        # 组件类型映射
        self.type_map = {
            'TextView': 1, 'ImageView': 2, 'Button': 3,
            'EditText': 4, 'WebView': 5, 'View': 6,
            'CheckBox': 7, 'RadioButton': 8, 'Switch': 9,
            'ToggleButton': 10, 'Widget': 11, 'SwitchMain': 12,
            'SwitchSlider': 13
        }

    @property
    def images_h5(self):
        if self._images_h5 is None:
            self._images_h5 = h5py.File(self.images_h5_path, 'r', libver='latest')
        return self._images_h5

    @property
    def masks_h5(self):
        if self._masks_h5 is None:
            self._masks_h5 = h5py.File(self.masks_h5_path, 'r', libver='latest')
        return self._masks_h5

    def _prefilter_samples(self) -> List[int]:
        """预过滤，避免运行时错误"""
        valid = []
        for idx in range(len(self.metadata)):
            try:
                meta = self.metadata[idx]
                class_name = meta['class_name']
                h5_idx = meta['hdf5_index']

                # 检查HDF5文件
                with h5py.File(self.images_h5_path, 'r') as temp_h5:
                    if class_name in temp_h5:
                        ref_group = temp_h5[class_name]
                        if h5_idx < len(ref_group['reference_images']):
                            # 第二阶段额外检查文本和组件信息
                            if not self.is_stage1:
                                if 'differ_text' in meta and meta['differ_text']:
                                    # 检查是否有组件信息
                                    if ('reference_components' in meta and
                                            'target_components' in meta):
                                        if (len(meta['reference_components']) > 0 or
                                                len(meta['target_components']) > 0):
                                            valid.append(idx)
                                else:
                                    # 第一阶段只需要图像
                                    valid.append(idx)
                            else:
                                valid.append(idx)
            except Exception as e:
                continue

        return valid

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本，增强第二阶段数据处理"""
        try:
            meta_idx = self.valid_indices[idx]
            meta = self.metadata[meta_idx]

            # 从HDF5加载图像数据
            class_name = meta['class_name']
            h5_idx = meta['hdf5_index']

            # 加载图像对
            ref_img = self.images_h5[class_name]['reference_images'][h5_idx]
            tar_img = self.images_h5[class_name]['target_images'][h5_idx]

            # 转换为float32并归一化到[0, 1]
            ref_img = ref_img.astype(np.float32) / 255.0
            tar_img = tar_img.astype(np.float32) / 255.0

            # 数据增强
            if self.augment:
                ref_img, tar_img = self._apply_augmentation(ref_img, tar_img)

            # 转换为tensor
            ref_tensor = torch.from_numpy(ref_img)
            tar_tensor = torch.from_numpy(tar_img)

            # 加载mask
            mask = self.masks_h5[class_name]['change_masks'][h5_idx]
            mask_tensor = torch.from_numpy(mask.astype(np.float32) / 255.0)

            # Stage1只需要图像和mask
            if self.is_stage1:
                return {
                    'ref_image': ref_tensor,
                    'tar_image': tar_tensor,
                    'mask': mask_tensor,
                    'idx': torch.tensor(meta_idx, dtype=torch.long)
                }

            # ============ 第二阶段增强数据 ============
            # 1. 文本描述
            text = meta.get('differ_text', '')

            # 2. 组件信息处理（关键修改）
            ref_components = self._process_components_detailed(
                meta.get('reference_components', []),
                is_target=False
            )
            tar_components = self._process_components_detailed(
                meta.get('target_components', []),
                is_target=True
            )

            # 3. 组件变化匹配（核心功能）
            changes = meta.get('changes', {})
            component_changes = self._extract_component_changes(
                ref_components, tar_components, changes
            )

            # 4. 变化类型编码
            change_type = self._encode_change_type(changes)

            # 5. 检查是否有变化
            has_change = (mask.sum() > 10).item()

            return {
                # 图像数据
                'ref_image': ref_tensor,
                'tar_image': tar_tensor,
                'mask': mask_tensor,

                # 文本数据
                'text': text,
                'text_tokens': self._tokenize_text(text),

                # 组件数据
                'ref_components': ref_components,
                'tar_components': tar_components,
                'component_changes': component_changes,

                # 变化信息
                'change_type': change_type,
                'has_change': torch.tensor(has_change, dtype=torch.float32),
                'changes_raw': changes,

                # 元数据
                'class_name': class_name,
                'idx': torch.tensor(meta_idx, dtype=torch.long)
            }

        except Exception as e:
            print(f"加载样本 {idx} 失败: {e}")
            return self._get_empty_sample()

    def _process_components_detailed(self, components: List[Dict], is_target: bool = False) -> torch.Tensor:
        """详细的组件信息处理"""
        max_comp = self.config.max_components

        if not components or len(components) == 0:
            # 返回[type, x1, y1, x2, y2, area, weight, is_target, is_changed]
            return torch.zeros((max_comp, 9), dtype=torch.float32)

        processed = []
        for comp in components[:max_comp]:
            # 类型编码
            comp_type = comp.get('type', 'View')
            type_id = self.type_map.get(comp_type, 6)  # 默认View

            # 边界框（已归一化到[0, 1]）
            bbox = comp.get('bbox', [0, 0, 0, 0])
            if isinstance(bbox, list) and len(bbox) == 4:
                # 确保在[0, 1]范围内
                bbox_norm = [max(0.0, min(1.0, x)) for x in bbox]
            else:
                bbox_norm = [0, 0, 0, 0]

            # 计算中心点和宽高
            center_x = (bbox_norm[0] + bbox_norm[2]) / 2
            center_y = (bbox_norm[1] + bbox_norm[3]) / 2
            width = bbox_norm[2] - bbox_norm[0]
            height = bbox_norm[3] - bbox_norm[1]

            # 面积（归一化）
            area = comp.get('area', 0) / (224 * 224)
            weight = comp.get('weight', 1.0)

            # 目标标识
            target_flag = 1.0 if is_target else 0.0

            # 变化标识（初始为0，后续计算）
            changed_flag = 0.0

            # 特征：[类型, x1, y1, x2, y2, 中心x, 中心y, 宽, 高, 面积, 权重, 是否目标, 是否变化]
            feature = [
                float(type_id),
                bbox_norm[0], bbox_norm[1], bbox_norm[2], bbox_norm[3],
                center_x, center_y, width, height,
                area, weight, target_flag, changed_flag
            ]

            processed.append(feature)

        # 填充到最大长度
        if len(processed) < max_comp:
            padding = [0.0] * 13
            padding[11] = 1.0 if is_target else 0.0  # 设置目标标识
            while len(processed) < max_comp:
                processed.append(padding)

        return torch.tensor(processed, dtype=torch.float32)

    def _extract_component_changes(self, ref_comps, tar_comps, changes):
        """提取组件变化矩阵"""
        max_comp = self.config.max_components

        # 创建变化矩阵: [max_comp, max_comp, 3]
        # 3个通道: 0=匹配分数, 1=变化类型, 2=是否变化
        change_matrix = torch.zeros((max_comp, max_comp, 3))

        # 计算IoU匹配
        for i in range(min(max_comp, ref_comps.shape[0])):
            for j in range(min(max_comp, tar_comps.shape[0])):
                if ref_comps[i, 0] > 0 and tar_comps[j, 0] > 0:  # 有效组件
                    # 计算IoU
                    box1 = ref_comps[i, 1:5]
                    box2 = tar_comps[j, 1:5]
                    iou = self._calculate_iou(box1, box2)
                    change_matrix[i, j, 0] = iou

                    # 如果类型相同且IoU>0.5，认为是同一组件
                    if ref_comps[i, 0] == tar_comps[j, 0] and iou > 0.5:
                        change_matrix[i, j, 1] = 0.0  # 不变
                        change_matrix[i, j, 2] = 1.0  # 有对应关系

        # 标记添加的组件（在目标中存在但在参考中没有匹配）
        for j in range(min(max_comp, tar_comps.shape[0])):
            if tar_comps[j, 0] > 0:
                has_match = change_matrix[:, j, 2].sum() > 0
                if not has_match:
                    # 可能是添加的组件
                    change_matrix[0, j, 1] = 1.0  # 添加
                    change_matrix[0, j, 2] = 1.0  # 有变化

        return change_matrix

    def _calculate_iou(self, box1, box2):
        """计算两个边界框的IoU"""
        # box: [x1, y1, x2, y2]
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / (union + 1e-8)

    def _encode_change_type(self, changes):
        """编码变化类型为one-hot向量"""
        # 4类: [无变化, 添加, 移除, 移动]
        change_type = torch.zeros(4, dtype=torch.float32)

        if not changes:
            change_type[0] = 1.0  # 无变化
            return change_type

        has_added = len(changes.get('added', [])) > 0
        has_removed = len(changes.get('removed', [])) > 0
        has_moved = len(changes.get('moved', [])) > 0

        if has_added:
            change_type[1] = 1.0
        if has_removed:
            change_type[2] = 1.0
        if has_moved:
            change_type[3] = 1.0

        # 如果没有明确类型，设为通用变化
        if not (has_added or has_removed or has_moved):
            change_type[0] = 1.0

        return change_type

    def _tokenize_text(self, text):
        """简单的文本分词"""
        # 按空格分词，限制长度
        tokens = text.split()[:self.config.max_text_len]
        # 简单的词汇表映射
        vocab = {'Added': 1, 'Removed': 2, 'TextView': 3, 'Button': 4,
                 'ImageView': 5, 'position': 6, 'from': 7, 'to': 8}

        token_ids = []
        for token in tokens:
            if token in vocab:
                token_ids.append(vocab[token])
            else:
                # 数字或标点
                if token.isdigit() or token.replace('.', '').isdigit():
                    token_ids.append(9)  # 数字
                elif token in '();':
                    token_ids.append(10)  # 标点
                else:
                    token_ids.append(11)  # 其他

        # 填充
        if len(token_ids) < self.config.max_text_len:
            token_ids.extend([0] * (self.config.max_text_len - len(token_ids)))

        return torch.tensor(token_ids[:self.config.max_text_len], dtype=torch.long)

    def _apply_augmentation(self, ref_img, tar_img):
        """应用数据增强"""
        # 随机颜色抖动
        if random.random() < 0.3:
            brightness = random.uniform(0.8, 1.2)
            ref_img = np.clip(ref_img * brightness, 0, 1)
            tar_img = np.clip(tar_img * brightness, 0, 1)

        # 随机水平翻转
        if random.random() < 0.5:
            ref_img = np.flip(ref_img, axis=2).copy()
            tar_img = np.flip(tar_img, axis=2).copy()

        return ref_img, tar_img

    def _get_empty_sample(self):
        """返回空样本"""
        if self.is_stage1:
            return {
                'ref_image': torch.zeros((3, 224, 224), dtype=torch.float32),
                'tar_image': torch.zeros((3, 224, 224), dtype=torch.float32),
                'mask': torch.zeros((224, 224), dtype=torch.float32),
                'idx': torch.tensor(-1, dtype=torch.long)
            }
        else:
            return {
                'ref_image': torch.zeros((3, 224, 224), dtype=torch.float32),
                'tar_image': torch.zeros((3, 224, 224), dtype=torch.float32),
                'mask': torch.zeros((224, 224), dtype=torch.float32),
                'text': '',
                'text_tokens': torch.zeros(self.config.max_text_len, dtype=torch.long),
                'ref_components': torch.zeros((self.config.max_components, 13), dtype=torch.float32),
                'tar_components': torch.zeros((self.config.max_components, 13), dtype=torch.float32),
                'component_changes': torch.zeros((self.config.max_components, self.config.max_components, 3),
                                                 dtype=torch.float32),
                'change_type': torch.zeros(4, dtype=torch.float32),
                'has_change': torch.tensor(0, dtype=torch.float32),
                'class_name': 'empty',
                'idx': torch.tensor(-1, dtype=torch.long)
            }

    def __del__(self):
        """清理资源"""
        if self._images_h5 is not None:
            self._images_h5.close()
        if self._masks_h5 is not None:
            self._masks_h5.close()
        gc.collect()


def enhanced_collate_fn(batch):
    """增强的自定义collate函数"""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    collated = {}
    keys = batch[0].keys()

    for key in keys:
        values = [item[key] for item in batch]

        if isinstance(values[0], torch.Tensor):
            try:
                collated[key] = torch.stack(values)
            except:
                # 对于不同形状的张量，保留列表
                collated[key] = values
        elif isinstance(values[0], str):
            collated[key] = values
        elif isinstance(values[0], dict):
            collated[key] = values
        else:
            collated[key] = torch.tensor(values) if isinstance(values[0], (int, float)) else values

    return collated


def create_enhanced_data_loader(split: str, config, is_stage1: bool = True, shuffle: bool = True):
    """创建增强的DataLoader"""
    dataset = EnhancedGUIDataset(split, config, is_stage1, augment=(split == 'train'))

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle and split == 'train',
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and config.cuda_available,
        drop_last=split == 'train',
        persistent_workers=config.num_workers > 0,
        collate_fn=enhanced_collate_fn
    )


# 兼容性函数
def create_data_loader(split: str, config, is_stage1: bool = True, shuffle: bool = True):
    return create_enhanced_data_loader(split, config, is_stage1, shuffle)


def collate_fn(batch):
    return enhanced_collate_fn(batch)