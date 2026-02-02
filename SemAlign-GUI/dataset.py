#!/usr/bin/env python3
"""
GUI数据完整预处理工具 - 修复版
修复组件坐标缩放问题，确保与224×224图像对齐
"""
import gc
import json
import random
import re
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import ijson
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append('.')
try:
    from config3 import get_full_config
except ImportError as e:
    print(f"❌ 无法导入配置: {e}")
    sys.exit(1)

# ---------- 颜色映射 ----------
COLOR_MAPPING = {
    (0, 255, 0): ("TextView", 1.2), (0, 0, 255): ("ImageView", 1.5),
    (198, 204, 79): ("CheckedTextView", 1.0), (93, 47, 207): ("WebView", 1.5),
    (187, 187, 187): ("View", 0.8), (255, 0, 0): ("EditText", 1.5),
    (238, 179, 142): ("ToggleButton", 1.0), (150, 105, 72): ("ToggleButtonOutline", 0.7),
    (0, 165, 255): ("RadioButton", 1.1), (0, 255, 255): ("Button", 1.3),
    (15, 196, 241): ("CheckBox", 1.0), (139, 125, 96): ("SwitchMain", 1.0),
    (56, 234, 251): ("SwitchSlider", 1.0), (203, 192, 255): ("Widget", 0.8)
}

# ---------- GPU 加速 ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def gpu_image_pipeline(image_path, target_size, dtype=torch.float16):
    """加载、缩放和标准化图像"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法加载图像: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img).to(DEVICE, dtype=dtype).permute(2, 0, 1)
    del img
    tensor = torch.nn.functional.interpolate(
        tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False
    ).squeeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE, dtype=dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE, dtype=dtype).view(3, 1, 1)
    return ((tensor / 255. - mean) / std).contiguous()


# ---------- 修复：组件检测（同步缩放） ----------
class FixedColorComponentDetector:
    def __init__(self, color_mapping=COLOR_MAPPING, tolerance=10):  # 减小容差
        self.color_mapping = color_mapping
        self.tolerance = tolerance

    def detect_components(self, image_path, target_size=(224, 224)):
        """检测组件并直接返回缩放后的坐标"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")

        original_h, original_w = image.shape[:2]
        target_w, target_h = target_size

        # 计算缩放比例
        scale_x = target_w / original_w
        scale_y = target_h / original_h

        components = []
        for color, (comp_type, weight) in self.color_mapping.items():
            lower = np.array([max(0, c - self.tolerance) for c in color])
            upper = np.array([min(255, c + self.tolerance) for c in color])

            mask = cv2.inRange(image, lower, upper)

            # 形态学处理（减少噪声）
            kernel = np.ones((2, 2), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 10:  # 减小面积阈值
                    x, y, ww, hh = cv2.boundingRect(cnt)

                    # === 关键修复：立即缩放坐标 ===
                    scaled_x1 = int(x * scale_x)
                    scaled_y1 = int(y * scale_y)
                    scaled_x2 = int((x + ww) * scale_x)
                    scaled_y2 = int((y + hh) * scale_y)

                    # 确保坐标在目标图像范围内
                    scaled_x1 = max(0, min(scaled_x1, target_w - 1))
                    scaled_y1 = max(0, min(scaled_y1, target_h - 1))
                    scaled_x2 = max(1, min(scaled_x2, target_w))
                    scaled_y2 = max(1, min(scaled_y2, target_h))

                    # 确保边界框有效
                    if scaled_x2 <= scaled_x1 or scaled_y2 <= scaled_y1:
                        continue

                    # 计算缩放后的面积
                    scaled_area = (scaled_x2 - scaled_x1) * (scaled_y2 - scaled_y1)
                    if scaled_area < 4:  # 跳过太小的组件
                        continue

                    components.append({
                        'type': comp_type,
                        'bbox': [scaled_x1, scaled_y1, scaled_x2, scaled_y2],  # 224×224坐标
                        'weight': weight,
                        'area': scaled_area
                    })

        # 按面积排序并限制数量
        components.sort(key=lambda x: x['area'], reverse=True)
        return components[:20]

    def validate_components(self, components, image_size=(224, 224)):
        """验证组件坐标是否有效"""
        img_w, img_h = image_size
        valid_components = []

        for comp in components:
            bbox = comp.get('bbox', [])
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox

            # 检查边界
            if (0 <= x1 < x2 <= img_w) and (0 <= y1 < y2 <= img_h):
                valid_components.append(comp)
            else:
                # 修复边界问题
                fixed_bbox = [
                    max(0, min(x1, img_w - 1)),
                    max(0, min(y1, img_h - 1)),
                    max(1, min(x2, img_w)),
                    max(1, min(y2, img_h))
                ]
                if fixed_bbox[0] < fixed_bbox[2] and fixed_bbox[1] < fixed_bbox[3]:
                    comp['bbox'] = fixed_bbox
                    valid_components.append(comp)

        return valid_components


# ---------- 修复：坐标处理器 ----------
class FixedCoordinateProcessor:
    def __init__(self, original_size=(144, 256), target_size=(224, 224)):
        self.original_size = original_size
        self.target_size = target_size

    def scale_bbox(self, bbox):
        """缩放bbox从原始尺寸到目标尺寸"""
        if len(bbox) != 4:
            return [0, 0, 0, 0]

        orig_w, orig_h = self.original_size
        target_w, target_h = self.target_size

        x1, y1, x2, y2 = bbox

        # 缩放
        scaled_x1 = int(x1 * target_w / orig_w)
        scaled_y1 = int(y1 * target_h / orig_h)
        scaled_x2 = int(x2 * target_w / orig_w)
        scaled_y2 = int(y2 * target_h / orig_h)

        # 确保在范围内
        scaled_x1 = max(0, min(scaled_x1, target_w - 1))
        scaled_y1 = max(0, min(scaled_y1, target_h - 1))
        scaled_x2 = max(1, min(scaled_x2, target_w))
        scaled_y2 = max(1, min(scaled_y2, target_h))

        return [scaled_x1, scaled_y1, scaled_x2, scaled_y2]

    def normalize_bbox(self, bbox):
        """将bbox归一化到[0,1]（用于changes字段）"""
        x1, y1, x2, y2 = bbox
        target_w, target_h = self.target_size
        return [x1 / target_w, y1 / target_h, x2 / target_w, y2 / target_h]

    def parse_difference_description(self, differ_str, max_changes=10):
        """解析差异描述，返回缩放后的坐标"""
        changes = {'moved': [], 'added': [], 'removed': [], 'unchanged': []}

        # 解析Added
        added_pattern = r'Added (\w+) at position \((\d+), (\d+), (\d+), (\d+)\)'
        for comp_type, x1, y1, x2, y2 in re.findall(added_pattern, differ_str):
            bbox = self.scale_bbox([int(x1), int(y1), int(x2), int(y2)])
            norm_bbox = self.normalize_bbox(bbox)
            changes['added'].append({'type': comp_type, 'bbox': norm_bbox})

        # 解析Removed
        removed_pattern = r'Removed (\w+) from position \((\d+), (\d+), (\d+), (\d+)\)'
        for comp_type, x1, y1, x2, y2 in re.findall(removed_pattern, differ_str):
            bbox = self.scale_bbox([int(x1), int(y1), int(x2), int(y2)])
            norm_bbox = self.normalize_bbox(bbox)
            changes['removed'].append({'type': comp_type, 'bbox': norm_bbox})

        # 解析Moved
        moved_pattern = r'(\w+) from \((\d+), (\d+), (\d+), (\d+)\) to \((\d+), (\d+), (\d+), (\d+)\)'
        for comp_type, fx1, fy1, fx2, fy2, tx1, ty1, tx2, ty2 in re.findall(moved_pattern, differ_str):
            from_bbox = self.scale_bbox([int(fx1), int(fy1), int(fx2), int(fy2)])
            to_bbox = self.scale_bbox([int(tx1), int(ty1), int(tx2), int(ty2)])
            changes['moved'].append({
                'type': comp_type,
                'from_bbox': self.normalize_bbox(from_bbox),
                'to_bbox': self.normalize_bbox(to_bbox)
            })

        # 限制数量并排序
        important = ['ImageView', 'Button', 'TextView', 'EditText', 'WebView']
        for t in ['moved', 'added', 'removed']:
            imp = [c for c in changes[t] if c['type'] in important]
            other = [c for c in changes[t] if c['type'] not in important]
            changes[t] = (imp + other)[:max_changes]

        return changes

    def truncate_description(self, differ_str, max_length=512):
        """截断描述文本"""
        if len(differ_str) <= max_length:
            return differ_str
        changes = differ_str.split('; ')
        imp = [c for c in changes if any(k in c for k in ['ImageView', 'Button', 'TextView', 'EditText'])]
        other = [c for c in changes if c not in imp]
        truncated = '; '.join(imp + other)
        return truncated[:max_length - 3] + '...' if len(truncated) > max_length else truncated


# ---------- 变化 mask 生成 ----------
def generate_change_mask(differ_str, target_size=(224, 224)):
    """生成变化mask（使用缩放后的坐标）"""
    h, w = target_size
    mask = np.zeros((h, w), dtype=np.uint8)

    # 计算缩放比例（从144×256到224×224）
    scale_w = w / 144
    scale_h = h / 256

    # 处理Added
    for comp_type, x1, y1, x2, y2 in re.findall(r'Added (\w+) at position \((\d+), (\d+), (\d+), (\d+)\)', differ_str):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        # 缩放坐标
        sx1 = int(x1 * scale_w)
        sy1 = int(y1 * scale_h)
        sx2 = int(x2 * scale_w)
        sy2 = int(y2 * scale_h)
        # 确保在范围内
        sx1, sy1 = max(0, sx1), max(0, sy1)
        sx2, sy2 = min(w, sx2), min(h, sy2)
        if sx1 < sx2 and sy1 < sy2:
            mask[sy1:sy2, sx1:sx2] = 1

    # 处理Removed
    for comp_type, x1, y1, x2, y2 in re.findall(r'Removed (\w+) from position \((\d+), (\d+), (\d+), (\d+)\)',
                                                differ_str):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        sx1 = int(x1 * scale_w)
        sy1 = int(y1 * scale_h)
        sx2 = int(x2 * scale_w)
        sy2 = int(y2 * scale_h)
        sx1, sy1 = max(0, sx1), max(0, sy1)
        sx2, sy2 = min(w, sx2), min(h, sy2)
        if sx1 < sx2 and sy1 < sy2:
            mask[sy1:sy2, sx1:sx2] = 1

    # 处理Moved（目标位置）
    for comp_type, from_pos, to_pos in re.findall(r'(\w+) from \(([\d, ]+)\) to \(([\d, ]+)\)', differ_str):
        x1, y1, x2, y2 = map(int, to_pos.split(', '))
        sx1 = int(x1 * scale_w)
        sy1 = int(y1 * scale_h)
        sx2 = int(x2 * scale_w)
        sy2 = int(y2 * scale_h)
        sx1, sy1 = max(0, sx1), max(0, sy1)
        sx2, sy2 = min(w, sx2), min(h, sy2)
        if sx1 < sx2 and sy1 < sy2:
            mask[sy1:sy2, sx1:sx2] = 1

    return mask


# ---------- HDF5 存储 ----------
class HDF5Storage:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.img_path = self.output_dir / "images.h5"
        self.mask_path = self.output_dir / "masks.h5"
        self._files = {}

    def _open(self, flag='a'):
        if 'img' not in self._files:
            self._files['img'] = h5py.File(self.img_path, flag)
            self._files['mask'] = h5py.File(self.mask_path, flag)
        return self._files['img'], self._files['mask']

    def append_single(self, class_name, ref_tensor, tar_tensor, mask):
        """单条追加到HDF5"""
        img_f, mask_f = self._open()

        # 创建组（如果不存在）
        if class_name not in img_f:
            img_grp = img_f.create_group(class_name)
            mask_grp = mask_f.create_group(class_name)

            img_grp.create_dataset('reference_images',
                                   shape=(0, 3, 224, 224),
                                   maxshape=(None, 3, 224, 224),
                                   dtype=np.float16,
                                   compression='gzip', compression_opts=4,
                                   chunks=(1, 3, 224, 224))
            img_grp.create_dataset('target_images',
                                   shape=(0, 3, 224, 224),
                                   maxshape=(None, 3, 224, 224),
                                   dtype=np.float16,
                                   compression='gzip', compression_opts=4,
                                   chunks=(1, 3, 224, 224))
            mask_grp.create_dataset('change_masks',
                                    shape=(0, 224, 224),
                                    maxshape=(None, 224, 224),
                                    dtype=np.uint8,
                                    compression='gzip', compression_opts=4,
                                    chunks=(1, 224, 224))

        # 获取数据集
        ref_dset = img_f[class_name]['reference_images']
        tar_dset = img_f[class_name]['target_images']
        mask_dset = mask_f[class_name]['change_masks']

        # 扩展并写入
        old_size = ref_dset.shape[0]
        ref_dset.resize(old_size + 1, axis=0)
        tar_dset.resize(old_size + 1, axis=0)
        mask_dset.resize(old_size + 1, axis=0)

        ref_dset[old_size] = ref_tensor.cpu().numpy().astype(np.float16)
        tar_dset[old_size] = tar_tensor.cpu().numpy().astype(np.float16)
        mask_dset[old_size] = mask

        return old_size

    def close(self):
        for f in self._files.values():
            f.close()
        self._files.clear()


# ---------- 预处理主类 ----------
class FixedGUIPreprocessor:
    def __init__(self, config):
        self.cfg = config
        self.detector = FixedColorComponentDetector()
        self.coord_proc = FixedCoordinateProcessor(
            original_size=config['data']['original_size'],
            target_size=config['data']['image_size']
        )
        self.target_size = config['data']['image_size']

    def process_single(self, sample, class_name):
        """处理单个样本"""
        try:
            # 构建路径
            ref_path = Path(self.cfg['data']['gui_dir']) / class_name / sample['reference']
            tar_path = Path(self.cfg['data']['gui_dir']) / class_name / sample['target']

            if not ref_path.exists() or not tar_path.exists():
                print(f"跳过：文件不存在 - {ref_path} 或 {tar_path}")
                return None

            # === 关键修复：使用相同的target_size ===
            target_size = self.target_size

            # 1. 处理图像（缩放并标准化）
            ref_tensor = gpu_image_pipeline(str(ref_path), target_size).cpu()
            tar_tensor = gpu_image_pipeline(str(tar_path), target_size).cpu()

            # 2. 检测组件（返回缩放后的坐标）
            ref_comp = self.detector.detect_components(str(ref_path), target_size=target_size)
            tar_comp = self.detector.detect_components(str(tar_path), target_size=target_size)

            # 验证组件坐标
            ref_comp = self.detector.validate_components(ref_comp, target_size)
            tar_comp = self.detector.validate_components(tar_comp, target_size)

            # 3. 处理文本差异
            differ_text = self.coord_proc.truncate_description(
                sample['differ'],
                self.cfg['data']['max_text_length']
            )

            # 4. 解析变化（使用缩放后的坐标）
            changes = self.coord_proc.parse_difference_description(sample['differ'])

            # 5. 生成变化mask（使用缩放后的坐标）
            change_mask = generate_change_mask(sample['differ'], target_size=target_size)

            # 6. 返回结果
            return {
                'image_pair': (ref_tensor, tar_tensor),
                'mask': change_mask,
                'reference_components': ref_comp,
                'target_components': tar_comp,
                'differ_text': differ_text,
                'changes': changes,
                'class_name': class_name,
                'reference_path': str(ref_path),
                'target_path': str(tar_path)
            }

        except Exception as e:
            print(f"跳过样本 {sample.get('reference', 'unknown')}: {e}")
            return None

    def process_all_stream(self, debug=False, gc_every=500):
        """流式处理所有数据"""
        text_dir = Path(self.cfg['data']['text_dir'])
        if not text_dir.exists():
            print(f"❌ 文本目录不存在: {text_dir}")
            return []

        json_files = list(text_dir.glob('*.json'))
        if debug:
            json_files = json_files[:2]
            print(f"调试模式：只处理前{len(json_files)}个类别")

        meta_dir = Path(self.cfg['data']['output_dir']) / 'meta'
        meta_dir.mkdir(exist_ok=True)

        all_meta_files = []

        for json_file in tqdm(json_files, desc="处理JSON文件"):
            class_name = json_file.stem
            print(f"\n处理类别: {class_name}")

            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    samples = list(ijson.items(f, 'item'))

                if debug:
                    samples = samples[:50]
                    print(f"  调试模式：只处理前{len(samples)}个样本")

                meta_list = []
                h5_storage = HDF5Storage(self.cfg['data']['output_dir'])

                for idx, smp in enumerate(tqdm(samples, desc=f"  样本", leave=False)):
                    result = self.process_single(smp, class_name)

                    if result is None:
                        continue

                    # 写入HDF5
                    h5_idx = h5_storage.append_single(
                        class_name,
                        result['image_pair'][0],
                        result['image_pair'][1],
                        result['mask']
                    )

                    # 准备元数据
                    result['hdf5_index'] = h5_idx
                    del result['image_pair']  # 图像已保存到HDF5
                    meta_list.append(result)

                    # 定期清理内存
                    if idx > 0 and idx % gc_every == 0:
                        gc.collect()
                        torch.cuda.empty_cache()

                # 保存该类别的元数据
                if meta_list:
                    meta_file = meta_dir / f"{class_name}.json"
                    with open(meta_file, 'w', encoding='utf-8') as mf:
                        json.dump(meta_list, mf, indent=2, ensure_ascii=False, default=str)
                    all_meta_files.append(meta_file)

                    # 打印统计信息
                    print(f"  ✅ 完成: {len(meta_list)}个样本")
                    print(f"  📊 参考组件平均: {sum(len(m['reference_components']) for m in meta_list) / len(meta_list):.1f}")
                    print(f"  📊 目标组件平均: {sum(len(m['target_components']) for m in meta_list) / len(meta_list):.1f}")

                    # 验证几个样本的坐标
                    for i, meta in enumerate(meta_list[:3]):
                        ref_comps = meta['reference_components']
                        tar_comps = meta['target_components']
                        if ref_comps:
                            bbox = ref_comps[0]['bbox']
                            print(f"    样本{i}参考组件坐标: {bbox} (应在0-224范围内)")

                h5_storage.close()

            except Exception as e:
                print(f"❌ 处理类别 {class_name} 时出错: {e}")
                continue

        print(f"\n✅ 所有类别处理完成，共生成 {len(all_meta_files)} 个元数据文件")
        return all_meta_files


# ---------- 类别感知的数据划分 ----------
class ClassAwareSplitter:
    def __init__(self, config):
        self.cfg = config

    def split_stream(self, meta_files):
        """流式划分数据集"""
        out_dir = Path(self.cfg['data']['output_dir'])
        out_dir.mkdir(parents=True, exist_ok=True)

        train_idx, val_idx, test_idx = [], [], []

        for mf in tqdm(meta_files, desc="划分数据集"):
            cls = mf.stem
            try:
                with open(mf, 'r', encoding='utf-8') as f:
                    meta = json.load(f)

                if len(meta) < 3:
                    # 样本太少，全部放入训练集
                    train_idx.extend([{'class_name': cls, **m} for m in meta])
                    continue

                # 划分数据集
                tv, te = train_test_split(
                    meta,
                    test_size=self.cfg['data']['test_split'],
                    random_state=42
                )
                tr, va = train_test_split(
                    tv,
                    test_size=self.cfg['data']['val_split'] / (1 - self.cfg['data']['test_split']),
                    random_state=42
                )

                train_idx.extend(tr)
                val_idx.extend(va)
                test_idx.extend(te)

            except Exception as e:
                print(f"❌ 划分类别 {cls} 时出错: {e}")
                continue

        # 打乱数据
        random.shuffle(train_idx)
        random.shuffle(val_idx)
        random.shuffle(test_idx)

        # 保存最终文件
        for name, lst in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
            output_file = out_dir / f"{name}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(lst, f, indent=2, ensure_ascii=False, default=str)
            print(f"✅ {name}.json 保存完成，共 {len(lst)} 条样本")

            # 验证前几个样本
            if lst:
                sample = lst[0]
                print(f"  📋 {name}集样本示例:")
                print(f"    类别: {sample.get('class_name')}")
                print(f"    HDF5索引: {sample.get('hdf5_index')}")
                print(f"    文本长度: {len(sample.get('differ_text', ''))}")
                if sample.get('reference_components'):
                    bbox = sample['reference_components'][0]['bbox']
                    print(f"    组件坐标示例: {bbox}")


# ---------- 主入口 ----------
def main():
    parser = ArgumentParser(description="GUI数据预处理工具 - 修复坐标对齐问题")
    parser.add_argument('--debug', action='store_true', help="调试模式，只处理少量数据")
    parser.add_argument('--gui_dir', type=str, help="GUI图像目录路径")
    parser.add_argument('--text_dir', type=str, help="文本描述目录路径")
    parser.add_argument('--output_dir', type=str, help="输出目录路径")
    parser.add_argument('--validate_only', action='store_true', help="只验证不处理")

    args = parser.parse_args()

    # 加载配置
    cfg = get_full_config()

    # 覆盖配置（如果提供了命令行参数）
    for k in ['gui_dir', 'text_dir', 'output_dir']:
        if getattr(args, k):
            cfg['data'][k] = getattr(args, k)

    # 创建输出目录
    output_dir = Path(cfg['data']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🎯 GUI数据预处理开始")
    print(f"   图像目录: {cfg['data']['gui_dir']}")
    print(f"   文本目录: {cfg['data']['text_dir']}")
    print(f"   输出目录: {cfg['data']['output_dir']}")
    print(f"   原始尺寸: {cfg['data']['original_size']}")
    print(f"   目标尺寸: {cfg['data']['image_size']}")
    print(f"   调试模式: {args.debug}")

    if args.validate_only:
        print("\n🔍 验证模式：检查数据完整性")
        # 这里可以添加验证代码
        return

    # 创建预处理器
    preprocessor = FixedGUIPreprocessor(cfg)

    # 处理所有数据
    meta_files = preprocessor.process_all_stream(debug=args.debug)

    if not meta_files:
        print("❌ 没有生成有效的元数据文件")
        return

    # 划分数据集
    splitter = ClassAwareSplitter(cfg)
    splitter.split_stream(meta_files)

    print("\n🎉 数据预处理完成！")
    print(f"   输出文件:")
    print(f"     - {output_dir}/images.h5 (图像数据)")
    print(f"     - {output_dir}/masks.h5 (mask数据)")
    print(f"     - {output_dir}/train.json (训练集)")
    print(f"     - {output_dir}/val.json (验证集)")
    print(f"     - {output_dir}/test.json (测试集)")
    print(f"     - {output_dir}/meta/*.json (各类别元数据)")


if __name__ == "__main__":
    main()