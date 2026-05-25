#!/usr/bin/env python3
"""
三阶段GUI变化分析系统 - 集成推理版
使用训练好的三个阶段权重进行完整分析
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from pathlib import Path
import json
import re
import time
import warnings
from typing import Dict, List, Tuple, Optional, Any
import sys

# 添加模型路径
sys.path.append('.')
warnings.filterwarnings('ignore')

# 导入模型类
from models import Stage1VisualModel
from model2 import Stage2AlignmentModel
from model3 import Stage3PhraseContrastiveModel


class IntegratedGUIAnalyzer:
    """集成三阶段模型的GUI分析系统"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"\n{'=' * 60}")
        print("初始化集成GUI分析系统")
        print(f"设备: {self.device}")
        print(f"{'=' * 60}")

        # 模型路径 - 根据您的实际路径修改
        STAGE1_MODEL = "/home/common-dir/result/training_output/stage1/checkpoints/best_model.pt"
        STAGE2_MODEL = "/home/common-dir/result/training_output/stage2_alignment/checkpoints/best_model.pt"
        STAGE3_MODEL = "/home/common-dir/result/training_output/stage3_phrase_20260114_053101/checkpoints/checkpoint_final.pt"

        # ============ 1. 初始化第一阶段模型 ============
        print("\n1. 加载第一阶段模型...")
        self.stage1_model = Stage1VisualModel(config).to(self.device)

        if Path(STAGE1_MODEL).exists():
            try:
                # 尝试使用weights_only=True
                checkpoint = torch.load(STAGE1_MODEL, map_location=self.device, weights_only=True)
                state_dict = checkpoint.get('model_state_dict', checkpoint)

                # 如果加载失败，尝试其他方法
                if state_dict is None:
                    # 使用pickle直接加载
                    import pickle
                    with open(STAGE1_MODEL, 'rb') as f:
                        state_dict = pickle.load(f)

            except Exception as e:
                print(f"标准加载失败，尝试备选方法: {e}")
                # 备选加载方法
                try:
                    checkpoint = torch.load(STAGE1_MODEL, map_location=self.device,
                                            weights_only=False, pickle_module=pickle)
                    state_dict = checkpoint.get('model_state_dict', checkpoint)
                except:
                    print(f" 所有加载方法均失败")
                    state_dict = None

            if state_dict is not None:
                self.stage1_model.load_state_dict(state_dict, strict=False)
                print("第一阶段模型加载成功")
            else:
                print(f"无法加载模型权重")
        else:
            print(f"第一阶段模型文件不存在: {STAGE1_MODEL}")

        self.stage1_model.eval()

        # ============ 2. 初始化第二阶段模型 ============
        print("\n2. 加载第二阶段模型...")
        self.stage2_model = Stage2AlignmentModel(
            stage1_checkpoint="",  # 不需要stage1检查点，因为已经初始化了模型
            config=config,
            use_components=True
        ).to(self.device)

        if Path(STAGE2_MODEL).exists():
            checkpoint = torch.load(STAGE2_MODEL, map_location=self.device, weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            self.stage2_model.load_state_dict(state_dict, strict=False)
            print("第二阶段模型加载成功")
        else:
            print(f"第二阶段模型文件不存在: {STAGE2_MODEL}")

        self.stage2_model.eval()

        # ============ 3. 初始化第三阶段模型 ============
        print("\n3. 加载第三阶段模型...")
        self.stage3_model = Stage3PhraseContrastiveModel(
            stage2_checkpoint="",  # 不需要stage2检查点
            config=config
        ).to(self.device)

        if Path(STAGE3_MODEL).exists():
            checkpoint = torch.load(STAGE3_MODEL, map_location=self.device, weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            self.stage3_model.load_state_dict(state_dict, strict=False)
            print("第三阶段模型加载成功")
        else:
            print(f"第三阶段模型文件不存在: {STAGE3_MODEL}")

        self.stage3_model.eval()

        # ============ 4. 数据预处理工具 ============
        self.component_types = {
            'TextView': 1, 'ImageView': 2, 'Button': 3,
            'EditText': 4, 'WebView': 5, 'View': 6,
            'CheckBox': 7, 'RadioButton': 8, 'Switch': 9,
            'ToggleButton': 10, 'Widget': 11, 'SwitchMain': 12,
            'SwitchSlider': 13
        }

        # ============ 5. 可视化颜色映射 ============
        self.colors = {
            'addition': '#4CAF50',  # 绿色 - 新增
            'removal': '#F44336',  # 红色 - 移除
            'movement': '#2196F3',  # 蓝色 - 移动
            'TextView': '#FF9800',  # 橙色
            'ImageView': '#3F51B5',  # 深蓝
            'Button': '#009688',  # 青色
            'EditText': '#9C27B0',  # 紫色
            'WebView': '#795548',  # 棕色
            'success': '#4CAF50',  # 成功
            'warning': '#FF9800',  # 警告
            'error': '#F44336',  # 错误
        }

        print(f"\n{'=' * 60}")
        print("集成分析系统初始化完成")
        print(f"设备: {self.device}")
        print(f"{'=' * 60}")

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """预处理图像"""
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((224, 224))

            # 转换为tensor并归一化
            img_np = np.array(img).astype(np.float32) / 255.0
            img_np = (img_np - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

            # [H, W, C] -> [C, H, W]
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
            return img_tensor.to(self.device)

        except Exception as e:
            print(f"图像预处理失败: {e}")
            # 返回随机图像作为后备
            return torch.randn(1, 3, 224, 224).to(self.device)

    def tokenize_text(self, text: str) -> torch.Tensor:
        """简单文本分词"""
        # 简单的词汇表映射
        vocab = {
            'Added': 1, 'Removed': 2, 'TextView': 3, 'Button': 4,
            'ImageView': 5, 'position': 6, 'from': 7, 'to': 8,
            'EditText': 9, 'WebView': 10, 'View': 11, 'SwitchMain': 12,
            'SwitchSlider': 13
        }

        # 按空格分词
        tokens = text.split()
        token_ids = []

        for token in tokens[:self.config.max_text_len]:
            if token in vocab:
                token_ids.append(vocab[token])
            elif token.isdigit() or token.replace('.', '').isdigit():
                token_ids.append(14)  # 数字
            elif token in '();,':  # 标点符号
                token_ids.append(15)
            else:
                token_ids.append(16)  # 其他

        # 填充
        if len(token_ids) < self.config.max_text_len:
            token_ids.extend([0] * (self.config.max_text_len - len(token_ids)))

        return torch.tensor(token_ids[:self.config.max_text_len], dtype=torch.long).unsqueeze(0).to(self.device)

    def parse_components_from_description(self, description: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """从描述中解析组件信息"""
        max_components = self.config.max_components

        # 初始化组件矩阵
        ref_components = torch.zeros((1, max_components, 13), dtype=torch.float32)
        tar_components = torch.zeros((1, max_components, 13), dtype=torch.float32)

        ref_idx = 0
        tar_idx = 0

        # 解析Added组件 (目标组件)
        added_pattern = r'Added (\w+) at position \((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)'
        for comp_type, x1, y1, x2, y2 in re.findall(added_pattern, description):
            if tar_idx >= max_components:
                break

            # 类型编码
            type_id = self.component_types.get(comp_type, 6)  # 默认View

            # 归一化坐标 (0-1)
            x1_norm = int(x1) / 144.0
            y1_norm = int(y1) / 256.0
            x2_norm = int(x2) / 144.0
            y2_norm = int(y2) / 256.0

            # 确保在范围内
            bbox = [
                max(0.0, min(1.0, x1_norm)),
                max(0.0, min(1.0, y1_norm)),
                max(0.0, min(1.0, x2_norm)),
                max(0.0, min(1.0, y2_norm))
            ]

            # 计算特征
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height
            weight = 1.0

            # 填充目标组件
            tar_components[0, tar_idx] = torch.tensor([
                float(type_id),
                bbox[0], bbox[1], bbox[2], bbox[3],  # bbox
                center_x, center_y, width, height,  # 几何特征
                area, weight, 0.0, 0.0  # 面积、权重、非目标、未变化
            ])
            tar_idx += 1

        # 解析Removed组件 (参考组件)
        removed_pattern = r'Removed (\w+) from position \((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)'
        for comp_type, x1, y1, x2, y2 in re.findall(removed_pattern, description):
            if ref_idx >= max_components:
                break

            type_id = self.component_types.get(comp_type, 6)

            x1_norm = int(x1) / 144.0
            y1_norm = int(y1) / 256.0
            x2_norm = int(x2) / 144.0
            y2_norm = int(y2) / 256.0

            bbox = [
                max(0.0, min(1.0, x1_norm)),
                max(0.0, min(1.0, y1_norm)),
                max(0.0, min(1.0, x2_norm)),
                max(0.0, min(1.0, y2_norm))
            ]

            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height
            weight = 1.0

            # 填充参考组件
            ref_components[0, ref_idx] = torch.tensor([
                float(type_id),
                bbox[0], bbox[1], bbox[2], bbox[3],
                center_x, center_y, width, height,
                area, weight, 1.0, 0.0  # 是目标、未变化
            ])
            ref_idx += 1

        return ref_components.to(self.device), tar_components.to(self.device)

    def run_stage1_analysis(self, ref_image: torch.Tensor, tar_image: torch.Tensor) -> Dict:
        """运行第一阶段分析"""
        print("\n第一阶段：视觉基础分析...")

        with torch.no_grad():
            outputs = self.stage1_model(ref_image, tar_image)

        # 提取变化mask
        pred_logits = outputs.get('pred_logits')
        if pred_logits is not None:
            change_mask = torch.sigmoid(pred_logits).cpu().numpy()[0]
        else:
            change_mask = np.zeros((224, 224))

        # 计算视觉差异
        diff_map = self.compute_visual_diff(ref_image, tar_image)

        return {
            'diff_features': outputs.get('diff_features'),
            'change_mask': change_mask,
            'change_probability': outputs.get('change_logits', torch.zeros(1, 1)).sigmoid().item(),
            'diff_map': diff_map,
            'raw_outputs': outputs
        }

    def compute_visual_diff(self, ref_image: torch.Tensor, tar_image: torch.Tensor) -> np.ndarray:
        """计算视觉差异图"""
        ref_np = ref_image.cpu().numpy()[0]
        tar_np = tar_image.cpu().numpy()[0]

        # 反归一化
        mean = np.array([0.485, 0.456, 0.406])[:, None, None]
        std = np.array([0.229, 0.224, 0.225])[:, None, None]

        ref_img = (ref_np * std + mean).transpose(1, 2, 0)
        tar_img = (tar_np * std + mean).transpose(1, 2, 0)

        # 计算差异
        diff = np.abs(ref_img - tar_img).mean(axis=2)

        # 增强对比度
        diff_enhanced = np.power(diff, 0.7)
        if diff_enhanced.max() > 0:
            diff_enhanced = diff_enhanced / diff_enhanced.max()

        return diff_enhanced

    def run_stage2_analysis(self, ref_image: torch.Tensor, tar_image: torch.Tensor,
                            text_tokens: torch.Tensor, ref_components: torch.Tensor,
                            tar_components: torch.Tensor) -> Dict:
        """运行第二阶段分析"""
        print("第二阶段：多模态对齐分析...")

        with torch.no_grad():
            outputs = self.stage2_model(
                ref_image, tar_image, text_tokens,
                ref_components, tar_components
            )

        return {
            'alignment_score': outputs.get('alignment_scores', torch.zeros(1, 1)).item(),
            'alignment_logits': outputs.get('alignment_logits'),
            'fused_features': outputs.get('fused_features'),
            'visual_features': outputs.get('visual_features'),
            'text_features': outputs.get('text_features'),
            'component_features': outputs.get('component_outputs', {}).get('change_features'),
            'raw_outputs': outputs
        }

    def run_stage3_analysis(self, ref_image: torch.Tensor, tar_image: torch.Tensor,
                            text_tokens: torch.Tensor, ref_components: torch.Tensor,
                            tar_components: torch.Tensor, differ_text: str) -> Dict:
        """运行第三阶段分析"""
        print("第三阶段：短语级对比分析...")

        with torch.no_grad():
            outputs = self.stage3_model(
                ref_image, tar_image, text_tokens,
                ref_components, tar_components, [differ_text]
            )

        return {
            'correspondences': outputs.get('correspondences', []),
            'phrase_features': outputs.get('phrase_features'),
            'patch_features': outputs.get('patch_features'),
            'contrastive_loss': outputs.get('total_contrastive_loss', torch.tensor(0.0)).item(),
            'parsed_phrases': outputs.get('parsed_phrases', []),
            'raw_outputs': outputs
        }

    def analyze_with_models(self, ref_image_path: str, tar_image_path: str,
                            description: str, output_prefix: str = "integrated_analysis") -> Dict:
        """使用三阶段模型进行完整分析"""
        print(f"\n{'=' * 60}")
        print("开始集成GUI分析")
        print(f"参考图像: {Path(ref_image_path).name}")
        print(f"目标图像: {Path(tar_image_path).name}")
        print(f"{'=' * 60}")

        start_time = time.time()

        try:
            # ============ 数据预处理 ============
            print("\n1. 数据预处理...")
            ref_image = self.preprocess_image(ref_image_path)
            tar_image = self.preprocess_image(tar_image_path)
            text_tokens = self.tokenize_text(description)
            ref_components, tar_components = self.parse_components_from_description(description)

            # ============ 阶段分析 ============
            print("\n2. 执行三阶段分析...")

            # 第一阶段：视觉基础分析
            stage1_results = self.run_stage1_analysis(ref_image, tar_image)

            # 第二阶段：多模态对齐分析
            stage2_results = self.run_stage2_analysis(
                ref_image, tar_image, text_tokens,
                ref_components, tar_components
            )

            # 第三阶段：短语级对比分析
            stage3_results = self.run_stage3_analysis(
                ref_image, tar_image, text_tokens,
                ref_components, tar_components, description
            )

            # ============ 解析详细变化 ============
            print("\n3. 解析详细变化...")
            detailed_changes = self.parse_detailed_changes(description)

            # ============ 综合评估 ============
            print("\n4. 综合评估...")
            overall_assessment = self.assess_overall_quality(
                detailed_changes,
                stage1_results,
                stage2_results,
                stage3_results
            )

            # ============ 构建结果 ============
            results = {
                'reference_image': ref_image_path,
                'target_image': tar_image_path,
                'description': description,
                'analysis_time': time.time() - start_time,
                'detailed_changes': detailed_changes,
                'stage1_results': stage1_results,
                'stage2_results': stage2_results,
                'stage3_results': stage3_results,
                'overall_assessment': overall_assessment,
                'config': self.config.__dict__
            }

            # ============ 生成可视化 ============
            print("\n5. 生成可视化报告...")
            self.create_integrated_visualization(results, output_prefix)

            # ============ 保存结果 ============
            print("\n6. 保存分析结果...")
            self.save_analysis_results(results, output_prefix)

            # ============ 打印摘要 ============
            self.print_summary(results)

            print(f"\n{'=' * 60}")
            print("集成分析完成!")
            print(f"{'=' * 60}")

            return results

        except Exception as e:
            print(f"\n 分析失败: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}

    def parse_detailed_changes(self, description: str) -> Dict[str, List[Dict]]:
        """解析详细变化"""
        changes = {
            'additions': [],
            'removals': [],
            'movements': [],
            'all_components': []
        }

        # 解析Added
        added_pattern = r'Added (\w+) at position \((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)'
        for comp_type, x1, y1, x2, y2 in re.findall(added_pattern, description):
            changes['additions'].append({
                'type': comp_type,
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'center': [(int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2],
                'width': int(x2) - int(x1),
                'height': int(y2) - int(y1),
                'area': (int(x2) - int(x1)) * (int(y2) - int(y1)),
                'change_type': 'addition',
                'description': f"Added {comp_type}"
            })

        # 解析Removed
        removed_pattern = r'Removed (\w+) from position \((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)'
        for comp_type, x1, y1, x2, y2 in re.findall(removed_pattern, description):
            changes['removals'].append({
                'type': comp_type,
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'center': [(int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2],
                'width': int(x2) - int(x1),
                'height': int(y2) - int(y1),
                'area': (int(x2) - int(x1)) * (int(y2) - int(y1)),
                'change_type': 'removal',
                'description': f"Removed {comp_type}"
            })

        # 解析Moved
        moved_pattern = r'(\w+) from \((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\) to \((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)'
        for comp_type, fx1, fy1, fx2, fy2, tx1, ty1, tx2, ty2 in re.findall(moved_pattern, description):
            changes['movements'].append({
                'type': comp_type,
                'from_bbox': [int(fx1), int(fy1), int(fx2), int(fy2)],
                'to_bbox': [int(tx1), int(ty1), int(tx2), int(ty2)],
                'from_center': [(int(fx1) + int(fx2)) // 2, (int(fy1) + int(fy2)) // 2],
                'to_center': [(int(tx1) + int(tx2)) // 2, (int(ty1) + int(ty2)) // 2],
                'change_type': 'movement',
                'description': f"Moved {comp_type}"
            })

        # 合并所有组件
        changes['all_components'] = (
                changes['additions'] +
                changes['removals'] +
                changes['movements']
        )

        # 统计信息
        change_types = {}
        component_types = {}
        for comp in changes['all_components']:
            change_type = comp['change_type']
            comp_type = comp['type']
            change_types[change_type] = change_types.get(change_type, 0) + 1
            component_types[comp_type] = component_types.get(comp_type, 0) + 1

        changes['statistics'] = {
            'change_type_distribution': change_types,
            'component_type_distribution': component_types,
            'total_changes': len(changes['all_components'])
        }

        return changes

    def assess_overall_quality(self, detailed_changes: Dict,
                               stage1_results: Dict,
                               stage2_results: Dict,
                               stage3_results: Dict) -> Dict:
        """综合评估质量"""
        stats = detailed_changes['statistics']

        # 视觉变化检测
        visual_change_prob = stage1_results.get('change_probability', 0)
        visual_change_detected = visual_change_prob > 0.05

        # 对齐分数
        alignment_score = stage2_results.get('alignment_score', 0)

        # 短语对齐质量
        correspondences = stage3_results.get('correspondences', [])
        phrase_alignment_score = 0.0
        if correspondences:
            phrase_alignment_score = np.mean([c.get('max_score', 0) for c in correspondences])

        # 综合置信度
        if stats['total_changes'] > 0:
            overall_confidence = (
                    visual_change_prob * 0.3 +
                    alignment_score * 0.4 +
                    phrase_alignment_score * 0.3
            )
        else:
            overall_confidence = visual_change_prob if not visual_change_detected else 0.0

        overall_confidence = np.clip(overall_confidence, 0.0, 1.0)

        # 验证结果
        change_validated = (
                visual_change_detected and
                stats['total_changes'] > 0 and
                alignment_score > 0.6 and
                overall_confidence > 0.5
        )

        return {
            'alignment_score': alignment_score,
            'phrase_alignment_score': phrase_alignment_score,
            'visual_change_probability': visual_change_prob,
            'overall_confidence': overall_confidence,
            'described_changes_count': stats['total_changes'],
            'visual_change_detected': visual_change_detected,
            'change_validated': change_validated,
            'summary': self.generate_quality_summary(
                stats, visual_change_detected, change_validated
            )
        }

    def generate_quality_summary(self, stats: Dict, visual_change: bool, validated: bool) -> str:
        """生成质量摘要"""
        if validated:
            return f" 验证通过：{stats['total_changes']}个变化与视觉检测一致"

        if stats['total_changes'] == 0 and not visual_change:
            return "无变化检测"

        if stats['total_changes'] == 0 and visual_change:
            return "有视觉变化但无描述"

        if stats['total_changes'] > 0 and not visual_change:
            return "有描述变化但无显著视觉变化"

        return "变化需要进一步验证"

    def create_integrated_visualization(self, results: Dict, output_prefix: str):
        """创建集成可视化报告"""
        try:
            fig = plt.figure(figsize=(28, 24))

            # 加载图像
            ref_img = Image.open(results['reference_image']).convert('RGB').resize((340, 340))
            tar_img = Image.open(results['target_image']).convert('RGB').resize((340, 340))

            # 设置网格
            gs = plt.GridSpec(3, 3, hspace=0.4, wspace=0.4)

            # 1. 参考图像
            ax1 = plt.subplot(gs[0, 0])
            ax1.imshow(ref_img)
            ax1.set_title('Reference Image\n(Original GUI)', fontsize=24, fontweight='bold', pad=20)
            ax1.axis('off')

            # 标注移除组件
            for removal in results['detailed_changes']['removals'][:2]:
                bbox = removal['bbox']
                # 缩放坐标到显示尺寸
                scale_x = 340 / 144
                scale_y = 340 / 256
                scaled_bbox = [bbox[0] * scale_x, bbox[1] * scale_y, bbox[2] * scale_x, bbox[3] * scale_y]

                rect = patches.Rectangle(
                    (scaled_bbox[0], scaled_bbox[1]),
                    scaled_bbox[2] - scaled_bbox[0], scaled_bbox[3] - scaled_bbox[1],
                    linewidth=3, edgecolor=self.colors['removal'],
                    facecolor='none', linestyle='--', alpha=0.8
                )
                ax1.add_patch(rect)

            # 2. 目标图像
            ax2 = plt.subplot(gs[0, 1])
            ax2.imshow(tar_img)
            ax2.set_title('Target Image\n(Modified GUI)', fontsize=24, fontweight='bold', pad=20)
            ax2.axis('off')

            # 标注新增组件
            for addition in results['detailed_changes']['additions'][:2]:
                bbox = addition['bbox']
                scale_x = 340 / 144
                scale_y = 340 / 256
                scaled_bbox = [bbox[0] * scale_x, bbox[1] * scale_y, bbox[2] * scale_x, bbox[3] * scale_y]

                rect = patches.Rectangle(
                    (scaled_bbox[0], scaled_bbox[1]),
                    scaled_bbox[2] - scaled_bbox[0], scaled_bbox[3] - scaled_bbox[1],
                    linewidth=3, edgecolor=self.colors['addition'],
                    facecolor='none', linestyle='-', alpha=0.8
                )
                ax2.add_patch(rect)

            # 3. 热力图
            ax3 = plt.subplot(gs[0, 2])
            diff_map = results['stage1_results'].get('diff_map')
            if diff_map is not None:
                im = ax3.imshow(diff_map, cmap='hot', vmin=0, vmax=1)
                ax3.set_title('Stage1: Visual Change Heatmap', fontsize=24, fontweight='bold', pad=20)
                plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
            else:
                ax3.text(0.5, 0.5, 'No heatmap data', ha='center', va='center', fontsize=20)
                ax3.set_title('Stage1: Visual Change Heatmap', fontsize=24, fontweight='bold', pad=20)

            # 4. Stage1变化mask
            ax4 = plt.subplot(gs[1, 0])
            change_mask = results['stage1_results'].get('change_mask')
            if change_mask is not None:
                ax4.imshow(change_mask, cmap='binary')
                ax4.set_title('Stage1: Change Detection Mask', fontsize=24, fontweight='bold', pad=20)
            else:
                ax4.text(0.5, 0.5, 'No mask data', ha='center', va='center', fontsize=20)
                ax4.set_title('Stage1: Change Detection Mask', fontsize=24, fontweight='bold', pad=20)

            # 5. Stage2对齐结果
            ax5 = plt.subplot(gs[1, 1])
            ax5.axis('off')

            alignment_text = "Stage2: Multi-modal Alignment\n"
            alignment_text += "=" * 30 + "\n\n"

            alignment_score = results['stage2_results'].get('alignment_score', 0)
            alignment_text += f"Alignment Score: {alignment_score:.4f}\n\n"

            visual_feat_norm = torch.norm(results['stage2_results'].get('visual_features', torch.zeros(1, 1))).item()
            text_feat_norm = torch.norm(results['stage2_results'].get('text_features', torch.zeros(1, 1))).item()

            alignment_text += f"Visual Feature Norm: {visual_feat_norm:.3f}\n"
            alignment_text += f"Text Feature Norm: {text_feat_norm:.3f}\n"
            alignment_text += f"Similarity: {alignment_score:.3f}"

            ax5.text(0.1, 0.95, alignment_text, fontsize=20,
                     verticalalignment='top', transform=ax5.transAxes,
                     bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9))

            # 6. Stage3短语对齐
            ax6 = plt.subplot(gs[1, 2])
            ax6.axis('off')

            phrase_text = "Stage3: Phrase-Patch Alignment\n"
            phrase_text += "=" * 30 + "\n\n"

            correspondences = results['stage3_results'].get('correspondences', [])
            phrase_text += f"Phrases found: {len(correspondences)}\n\n"

            for i, corr in enumerate(correspondences[:3]):
                phrase_text += f"Phrase {i + 1}:\n"
                phrase_text += f"  Max score: {corr.get('max_score', 0):.3f}\n"
                phrase_text += f"  Top patches: {len(corr.get('top_patches', []))}\n\n"

            if correspondences:
                avg_score = np.mean([c.get('max_score', 0) for c in correspondences])
                phrase_text += f"Avg match score: {avg_score:.3f}"

            ax6.text(0.1, 0.95, phrase_text, fontsize=20,
                     verticalalignment='top', transform=ax6.transAxes,
                     bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9))

            # 7. 详细变化列表
            ax7 = plt.subplot(gs[2, 0])
            ax7.axis('off')

            changes_text = " Detailed Changes\n"
            changes_text += "=" * 20 + "\n\n"

            changes = results['detailed_changes']
            changes_text += f"Total: {changes['statistics']['total_changes']}\n"
            changes_text += f"Added: {len(changes['additions'])}\n"
            changes_text += f"Removed: {len(changes['removals'])}\n"
            changes_text += f"Moved: {len(changes['movements'])}\n\n"

            # 显示前3个变化
            for i, comp in enumerate(changes['all_components'][:3]):
                changes_text += f"{i + 1}. {comp['description']}\n"

            ax7.text(0.1, 0.95, changes_text, fontsize=20,
                     verticalalignment='top', transform=ax7.transAxes,
                     bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9))

            # 8. 综合评估
            ax8 = plt.subplot(gs[2, 1])
            ax8.axis('off')

            assessment = results['overall_assessment']
            assessment_text = "Overall Assessment\n"
            assessment_text += "=" * 20 + "\n\n"

            assessment_text += f"Alignment: {assessment['alignment_score']:.2%}\n"
            assessment_text += f"Phrase Alignment: {assessment['phrase_alignment_score']:.2%}\n"
            assessment_text += f"Overall Confidence: {assessment['overall_confidence']:.2%}\n\n"

            assessment_text += f"Validation: {'PASS' if assessment['change_validated'] else 'FAIL'}\n\n"
            assessment_text += f"Summary:\n{assessment['summary']}"

            color = self.colors['success'] if assessment['change_validated'] else self.colors['warning']
            ax8.text(0.1, 0.95, assessment_text, fontsize=20,
                     verticalalignment='top', transform=ax8.transAxes,
                     bbox=dict(boxstyle='round', facecolor=color, alpha=0.2, edgecolor=color, linewidth=3))

            # 9. 模型信息
            ax9 = plt.subplot(gs[2, 2])
            ax9.axis('off')

            model_text = "Model Information\n"
            model_text += "=" * 20 + "\n\n"
            model_text += f"Stage1: Visual Model\n"
            model_text += f"Stage2: Multi-modal Model\n"
            model_text += f"Stage3: Phrase Model\n\n"
            model_text += f"Analysis Time: {results['analysis_time']:.1f}s\n"
            model_text += f"Device: {self.device}\n\n"
            model_text += f"Total Parameters:\n"
            model_text += f"  Stage1: {sum(p.numel() for p in self.stage1_model.parameters()):,}\n"
            model_text += f"  Stage2: {sum(p.numel() for p in self.stage2_model.parameters()):,}\n"
            model_text += f"  Stage3: {sum(p.numel() for p in self.stage3_model.parameters()):,}"

            ax9.text(0.1, 0.95, model_text, fontsize=18,
                     verticalalignment='top', transform=ax9.transAxes,
                     bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9))

            # 主标题
            fig.suptitle('Three-Stage GUI Change Analysis - Integrated Report',
                         fontsize=32, fontweight='bold', y=0.98)

            # 副标题
            ref_name = Path(results['reference_image']).name
            tar_name = Path(results['target_image']).name
            fig.text(0.5, 0.95,
                     f"Reference: {ref_name}  |  Target: {tar_name}",
                     fontsize=20, ha='center', style='italic')

            # 保存
            output_path = f"{output_prefix}_integrated_report.png"
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"集成可视化报告保存: {output_path}")

        except Exception as e:
            print(f"可视化创建失败: {e}")
            import traceback
            traceback.print_exc()

    def save_analysis_results(self, results: Dict, output_prefix: str):
        """保存分析结果"""
        try:
            # 创建可序列化的副本
            def make_serializable(obj):
                if isinstance(obj, (str, int, float, bool, type(None))):
                    return obj
                elif isinstance(obj, (list, tuple)):
                    return [make_serializable(item) for item in obj]
                elif isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, torch.Tensor):
                    return obj.cpu().numpy().tolist()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.generic):
                    return obj.item()
                else:
                    return str(obj)

            serializable_results = make_serializable(results)

            # 保存JSON
            json_path = f"{output_prefix}_results.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)

            print(f"分析结果保存: {json_path}")

        except Exception as e:
            print(f"结果保存失败: {e}")

    def print_summary(self, results: Dict):
        """打印分析摘要"""
        print(f"\n{'=' * 60}")
        print("分析摘要")
        print(f"{'=' * 60}")

        # 基础信息
        print(f"图像:")
        print(f"  参考: {Path(results['reference_image']).name}")
        print(f"  目标: {Path(results['target_image']).name}")
        print(f"  分析时间: {results['analysis_time']:.2f}秒")

        # 变化统计
        changes = results['detailed_changes']['statistics']
        print(f"\n 变化统计:")
        print(f"  总变化: {changes['total_changes']}")
        print(f"  变化类型: {changes['change_type_distribution']}")
        print(f"  组件类型: {changes['component_type_distribution']}")

        # 阶段结果
        print(f"\n阶段结果:")
        stage1_prob = results['stage1_results'].get('change_probability', 0)
        print(f"  Stage1 - 视觉变化概率: {stage1_prob:.2%}")

        stage2_score = results['stage2_results'].get('alignment_score', 0)
        print(f"  Stage2 - 对齐分数: {stage2_score:.2%}")

        correspondences = results['stage3_results'].get('correspondences', [])
        print(f"  Stage3 - 短语对齐数量: {len(correspondences)}")

        # 综合评估
        assessment = results['overall_assessment']
        print(f"\n 综合评估:")
        print(f"  对齐分数: {assessment['alignment_score']:.2%}")
        print(f"  短语对齐: {assessment['phrase_alignment_score']:.2%}")
        print(f"  整体置信度: {assessment['overall_confidence']:.2%}")
        print(f"  验证结果: {' PASS' if assessment['change_validated'] else ' FAIL'}")
        print(f"  总结: {assessment['summary']}")


# ============ 配置类 ============
class Config:
    """配置文件"""

    def __init__(self):
        self.hidden_dim = 768
        self.max_text_len = 512
        self.max_components = 20
        self.visual_dim = 768
        self.image_model = "vit-large-patch16-224"
        self.model_root = "/home/common-dir/models"
        self.output_dir = "./integrated_analysis_results"
        self.learning_rate = 1e-4
        self.batch_size = 1  # 推理时使用batch size 1
        self.num_workers = 0
        self.pin_memory = False
        self.cuda_available = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda_available else 'cpu')
        self.mixed_precision = False
        self.gradient_checkpointing = False


# ============ 主函数 ============
def main():
    """主函数"""
    # 加载配置
    config = Config()

    # 创建分析器
    analyzer = IntegratedGUIAnalyzer(config)

    # 测试用例
    test_cases = [
        {
            'name': 'Login Screen Analysis',
            'ref_image': "/home/common-dir/data/gui/settings/23319.png",
            'tar_image': "/home/common-dir/data/gui/settings/57622.png",
            'description': """Added TextView at position (0, 83, 144, 188); Added View at position (0, 9, 144, 38); Added SwitchMain at position (0, 37, 72, 81); Added SwitchSlider at position (72, 37, 144, 81); TextView from (26, 130, 40, 137) to (0, 83, 144, 188); TextView from (26, 111, 49, 118) to (0, 83, 144, 188)"""
        }
    ]

    for i, test in enumerate(test_cases):
        print(f"\n{'=' * 60}")
        print(f"测试用例 {i + 1}: {test['name']}")
        print(f"{'=' * 60}")

        # 运行集成分析
        results = analyzer.analyze_with_models(
            test['ref_image'],
            test['tar_image'],
            test['description'],
            output_prefix=f"test_case_{i + 1}"
        )

        if 'error' not in results:
            print(f"\n 分析完成!")
            print(f"  可视化报告已保存")
            print(f"  详细结果已保存")
        else:
            print(f"\n 分析失败: {results['error']}")


if __name__ == "__main__":
    main()
