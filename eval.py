#!/usr/bin/env python3
"""
短语级对应关系评估
"""
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from dataloder import create_data_loader
from model3 import Stage3PhraseContrastiveModel


class PhraseAlignmentEvaluator:
    """短语对齐评估器"""

    def __init__(self, checkpoint_path: str):
        self.config = config
        self.device = config.device

        print(f"加载模型: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # 初始化模型
        self.model = Stage3PhraseContrastiveModel(
            stage2_checkpoint="",  # 不需要stage2
            config=self.config
        ).to(self.device)

        # 加载权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"模型加载完成")

        # 数据加载器
        self.test_loader = create_data_loader('test', self.config, is_stage1=False, shuffle=False)

    def evaluate_sample(self, idx: int = 0):
        """评估单个样本"""
        batch = None
        for i, b in enumerate(self.test_loader):
            if i == idx:
                batch = b
                break

        if batch is None:
            print(f"找不到样本 {idx}")
            return

        with torch.no_grad():
            outputs = self.model(
                batch['ref_image'].to(self.device),
                batch['tar_image'].to(self.device),
                batch['text_tokens'].to(self.device),
                batch['ref_components'].to(self.device),
                batch['tar_components'].to(self.device),
                batch.get('text', [''] * len(batch['ref_image']))
            )

        # 解析结果
        self.visualize_results(batch, outputs, idx)

        # 计算指标
        self.compute_metrics(batch, outputs)

    def visualize_results(self, batch, outputs, sample_idx: int = 0):
        """可视化结果"""
        import matplotlib.patches as patches

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 参考图像
        ref_img = batch['ref_image'][sample_idx].cpu().permute(1, 2, 0).numpy()
        axes[0, 0].imshow(ref_img)
        axes[0, 0].set_title('Reference Image')
        axes[0, 0].axis('off')

        # 目标图像
        tar_img = batch['tar_image'][sample_idx].cpu().permute(1, 2, 0).numpy()
        axes[0, 1].imshow(tar_img)
        axes[0, 1].set_title('Target Image')
        axes[0, 1].axis('off')

        # 真实变化掩码
        mask = batch['mask'][sample_idx].cpu().numpy()
        axes[0, 2].imshow(mask, cmap='hot')
        axes[0, 2].set_title('Ground Truth Mask')
        axes[0, 2].axis('off')

        # 短语-Patch对应
        axes[1, 0].imshow(tar_img)
        parsed_phrases = outputs.get('parsed_phrases', [])
        correspondences = outputs.get('correspondences', [])

        if sample_idx < len(parsed_phrases):
            phrases = parsed_phrases[sample_idx]

            colors = ['red', 'blue', 'green', 'orange', 'purple']

            for phrase_idx, phrase in enumerate(phrases):
                color = colors[phrase_idx % len(colors)]

                # 查找对应关系
                for corr in correspondences:
                    if corr['batch_idx'] == sample_idx and corr['phrase_idx'] == phrase_idx:
                        # 绘制top-3 patches
                        for patch_idx, patch in enumerate(corr['top_patches'][:3]):
                            bbox = patch['bbox']
                            rect = patches.Rectangle(
                                (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                linewidth=2, edgecolor=color, facecolor='none',
                                alpha=0.7 - 0.2 * patch_idx
                            )
                            axes[1, 0].add_patch(rect)

                        # 添加短语文本
                        phrase_text = f"{phrase.get('type', '')} {phrase.get('component', '')}"
                        axes[1, 0].text(
                            10, 20 + phrase_idx * 25,
                            phrase_text[:30],
                            color='white', backgroundcolor=color,
                            fontsize=9, alpha=0.8
                        )
                        break

        axes[1, 0].set_title('Phrase-Patch Correspondence')
        axes[1, 0].axis('off')

        # 短语热力图
        phrase_heatmaps = outputs.get('phrase_heatmaps', [])
        if phrase_heatmaps:
            # 找到属于当前样本的热力图
            sample_heatmaps = []
            phrase_batch_indices = outputs.get('phrase_batch_indices', [])

            for i, batch_idx in enumerate(phrase_batch_indices):
                if batch_idx.item() == sample_idx:
                    if i < len(phrase_heatmaps):
                        sample_heatmaps.append(phrase_heatmaps[i])

            if sample_heatmaps:
                # 合并热力图
                combined = torch.zeros_like(sample_heatmaps[0])
                for heatmap in sample_heatmaps:
                    combined = torch.max(combined, heatmap)

                axes[1, 1].imshow(tar_img, alpha=0.6)
                axes[1, 1].imshow(combined.cpu().numpy(), cmap='jet', alpha=0.4)
                axes[1, 1].set_title('Phrase Activation Heatmap')
                axes[1, 1].axis('off')
            else:
                axes[1, 1].text(0.5, 0.5, 'No Phrases',
                                ha='center', va='center', fontsize=12)
                axes[1, 1].axis('off')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Heatmaps',
                            ha='center', va='center', fontsize=12)
            axes[1, 1].axis('off')

        # 文本描述
        text = batch.get('text', [''])[sample_idx]
        axes[1, 2].text(0.1, 0.9, 'Text Description:',
                        fontsize=12, fontweight='bold')
        axes[1, 2].text(0.1, 0.7, text[:200] + ('...' if len(text) > 200 else ''),
                        fontsize=10, verticalalignment='top')
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(f'phrase_alignment_sample_{sample_idx}.png', dpi=150, bbox_inches='tight')
        plt.show()

    def compute_metrics(self, batch, outputs):
        """计算评估指标"""
        metrics = {}

        # 1. 短语数量
        num_phrases = outputs.get('num_phrases', 0)
        metrics['num_phrases'] = num_phrases

        # 2. 平均匹配分数
        correspondences = outputs.get('correspondences', [])
        if correspondences:
            match_scores = [corr['max_score'] for corr in correspondences]
            metrics['avg_match_score'] = np.mean(match_scores)
            metrics['max_match_score'] = np.max(match_scores)
            metrics['min_match_score'] = np.min(match_scores)

        # 3. 短语类型分布
        parsed_phrases = outputs.get('parsed_phrases', [])
        phrase_types = {}
        for phrases in parsed_phrases:
            for phrase in phrases:
                p_type = phrase.get('type', 'unknown')
                phrase_types[p_type] = phrase_types.get(p_type, 0) + 1
        metrics['phrase_types'] = phrase_types

        print(f"\n评估指标:")
        print(f"  短语数量: {metrics.get('num_phrases', 0)}")
        print(f"  平均匹配分数: {metrics.get('avg_match_score', 0):.4f}")
        print(f"  最大匹配分数: {metrics.get('max_match_score', 0):.4f}")

        print(f"\n短语类型分布:")
        for p_type, count in phrase_types.items():
            print(f"  {p_type}: {count}")

        return metrics

    def batch_evaluate(self, num_samples: int = 10):
        """批量评估"""
        all_metrics = []

        for idx in range(min(num_samples, len(self.test_loader))):
            print(f"\n评估样本 {idx + 1}/{num_samples}")

            batch = None
            for i, b in enumerate(self.test_loader):
                if i == idx:
                    batch = b
                    break

            if batch is None:
                continue

            with torch.no_grad():
                outputs = self.model(
                    batch['ref_image'].to(self.device),
                    batch['tar_image'].to(self.device),
                    batch['text_tokens'].to(self.device),
                    batch['ref_components'].to(self.device),
                    batch['tar_components'].to(self.device),
                    batch.get('text', [''] * len(batch['ref_image']))
                )

            metrics = self.compute_metrics(batch, outputs)
            all_metrics.append(metrics)

        # 计算总体统计
        if all_metrics:
            avg_phrases = np.mean([m.get('num_phrases', 0) for m in all_metrics])
            avg_match = np.mean([m.get('avg_match_score', 0) for m in all_metrics])

            print(f"\n{'=' * 60}")
            print(f"总体统计 ({len(all_metrics)} 个样本):")
            print(f"  平均每样本短语数: {avg_phrases:.2f}")
            print(f"  平均匹配分数: {avg_match:.4f}")
            print(f"{'=' * 60}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='短语对齐评估')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='第三阶段模型检查点')
    parser.add_argument('--sample-idx', type=int, default=0,
                        help='评估的样本索引')
    parser.add_argument('--batch-eval', action='store_true',
                        help='批量评估多个样本')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='批量评估的样本数量')

    args = parser.parse_args()

    evaluator = PhraseAlignmentEvaluator(args.checkpoint)

    if args.batch_eval:
        evaluator.batch_evaluate(args.num_samples)
    else:
        evaluator.evaluate_sample(args.sample_idx)


if __name__ == "__main__":
    main()