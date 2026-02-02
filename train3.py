#!/usr/bin/env python3
"""
第三阶段训练：短语级对比学习
建立短语-token ⟷ 图像-patch的双向细粒度对应
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from pathlib import Path
import numpy as np
import time
import os
import sys
from tqdm import tqdm
import wandb
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from dataloder import create_data_loader
from model3 import Stage3PhraseContrastiveModel
from memory import memory_monitor

# ============ 第三阶段特定配置 ============
# 这些配置直接在train3.py中定义，避免修改原有的config.py
STAGE3_CONFIG = {
    'stage3_epochs': 15,
    'max_phrases_per_sample': 5,
    'phrase_contrastive_temp': 0.07,
    'phrase_hidden_dim': 768,  # 修改为768，与第二阶段一致
    'learning_rate_stage3': 3e-4,
    'weight_decay_stage3': 0.01,
    'grad_accum_steps_stage3': 2,
}


def auto_adjust_batch_size_stage3():
    """根据GPU内存自动调整第三阶段batch size"""
    if not config.cuda_available:
        return config.batch_size

    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3  # GB

    if gpu_memory >= 24:  # A100 40GB
        batch_size = 8
    elif gpu_memory >= 16:  # V100 16GB
        batch_size = 4
    elif gpu_memory >= 8:  # 2080Ti 11GB
        batch_size = 2
    else:  # 低端GPU
        batch_size = 1

    print(f"根据GPU内存 ({gpu_memory:.1f}GB) 调整第三阶段batch size为: {batch_size}")
    return batch_size


class Stage3PhraseTrainer:
    """第三阶段短语级对比学习训练器"""

    def __init__(self, stage2_checkpoint: str, use_wandb: bool = True):
        self.config = config
        self.device = config.device
        self.use_wandb = use_wandb

        # 加载第三阶段配置
        self.stage3_config = STAGE3_CONFIG

        print(f"\n{'=' * 60}")
        print("第三阶段训练：短语级对比学习")
        print(f"{'=' * 60}")
        print(f"设备: {self.device}")
        print(f"目标: 建立短语↔Patch细粒度对应")
        print(f"Stage2隐藏维度: {self.config.hidden_dim}")
        print(f"Stage3配置的隐藏维度: {self.stage3_config['phrase_hidden_dim']}")

        # 检查检查点
        if not Path(stage2_checkpoint).exists():
            print(f"⚠️ 警告：找不到stage2检查点 {stage2_checkpoint}")
            print("将使用随机初始化的Stage2模型")

        self.stage2_checkpoint = stage2_checkpoint

        # 调整配置 - 第三阶段需要更多内存
        if self.config.cuda_available:
            self.config.batch_size = auto_adjust_batch_size_stage3()

        print(f"\n第三阶段配置:")
        print(f"  训练轮数: {self.stage3_config['stage3_epochs']}")
        print(f"  批次大小: {self.config.batch_size}")
        print(f"  学习率: {self.stage3_config['learning_rate_stage3']}")
        print(f"  梯度累积: {self.stage3_config['grad_accum_steps_stage3']}")
        print(f"  最大短语数: {self.stage3_config['max_phrases_per_sample']}")
        print(f"  短语隐藏维度: {self.stage3_config['phrase_hidden_dim']}")
        print(f"  对比温度: {self.stage3_config['phrase_contrastive_temp']}")

        # ============ 初始化模型 ============
        print("\n初始化短语级对比学习模型...")
        memory_monitor.print_memory_stats("初始化前")

        # 创建模型时使用原有配置，不要修改hidden_dim
        self.model = Stage3PhraseContrastiveModel(
            stage2_checkpoint=self.stage2_checkpoint,
            config=self.config  # 直接使用原有配置
        ).to(self.device)

        # 打印参数统计
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        print(f"模型参数统计:")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  冻结参数: {frozen_params:,}")
        print(f"  训练比例: {trainable_params / total_params * 100:.1f}%")

        # ============ 优化器配置 ============
        # 分层学习率
        phrase_params = []
        patch_params = []
        contrastive_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if 'phrase_encoder' in name:
                phrase_params.append(param)
            elif 'patch_encoder' in name:
                patch_params.append(param)
            elif 'contrastive_module' in name:
                contrastive_params.append(param)
            elif 'phrase_projection' in name:
                other_params.append(param)  # 短语投影层
            elif 'visual_adapter' in name:
                other_params.append(param)  # 视觉适配器
            elif 'visualization_head' in name:
                other_params.append(param)  # 可视化头

        print(f"\n优化器参数分组:")
        print(f"  短语编码器参数: {len(phrase_params)}层")
        print(f"  Patch编码器参数: {len(patch_params)}层")
        print(f"  对比学习参数: {len(contrastive_params)}层")
        print(f"  其他参数: {len(other_params)}层")

        self.optimizer = optim.AdamW([
            {
                'params': phrase_params,
                'lr': self.stage3_config['learning_rate_stage3'],
                'weight_decay': self.stage3_config['weight_decay_stage3']
            },
            {
                'params': patch_params,
                'lr': self.stage3_config['learning_rate_stage3'] * 0.67,  # 2e-4
                'weight_decay': self.stage3_config['weight_decay_stage3'] * 0.1
            },
            {
                'params': contrastive_params,
                'lr': self.stage3_config['learning_rate_stage3'] * 0.33,  # 1e-4
                'weight_decay': self.stage3_config['weight_decay_stage3'] * 0.1
            },
            {
                'params': other_params,
                'lr': self.stage3_config['learning_rate_stage3'] * 0.33,
                'weight_decay': self.stage3_config['weight_decay_stage3']
            }
        ])

        # ============ 数据加载器 ============
        print(f"\n加载数据集...")
        self.train_loader = create_data_loader('train', self.config, is_stage1=False)
        self.val_loader = create_data_loader('val', self.config, is_stage1=False)

        total_steps = self.stage3_config['stage3_epochs'] * len(self.train_loader)

        print(f"数据集统计:")
        print(f"  训练集: {len(self.train_loader.dataset)} 样本")
        print(f"  验证集: {len(self.val_loader.dataset)} 样本")
        print(f"  总训练步数: {total_steps}")

        # ============ 学习率调度器 ============
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[
                self.stage3_config['learning_rate_stage3'],  # phrase_params
                self.stage3_config['learning_rate_stage3'] * 0.67,  # patch_params
                self.stage3_config['learning_rate_stage3'] * 0.33,  # contrastive_params
                self.stage3_config['learning_rate_stage3'] * 0.33  # other_params
            ],
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )

        # ============ 混合精度训练 ============
        if self.config.mixed_precision and self.config.cuda_available:
            self.scaler = GradScaler('cuda')
            print("启用混合精度训练")
        else:
            self.scaler = None
            print("禁用混合精度训练")

        # ============ 损失函数 ============
        # 对比学习损失已经在模型中计算
        # 添加辅助对齐损失
        self.alignment_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')

        # ============ 输出目录 ============
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.stage_dir = self.config.output_dir / f"stage3{timestamp}"
        self.checkpoint_dir = self.stage_dir / "checkpoints"
        self.log_dir = self.stage_dir / "logs"
        self.viz_dir = self.stage_dir / "visualizations"

        # 创建目录
        for dir_path in [self.stage_dir, self.checkpoint_dir, self.log_dir, self.viz_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        print(f"\n输出目录: {self.stage_dir}")

        # ============ 训练状态 ============
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_alignment_score = 0.0

        # 训练历史
        self.train_history = []
        self.val_history = []

        # 短语统计
        self.phrase_stats = {
            'total_phrases': 0,
            'avg_phrases_per_sample': 0,
            'phrase_types': {'addition': 0, 'removal': 0, 'movement': 0}
        }

        memory_monitor.print_memory_stats("初始化后")

        # ============ 初始化Wandb ============
        if self.use_wandb:
            try:
                # 合并配置
                wandb_config = self.config.__dict__.copy()
                wandb_config.update(self.stage3_config)

                wandb.init(
                    project="gui-change-detection",
                    name=f"stage3{timestamp}",
                    config=wandb_config,
                    dir=str(self.log_dir)
                )
                wandb.watch(self.model, log='all', log_freq=50)
                print("Wandb初始化成功")
            except Exception as e:
                print(f"Wandb初始化失败: {e}")
                self.use_wandb = False

    def compute_total_loss(self, outputs, batch):
        """计算总损失 - 修复设备不匹配问题"""
        losses = {}

        try:
            # 获取设备信息
            device = outputs.get('patch_features', torch.zeros(1).to(self.device)).device

            # 1. 对比学习损失（主要损失）
            contrastive_loss = outputs.get('total_contrastive_loss', torch.tensor(0.0, device=device))
            losses['contrastive'] = contrastive_loss

            # 2. 短语到patch损失
            phrase_to_patch = outputs.get('loss_phrase_to_patch', torch.tensor(0.0, device=device))
            losses['phrase_to_patch'] = phrase_to_patch

            # 3. patch到短语损失
            patch_to_phrase = outputs.get('loss_patch_to_phrase', torch.tensor(0.0, device=device))
            losses['patch_to_phrase'] = patch_to_phrase

            # 4. 对齐一致性损失（可选）
            if 'stage2_features' in outputs and outputs['stage2_features'] is not None:
                alignment_scores = outputs['stage2_features'].get('alignment_scores')
                if alignment_scores is not None and 'has_change' in batch:
                    # 确保标签在正确的设备上
                    alignment_labels = batch['has_change'].unsqueeze(1).float().to(device)
                    alignment_loss = self.alignment_loss_fn(alignment_scores, alignment_labels)
                    losses['alignment'] = alignment_loss * 0.1  # 较小的权重
                else:
                    losses['alignment'] = torch.tensor(0.0, device=device)
            else:
                losses['alignment'] = torch.tensor(0.0, device=device)

            # 5. 总损失
            total_loss = sum(losses.values())
            losses['total'] = total_loss

            # 6. 短语统计
            num_phrases = outputs.get('num_phrases', 0)
            parsed_phrases = outputs.get('parsed_phrases', [])

            if num_phrases > 0 and parsed_phrases:
                for phrases in parsed_phrases:
                    for phrase in phrases:
                        phrase_type = phrase.get('type', 'unknown')
                        if phrase_type in self.phrase_stats['phrase_types']:
                            self.phrase_stats['phrase_types'][phrase_type] += 1

        except Exception as e:
            print(f"损失计算错误: {e}")
            import traceback
            traceback.print_exc()
            # 返回默认损失
            device = outputs.get('patch_features', torch.zeros(1).to(self.device)).device
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            losses = {'total': total_loss}

        return total_loss, losses

    def train_epoch(self, epoch: int):
        """训练一个epoch - 修复梯度缩放器使用"""
        self.model.train()
        epoch_losses = {}
        total_phrases = 0
        total_samples = 0

        # 设置梯度累积步数
        grad_accum_steps = self.stage3_config['grad_accum_steps_stage3']

        # 记录当前是否已经unscale过
        unscaled_in_step = False

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.stage3_config['stage3_epochs']}")

        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue

            try:
                # 准备数据
                ref_image = batch['ref_image'].to(self.device, non_blocking=True)
                tar_image = batch['tar_image'].to(self.device, non_blocking=True)
                text_tokens = batch['text_tokens'].to(self.device, non_blocking=True)
                ref_components = batch['ref_components'].to(self.device, non_blocking=True)
                tar_components = batch['tar_components'].to(self.device, non_blocking=True)

                # 获取differ_text
                differ_texts = batch.get('text', [''] * len(ref_image))

                # 确保组件数据在有效范围内
                if ref_components is not None:
                    ref_components[:, :, 0] = torch.clamp(ref_components[:, :, 0], 0, 19)
                if tar_components is not None:
                    tar_components[:, :, 0] = torch.clamp(tar_components[:, :, 0], 0, 19)

                # 前向传播
                if self.scaler:
                    with autocast('cuda'):
                        outputs = self.model(
                            ref_image, tar_image, text_tokens,
                            ref_components, tar_components, differ_texts
                        )
                        total_loss, loss_dict = self.compute_total_loss(outputs, batch)
                else:
                    outputs = self.model(
                        ref_image, tar_image, text_tokens,
                        ref_components, tar_components, differ_texts
                    )
                    total_loss, loss_dict = self.compute_total_loss(outputs, batch)

                # 检查损失有效性
                if not total_loss.requires_grad or torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"警告: 无效损失，跳过批次 {batch_idx}")
                    self.optimizer.zero_grad(set_to_none=True)
                    unscaled_in_step = False
                    continue

                # 梯度累积
                loss = total_loss / grad_accum_steps

                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # 梯度累积步骤
                if (batch_idx + 1) % grad_accum_steps == 0:
                    # 梯度裁剪 - 只在混合精度训练且需要裁剪时才unscale
                    if self.scaler and not unscaled_in_step:
                        self.scaler.unscale_(self.optimizer)
                        unscaled_in_step = True

                    # 检查梯度
                    grad_norm = 0.0
                    grad_params = 0
                    for param in self.model.parameters():
                        if param.grad is not None:
                            grad_norm += param.grad.norm().item() ** 2
                            grad_params += 1

                    if grad_params > 0:
                        grad_norm = grad_norm ** 0.5
                        if grad_norm > 1.0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    # 优化器步骤
                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.optimizer.zero_grad(set_to_none=True)
                    unscaled_in_step = False  # 重置标志
                    self.scheduler.step()
                    self.global_step += 1

                # 累积损失
                for key, value in loss_dict.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = []
                    epoch_losses[key].append(value.item())

                # 更新短语统计
                num_phrases = outputs.get('num_phrases', 0)
                total_phrases += num_phrases
                total_samples += ref_image.shape[0]

                # 更新进度条
                pbar.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'phrases': num_phrases,
                    'contrastive': f"{loss_dict.get('contrastive', 0):.4f}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
                    'step': self.global_step
                })

                # 定期记录和清理
                if self.global_step % 50 == 0:
                    if self.use_wandb:
                        wandb.log({
                            'train/step_loss': total_loss.item(),
                            'train/step_contrastive_loss': loss_dict.get('contrastive', 0),
                            'train/step_phrase_to_patch': loss_dict.get('phrase_to_patch', 0),
                            'train/step_patch_to_phrase': loss_dict.get('patch_to_phrase', 0),
                            'train/learning_rate': self.scheduler.get_last_lr()[0],
                            'train/step': self.global_step,
                            'train/num_phrases': num_phrases
                        })

                    memory_monitor.clear_cache()

            except Exception as e:
                print(f"批次 {batch_idx} 处理错误: {e}")
                import traceback
                traceback.print_exc()
                self.optimizer.zero_grad(set_to_none=True)
                unscaled_in_step = False
                continue

        # 计算平均损失
        avg_losses = {}
        for key, values in epoch_losses.items():
            if values:
                avg_losses[key] = np.mean(values)
            else:
                avg_losses[key] = 0.0

        # 更新短语统计
        if total_samples > 0:
            self.phrase_stats['total_phrases'] += total_phrases
            self.phrase_stats['avg_phrases_per_sample'] = total_phrases / total_samples

        return avg_losses

    def save_visualization(self, batch, outputs, epoch, batch_idx):
        """保存可视化示例"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle

            batch_size = min(2, batch['ref_image'].shape[0])  # 只保存前2个样本

            for i in range(batch_size):
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))

                # 参考图像
                ref_img = batch['ref_image'][i].cpu().permute(1, 2, 0).numpy()
                axes[0, 0].imshow(ref_img)
                axes[0, 0].set_title('Reference Image')
                axes[0, 0].axis('off')

                # 目标图像
                tar_img = batch['tar_image'][i].cpu().permute(1, 2, 0).numpy()
                axes[0, 1].imshow(tar_img)
                axes[0, 1].set_title('Target Image')
                axes[0, 1].axis('off')

                # Mask
                mask = batch['mask'][i].cpu().numpy()
                axes[0, 2].imshow(mask, cmap='hot')
                axes[0, 2].set_title('Change Mask')
                axes[0, 2].axis('off')

                # 短语-Patch对应关系
                parsed_phrases = outputs.get('parsed_phrases', [])
                if i < len(parsed_phrases):
                    phrases = parsed_phrases[i]
                    correspondences = outputs.get('correspondences', [])

                    # 显示目标图像
                    axes[1, 0].imshow(tar_img)

                    # 绘制短语对应的patch
                    for phrase_idx, phrase in enumerate(phrases):
                        # 查找对应关系
                        for corr in correspondences:
                            if corr['batch_idx'] == i and corr['phrase_idx'] == phrase_idx:
                                for patch in corr['top_patches'][:3]:  # 只显示前3个
                                    bbox = patch['bbox']
                                    rect = Rectangle(
                                        (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                        linewidth=2, edgecolor='r', facecolor='none'
                                    )
                                    axes[1, 0].add_patch(rect)

                                # 添加短语文本
                                axes[1, 0].text(
                                    10, 20 + phrase_idx * 20,
                                    f"{phrase.get('component', '')}",
                                    color='white', backgroundcolor='red',
                                    fontsize=8
                                )
                                break

                    axes[1, 0].set_title('Phrase-Patch Correspondence')
                    axes[1, 0].axis('off')

                # 短语热力图
                phrase_heatmaps = outputs.get('phrase_heatmaps', [])
                if phrase_heatmaps:
                    # 合并所有短语的热力图
                    combined_heatmap = torch.zeros(224, 224, device=self.device)
                    for heatmap in phrase_heatmaps:
                        if heatmap.shape == (224, 224):
                            combined_heatmap = torch.max(combined_heatmap, heatmap)

                    axes[1, 1].imshow(tar_img, alpha=0.7)
                    axes[1, 1].imshow(combined_heatmap.cpu().numpy(), cmap='jet', alpha=0.5)
                    axes[1, 1].set_title('Combined Phrase Heatmap')
                    axes[1, 1].axis('off')

                # 文本描述
                text = batch.get('text', [''])[i] if i < len(batch.get('text', [])) else ''
                axes[1, 2].text(0.1, 0.9, 'Text Description:', fontsize=12, fontweight='bold')
                axes[1, 2].text(0.1, 0.1, text[:100] + ('...' if len(text) > 100 else ''),
                                fontsize=9, verticalalignment='bottom')
                axes[1, 2].axis('off')

                plt.tight_layout()

                # 保存图像
                viz_path = self.viz_dir / f"epoch_{epoch}_step_{self.global_step}_sample_{i}.png"
                plt.savefig(viz_path, dpi=150, bbox_inches='tight')
                plt.close()

                # 保存到wandb
                if self.use_wandb:
                    wandb.log({
                        f"visualizations/sample_{i}": wandb.Image(str(viz_path)),
                        'step': self.global_step
                    })

                print(f"✅ 保存可视化: {viz_path}")

        except Exception as e:
            print(f"可视化保存失败: {e}")

    @torch.no_grad()
    def validate(self):
        """验证过程 - 修复设备问题"""
        self.model.eval()
        val_losses = []
        val_metrics = []

        print(f"\n验证模型 (Step {self.global_step})...")
        print(f"采样验证批次进行详细分析...")

        # 只验证部分批次以节省时间
        max_val_batches = min(20, len(self.val_loader))
        val_iterator = iter(self.val_loader)

        for batch_idx in tqdm(range(max_val_batches), desc="Validation"):
            try:
                batch = next(val_iterator)
                if batch is None:
                    continue

                ref_image = batch['ref_image'].to(self.device, non_blocking=True)
                tar_image = batch['tar_image'].to(self.device, non_blocking=True)
                text_tokens = batch['text_tokens'].to(self.device, non_blocking=True)
                ref_components = batch['ref_components'].to(self.device, non_blocking=True)
                tar_components = batch['tar_components'].to(self.device, non_blocking=True)
                differ_texts = batch.get('text', [''] * len(ref_image))

                outputs = self.model(
                    ref_image, tar_image, text_tokens,
                    ref_components, tar_components, differ_texts
                )

                # 确保batch数据在正确设备上
                batch_on_device = {}
                for key, value in batch.items():
                    if torch.is_tensor(value):
                        batch_on_device[key] = value.to(self.device)
                    else:
                        batch_on_device[key] = value

                total_loss, loss_dict = self.compute_total_loss(outputs, batch_on_device)
                val_losses.append(total_loss.item())

                # 收集指标
                metrics = {'val_loss': total_loss.item()}
                for k, v in loss_dict.items():
                    if torch.is_tensor(v):
                        metrics[f'val_{k}'] = v.item()
                    else:
                        metrics[f'val_{k}'] = v

                # 短语对齐质量指标
                num_phrases = outputs.get('num_phrases', 0)
                correspondences = outputs.get('correspondences', [])

                if num_phrases > 0 and correspondences:
                    # 计算平均匹配分数
                    match_scores = [corr['max_score'] for corr in correspondences]
                    avg_match_score = np.mean(match_scores) if match_scores else 0.0

                    metrics['val_avg_match_score'] = avg_match_score
                    metrics['val_num_phrases'] = num_phrases

                val_metrics.append(metrics)

            except Exception as e:
                print(f"验证批次 {batch_idx} 错误: {e}")
                continue

        if not val_losses:
            print("警告：验证集为空或所有批次都失败")
            return {'val_loss': float('inf'), 'alignment_score': 0.0}

        # 计算平均指标
        avg_metrics = {}
        if val_metrics:
            for key in val_metrics[0].keys():
                values = [m[key] for m in val_metrics if key in m]
                if values:
                    avg_metrics[key] = np.mean(values)

        avg_val_loss = np.mean(val_losses)
        alignment_score = 1.0 / (avg_val_loss + 1e-8)

        avg_metrics['val_loss'] = avg_val_loss
        avg_metrics['alignment_score'] = alignment_score

        # 打印重要指标
        print(f"\n验证结果:")
        print(f"  验证损失: {avg_val_loss:.4f}")
        print(f"  对齐分数: {alignment_score:.4f}")
        print(f"  总短语数: {self.phrase_stats['total_phrases']}")

        if 'val_avg_match_score' in avg_metrics:
            print(f"  平均匹配分数: {avg_metrics['val_avg_match_score']:.4f}")

        # 短语类型统计
        print(f"\n短语类型统计:")
        for phrase_type, count in self.phrase_stats['phrase_types'].items():
            print(f"  {phrase_type}: {count}")

        return avg_metrics

    def save_checkpoint(self, is_best: bool = False, suffix: str = ""):
        """保存检查点"""
        try:
            checkpoint = {
                'epoch': self.current_epoch,
                'global_step': self.global_step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_val_loss': self.best_val_loss,
                'best_alignment_score': self.best_alignment_score,
                'config': self.config.__dict__,
                'stage3_config': self.stage3_config,
                'train_history': self.train_history,
                'val_history': self.val_history,
                'phrase_stats': self.phrase_stats,
                'timestamp': time.time()
            }

            if self.scaler:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()

            if suffix:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_{suffix}.pt"
            else:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}_step_{self.global_step}.pt"

            torch.save(checkpoint, checkpoint_path)
            print(f"✅ 保存检查点: {checkpoint_path}")

            # 保存配置
            config_path = self.checkpoint_dir / "config.json"
            with open(config_path, 'w') as f:
                config_dict = self.config.__dict__.copy()
                config_dict.update(self.stage3_config)
                json.dump(config_dict, f, indent=2)

            if is_best:
                best_path = self.checkpoint_dir / "best_model.pt"
                torch.save(checkpoint, best_path)
                print(f"🎉 保存最佳模型: {best_path}")

                # 导出为部署格式
                self.export_model()

        except Exception as e:
            print(f"保存检查点失败: {e}")

    def export_model(self):
        """导出模型"""
        try:
            export_path = self.stage_dir / "stage3_model.pth"
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config.__dict__,
                'stage3_config': self.stage3_config,
                'phrase_stats': self.phrase_stats,
                'global_step': self.global_step,
                'best_val_loss': self.best_val_loss,
                'best_alignment_score': self.best_alignment_score
            }, export_path)
            print(f"✅ 模型导出到: {export_path}")
        except Exception as e:
            print(f"模型导出失败: {e}")

    def train(self):
        """主训练循环"""
        print(f"\n{'=' * 60}")
        print(f"开始第三阶段训练，共 {self.stage3_config['stage3_epochs']} 个epoch")
        print(f"{'=' * 60}")

        start_time = time.time()
        epoch_times = []

        try:
            for epoch in range(self.current_epoch, self.stage3_config['stage3_epochs']):
                self.current_epoch = epoch
                epoch_start_time = time.time()

                print(f"\n{'=' * 50}")
                print(f"Epoch {epoch + 1}/{self.stage3_config['stage3_epochs']}")
                print(f"{'=' * 50}")

                # 训练
                train_losses = self.train_epoch(epoch)

                if train_losses:
                    print(f"\nEpoch {epoch + 1} 训练完成:")
                    for key, value in train_losses.items():
                        print(f"  {key}: {value:.4f}")

                    # 记录训练历史
                    self.train_history.append({
                        'epoch': epoch,
                        'step': self.global_step,
                        **train_losses
                    })

                # 验证
                val_metrics = self.validate()

                # 记录验证历史
                self.val_history.append({
                    'epoch': epoch,
                    'step': self.global_step,
                    **val_metrics
                })

                # 保存最佳模型
                if val_metrics['alignment_score'] > self.best_alignment_score:
                    self.best_alignment_score = val_metrics['alignment_score']
                    self.best_val_loss = val_metrics['val_loss']
                    self.save_checkpoint(is_best=True, suffix="best")
                    print(f"🎉 新的最佳模型！对齐分数: {self.best_alignment_score:.4f}")

                # 保存定期检查点
                if (epoch + 1) % 5 == 0 or epoch == self.stage3_config['stage3_epochs'] - 1:
                    self.save_checkpoint(suffix=f"epoch_{epoch + 1}")

                # 计算epoch时间
                epoch_time = time.time() - epoch_start_time
                epoch_times.append(epoch_time)
                avg_epoch_time = np.mean(epoch_times) if epoch_times else epoch_time
                remaining_time = avg_epoch_time * (self.stage3_config['stage3_epochs'] - epoch - 1)

                print(f"\nEpoch {epoch + 1} 时间: {epoch_time:.1f}s")
                print(f"预计剩余时间: {remaining_time / 60:.1f}分钟")

                # 记录到wandb
                if self.use_wandb:
                    wandb.log({
                        'train/epoch_loss': train_losses.get('total', 0),
                        'train/epoch_contrastive_loss': train_losses.get('contrastive', 0),
                        'val/epoch_loss': val_metrics.get('val_loss', 0),
                        'val/alignment_score': val_metrics.get('alignment_score', 0),
                        'val/epoch': epoch,
                        'train/epoch': epoch
                    })

                print(f"{'=' * 50}")

                # 内存清理
                memory_monitor.clear_cache()

        except KeyboardInterrupt:
            print("\n\n训练被用户中断")
            self.save_checkpoint(suffix="interrupted")
        except Exception as e:
            print(f"\n\n训练出错: {e}")
            import traceback
            traceback.print_exc()
            self.save_checkpoint(suffix="error")
        finally:
            self.save_checkpoint(suffix="final")

        # 训练总结
        total_time = time.time() - start_time
        print(f"\n{'=' * 60}")
        print("第三阶段训练总结:")
        print(f"  总时间: {total_time / 60:.1f} 分钟")
        print(f"  总步数: {self.global_step}")
        print(f"  最佳对齐分数: {self.best_alignment_score:.4f}")
        print(f"  最佳验证损失: {self.best_val_loss:.4f}")

        # 短语统计
        print(f"\n短语统计:")
        print(f"  总短语数: {self.phrase_stats['total_phrases']}")
        print(f"  平均每样本短语数: {self.phrase_stats.get('avg_phrases_per_sample', 0):.2f}")
        for phrase_type, count in self.phrase_stats['phrase_types'].items():
            print(f"  {phrase_type}: {count}")

        print(f"{'=' * 60}")

        # 关闭wandb
        if self.use_wandb:
            wandb.finish()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='第三阶段训练：短语级对比学习')
    parser.add_argument('--stage2-checkpoint', type=str, required=True,
                        default="/home/common-dir/result/training_output/stage2_alignment/checkpoints/best_model.pt",
                        help='Stage2检查点路径')
    parser.add_argument('--resume', type=str, help='从检查点恢复训练')
    parser.add_argument('--no-wandb', action='store_true', default=False, help='禁用wandb')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=None, help='batch size')
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    parser.add_argument('--no-mixed-precision', action='store_true', default=False, help='禁用混合精度')
    parser.add_argument('--grad-accum-steps', type=int, default=None, help='梯度累积步数')
    parser.add_argument('--num_workers', type=int, default=None, help='工作台数量')

    args = parser.parse_args()

    # 创建训练器
    trainer = Stage3PhraseTrainer(args.stage2_checkpoint, use_wandb=not args.no_wandb)

    # 覆盖训练器配置（如果提供了命令行参数）
    if args.epochs:
        trainer.stage3_config['stage3_epochs'] = args.epochs
    if args.batch_size:
        trainer.config.batch_size = args.batch_size
    if args.lr:
        trainer.stage3_config['learning_rate_stage3'] = args.lr
    if args.grad_accum_steps:
        trainer.stage3_config['grad_accum_steps_stage3'] = args.grad_accum_steps
    if args.no_mixed_precision:
        trainer.config.mixed_precision = False

    print(f"\n最终第三阶段配置:")
    print(f"  训练轮数: {trainer.stage3_config['stage3_epochs']}")
    print(f"  批次大小: {trainer.config.batch_size}")
    print(f"  学习率: {trainer.stage3_config['learning_rate_stage3']}")
    print(f"  梯度累积: {trainer.stage3_config['grad_accum_steps_stage3']}")
    print(f"  混合精度: {trainer.config.mixed_precision}")

    # 处理恢复训练
    if args.resume and Path(args.resume).exists():
        print(f"\n从检查点恢复训练: {args.resume}")
        try:
            checkpoint = torch.load(args.resume, map_location='cpu')
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            trainer.current_epoch = checkpoint.get('epoch', 0)
            trainer.global_step = checkpoint.get('global_step', 0)
            trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            trainer.best_alignment_score = checkpoint.get('best_alignment_score', 0.0)
            trainer.train_history = checkpoint.get('train_history', [])
            trainer.val_history = checkpoint.get('val_history', [])
            trainer.phrase_stats = checkpoint.get('phrase_stats', {})

            if trainer.scaler and 'scaler_state_dict' in checkpoint:
                trainer.scaler.load_state_dict(checkpoint['scaler_state_dict'])

            print(f"恢复成功: epoch={trainer.current_epoch}, step={trainer.global_step}")
        except Exception as e:
            print(f"恢复失败: {e}")

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()


