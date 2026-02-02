#!/usr/bin/env python3
"""
第一阶段训练：视觉基础模型
适配新的模型输出
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import wandb
import os
import sys
from pathlib import Path
import numpy as np
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from dataloder import create_data_loader
from models import Stage1VisualModel
from metrics import compute_change_detection_metrics, print_metrics
from memory import memory_monitor


class Stage1Trainer:
    """第一阶段训练器：视觉基础"""

    def __init__(self, use_wandb: bool = True):
        self.config = config
        self.device = config.device
        self.use_wandb = use_wandb

        print(f"\n{'=' * 60}")
        print("第一阶段训练：视觉基础模型")
        print(f"{'=' * 60}")
        print(f"设备: {self.device}")
        print(f"CUDA可用: {self.config.cuda_available}")

        if self.config.cuda_available:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

        # 检查数据路径
        data_root = Path(self.config.data_root)
        required_files = ['train.json', 'val.json', 'images.h5', 'masks.h5']
        missing_files = [f for f in required_files if not (data_root / f).exists()]

        if missing_files:
            print(f"错误：缺少必要的文件: {missing_files}")
            sys.exit(1)

        # 根据GPU内存自动调整batch size
        if self.config.cuda_available:
            self.config.auto_adjust_batch_size()

        print(f"配置参数:")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  梯度累积步数: {self.config.grad_accum_steps}")
        print(f"  有效Batch size: {self.config.batch_size * self.config.grad_accum_steps}")
        print(f"  混合精度: {self.config.mixed_precision}")
        print(f"  梯度检查点: {self.config.gradient_checkpointing}")

        # 初始化模型
        print("\n初始化模型...")
        memory_monitor.print_memory_stats("初始化前")

        self.model = Stage1VisualModel(self.config).to(self.device)

        # 打印模型参数
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"总参数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        print(f"模型大小: {total_params * 4 / 1024 ** 2:.1f} MB (float32)")

        # 混合精度训练
        self.scaler = GradScaler() if self.config.mixed_precision and self.config.cuda_available else None

        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=self.config.max_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )

        # 损失函数 - 使用BCEWithLogitsLoss（mask_logits输出）
        self.mask_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        self.change_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')

        # 数据加载器
        print("\n加载数据集...")
        self.train_loader = create_data_loader('train', self.config, is_stage1=True)
        self.val_loader = create_data_loader('val', self.config, is_stage1=True)

        if len(self.train_loader) == 0 or len(self.val_loader) == 0:
            print("错误：数据加载器为空！")
            sys.exit(1)

        print(f"训练集: {len(self.train_loader.dataset)} 样本")
        print(f"验证集: {len(self.val_loader.dataset)} 样本")

        # 输出目录
        self.stage_dir = self.config.output_dir / "stage1"
        self.checkpoint_dir = self.stage_dir / "checkpoints"
        self.log_dir = self.stage_dir / "logs"

        # 训练状态
        self.global_step = 1000
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0

        # 训练历史
        self.train_history = []
        self.val_history = []

        # 初始化wandb
        if self.use_wandb:
            try:
                wandb.init(
                    project="gui-change-detection",
                    name=f"stage1-visual-{time.strftime('%Y%m%d-%H%M%S')}",
                    config=self.config.__dict__,
                    dir=str(self.log_dir)
                )
                wandb.watch(self.model, log='all', log_freq=100)
                print("Wandb初始化成功")
            except Exception as e:
                print(f"Wandb初始化失败: {e}")
                self.use_wandb = False

        memory_monitor.print_memory_stats("初始化后")
        print(f"\n准备开始训练...")

    def compute_loss(self, outputs, mask):
        """计算损失函数"""
        # 获取模型输出
        pred_logits = outputs['pred_logits']  # [B, H, W] mask logits
        change_logits = outputs['change_logits']  # [B, 1] 变化分类logits

        # Mask损失
        mask_loss = self.mask_loss_fn(pred_logits, mask)

        # 变化分类损失（基于mask是否有变化）
        has_change = (mask.sum(dim=(1, 2)) > 10).float().unsqueeze(1)
        change_loss = self.change_loss_fn(change_logits, has_change)

        # 总损失
        total_loss = mask_loss + 0.3 * change_loss

        return total_loss, mask_loss, change_loss

    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model.train()
        epoch_total_loss = 0.0
        epoch_mask_loss = 0.0
        epoch_change_loss = 0.0

        # 进度条
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.stage1_epochs}")

        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue

            # 移动到设备
            ref_image = batch['ref_image'].to(self.device, non_blocking=True)
            tar_image = batch['tar_image'].to(self.device, non_blocking=True)
            mask = batch['mask'].to(self.device, non_blocking=True)

            # 前向传播（混合精度）
            with autocast(enabled=self.scaler is not None):
                outputs = self.model(ref_image, tar_image)

                # 计算损失
                total_loss, mask_loss, change_loss = self.compute_loss(outputs, mask)

            # 反向传播（梯度累积）
            loss = total_loss / self.config.grad_accum_steps

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # 梯度累积步骤
            if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                # 梯度裁剪
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # 优化器步骤
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.global_step += 1

            # 累积损失
            epoch_total_loss += total_loss.item()
            epoch_mask_loss += mask_loss.item()
            epoch_change_loss += change_loss.item()

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'mask': f"{mask_loss.item():.4f}",
                'change': f"{change_loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })

            # 定期验证和保存
            if self.global_step % 100 == 0:
                if self.global_step % 500 == 0:
                    val_metrics = self.validate()
                    self.save_checkpoint()

                # 记录到wandb
                if self.use_wandb and self.global_step % 10 == 0:
                    wandb.log({
                        'train/step_loss': total_loss.item(),
                        'train/step_mask_loss': mask_loss.item(),
                        'train/step_change_loss': change_loss.item(),
                        'train/learning_rate': self.scheduler.get_last_lr()[0],
                        'train/step': self.global_step
                    })

            # 定期清理缓存
            if batch_idx % 50 == 0:
                memory_monitor.clear_cache()

        # 计算平均损失
        avg_total_loss = epoch_total_loss / len(self.train_loader)
        avg_mask_loss = epoch_mask_loss / len(self.train_loader)
        avg_change_loss = epoch_change_loss / len(self.train_loader)

        return avg_total_loss, avg_mask_loss, avg_change_loss

    @torch.no_grad()
    def validate(self):
        """验证"""
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        print(f"\n验证模型 (Step {self.global_step})...")

        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation")):
            if batch is None:
                continue

            ref_image = batch['ref_image'].to(self.device, non_blocking=True)
            tar_image = batch['tar_image'].to(self.device, non_blocking=True)
            mask = batch['mask'].to(self.device, non_blocking=True)

            outputs = self.model(ref_image, tar_image)
            pred_logits = outputs['pred_logits']

            # 计算损失
            total_loss, mask_loss, change_loss = self.compute_loss(outputs, mask)
            val_loss += total_loss.item()

            # 将logits转换为概率用于指标计算
            pred_mask = torch.sigmoid(pred_logits).cpu().numpy()
            all_preds.append(pred_mask)
            all_targets.append(mask.cpu().numpy())

        if len(all_preds) == 0:
            print("警告：验证集为空")
            return {}

        # 计算指标
        val_loss /= len(self.val_loader)
        all_preds_np = np.concatenate(all_preds, axis=0)
        all_targets_np = np.concatenate(all_targets, axis=0)

        metrics = compute_change_detection_metrics(all_preds_np, all_targets_np)
        metrics['val_loss'] = val_loss

        # 打印指标
        print_metrics(metrics, "验证集")

        # 记录到wandb
        if self.use_wandb:
            wandb.log({
                'val/loss': val_loss,
                'val/f1_score': metrics['f1_score'],
                'val/iou': metrics['iou'],
                'val/precision': metrics['precision'],
                'val/recall': metrics['recall'],
                'val/accuracy': metrics['accuracy'],
                'val/step': self.global_step
            })

        # 保存最佳模型
        if metrics['f1_score'] > self.best_val_f1:
            self.best_val_f1 = metrics['f1_score']
            self.best_val_loss = val_loss
            self.save_checkpoint(is_best=True)
            print(f"✅ 新的最佳模型！F1: {self.best_val_f1:.4f}, Loss: {self.best_val_loss:.4f}")

        self.model.train()
        return metrics

    def save_checkpoint(self, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_f1': self.best_val_f1,
            'config': self.config.__dict__,
            'train_history': self.train_history,
            'val_history': self.val_history
        }

        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # 定期保存
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{self.global_step}.pt"
        torch.save(checkpoint, checkpoint_path)

        # 只保留最近5个检查点
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
        if len(checkpoints) > 5:
            for old_checkpoint in checkpoints[:-5]:
                old_checkpoint.unlink()

        # 保存最佳模型
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)

            # 导出为可部署格式
            self.export_model()

    def export_model(self):
        """导出第一阶段模型"""
        export_path = self.stage_dir / "stage1_model.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
            'global_step': self.global_step,
            'best_val_f1': self.best_val_f1
        }, export_path)
        print(f"模型导出到 {export_path}")

    def train(self):
        """主训练循环"""
        print(f"\n{'=' * 60}")
        print(f"开始训练，共 {self.config.stage1_epochs} 个epoch")
        print(f"{'=' * 60}")

        start_time = time.time()

        for epoch in range(self.config.stage1_epochs):
            self.current_epoch = epoch

            print(f"\nEpoch {epoch + 1}/{self.config.stage1_epochs}")
            memory_monitor.print_memory_stats(f"Epoch {epoch + 1} 开始前")

            # 训练一个epoch
            train_loss, train_mask_loss, train_change_loss = self.train_epoch(epoch)

            # 记录训练历史
            self.train_history.append({
                'epoch': epoch,
                'step': self.global_step,
                'loss': train_loss,
                'mask_loss': train_mask_loss,
                'change_loss': train_change_loss
            })

            print(f"Epoch {epoch + 1} 训练完成:")
            print(f"  总损失: {train_loss:.4f}")
            print(f"  Mask损失: {train_mask_loss:.4f}")
            print(f"  变化损失: {train_change_loss:.4f}")

            # 验证
            val_metrics = self.validate()

            # 记录验证历史
            self.val_history.append({
                'epoch': epoch,
                'step': self.global_step,
                **val_metrics
            })

            # 记录到wandb
            if self.use_wandb:
                wandb.log({
                    'train/epoch_loss': train_loss,
                    'train/epoch_mask_loss': train_mask_loss,
                    'train/epoch_change_loss': train_change_loss,
                    'train/epoch': epoch,
                    'val/epoch_loss': val_metrics.get('val_loss', 0),
                    'val/epoch_f1': val_metrics.get('f1_score', 0),
                    'val/epoch': epoch
                })

            # 保存检查点
            self.save_checkpoint()

            memory_monitor.print_memory_stats(f"Epoch {epoch + 1} 结束后")

            # 提前停止检查
            if self.global_step >= self.config.max_steps:
                print(f"达到最大步数 {self.config.max_steps}，停止训练")
                break

        # 最终验证
        print(f"\n{'=' * 60}")
        print("训练完成，最终验证...")
        final_metrics = self.validate()

        # 保存最终模型
        self.save_checkpoint(is_best=True)

        # 训练总结
        total_time = time.time() - start_time
        print(f"\n{'=' * 60}")
        print("训练总结:")
        print(f"  总时间: {total_time:.1f} 秒")
        print(f"  总步数: {self.global_step}")
        print(f"  最佳验证F1: {self.best_val_f1:.4f}")
        print(f"  最佳验证损失: {self.best_val_loss:.4f}")
        print(f"  峰值GPU内存: {memory_monitor.record_peak_memory():.1f} GB")
        print(f"{'=' * 60}")

        # 关闭wandb
        if self.use_wandb:
            wandb.finish()

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        if not Path(checkpoint_path).exists():
            print(f"检查点不存在: {checkpoint_path}")
            return False

        print(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 加载模型状态
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # 加载优化器状态
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 加载调度器状态
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # 加载训练状态
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_f1 = checkpoint['best_val_f1']

        # 加载混合精度scaler
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # 加载历史
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])

        print(f"检查点加载成功，从epoch {self.current_epoch}继续训练")
        return True


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='第一阶段训练：视觉基础模型')
    parser.add_argument('--resume', type=str, help='从检查点恢复训练')
    parser.add_argument('--no-wandb', action='store_true', help='禁用wandb')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')

    args = parser.parse_args()

    # 更新配置
    if args.epochs:
        config.stage1_epochs = args.epochs

    # 创建训练器
    trainer = Stage1Trainer(use_wandb=not args.no_wandb)

    # 恢复训练
    if args.resume:
        if not trainer.load_checkpoint(args.resume):
            print("无法加载检查点，开始新训练")

    # 开始训练
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n训练被中断")
        trainer.save_checkpoint()
    except Exception as e:
        print(f"\n训练出错: {e}")
        import traceback
        traceback.print_exc()
        trainer.save_checkpoint()


if __name__ == "__main__":
    main()