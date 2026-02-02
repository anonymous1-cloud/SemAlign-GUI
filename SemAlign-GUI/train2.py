#!/usr/bin/env python3
"""
第二阶段训练：组件感知的视觉-文本对齐模型 - 正式版本
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import numpy as np
import time
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from dataloder import create_data_loader
from model2 import Stage2AlignmentModel
from memory import memory_monitor


class Stage2AlignmentTrainer:
    """第二阶段对齐训练器"""

    def __init__(self, stage1_checkpoint: str, use_wandb: bool = True):
        self.config = config
        self.device = config.device
        self.use_wandb = use_wandb

        print(f"\n{'=' * 60}")
        print("第二阶段训练：视觉-文本-组件对齐模型")
        print(f"{'=' * 60}")
        print(f"设备: {self.device}")
        print(f"隐藏维度: {self.config.hidden_dim}")
        print(f"批次大小: {self.config.batch_size}")

        # 检查检查点
        if not Path(stage1_checkpoint).exists():
            print(f"警告：找不到stage1检查点 {stage1_checkpoint}")
            print("将使用随机初始化的视觉编码器")

        self.stage1_checkpoint = stage1_checkpoint

        # ============ 初始化模型 ============
        print("\n初始化组件感知对齐模型...")
        memory_monitor.print_memory_stats("初始化前")

        self.model = Stage2AlignmentModel(
            stage1_checkpoint=self.stage1_checkpoint,
            config=self.config,
            use_components=True
        ).to(self.device)

        # 打印参数统计
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"总参数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        print(f"冻结比例: {(total_params - trainable_params) / total_params * 100:.1f}%")

        # ============ 优化器配置 ============
        # 分层学习率
        text_params = []
        component_params = []
        fusion_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if 'text_encoder' in name:
                text_params.append(param)
            elif 'component_encoder' in name:
                component_params.append(param)
            elif 'fusion_module' in name or 'alignment_head' in name or 'contrastive_proj' in name:
                fusion_params.append(param)
            else:
                other_params.append(param)

        print(f"\n优化器参数分组:")
        print(f"  文本参数: {len(text_params)}层")
        print(f"  组件参数: {len(component_params)}层")
        print(f"  融合参数: {len(fusion_params)}层")
        print(f"  其他参数: {len(other_params)}层")

        # 优化器配置
        self.optimizer = optim.AdamW([
            {'params': text_params, 'lr': 1e-4, 'weight_decay': 0.01},
            {'params': component_params, 'lr': 5e-4, 'weight_decay': 0.001},
            {'params': fusion_params, 'lr': 3e-4, 'weight_decay': 0.001},
            {'params': other_params, 'lr': 3e-4, 'weight_decay': 0.01}
        ])

        # ============ 损失函数权重 ============
        self.base_loss_weights = {
            'alignment': 0.6,
            'visual_text': 0.8,
            'contrastive': 0.2,
            'comp_visual': 0.6,
            'comp_text': 0.4,
            'change_type': 0.2,
            'gate_entropy': 0.05
        }

        self.loss_weights = self.base_loss_weights.copy()

        print(f"\n初始损失权重配置:")
        for key, weight in self.loss_weights.items():
            print(f"  {key}: {weight}")

        # ============ 数据加载器 ============
        self.train_loader = create_data_loader('train', self.config, is_stage1=False)
        self.val_loader = create_data_loader('val', self.config, is_stage1=False)

        total_steps = self.config.stage2_epochs * len(self.train_loader)

        # ============ 学习率调度器 ============
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[1e-4, 5e-4, 3e-4, 3e-4],
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )

        print(f"\n数据集统计:")
        print(f"  训练集: {len(self.train_loader.dataset)} 样本")
        print(f"  验证集: {len(self.val_loader.dataset)} 样本")
        print(f"  总训练步数: {total_steps}")

        # ============ 损失函数 ============
        self.alignment_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        self.contrastive_loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.change_type_loss_fn = nn.CrossEntropyLoss(reduction='mean')

        # ============ 训练监控 ============
        self.loss_history = {}
        self.similarity_history = {
            'visual_text': [],
            'comp_visual': [],
            'comp_text': []
        }
        self.gradient_history = {
            'text': [],
            'component': [],
            'fusion': []
        }

        # ============ 输出目录 ============
        self.stage_dir = self.config.output_dir / "stage2_alignment"
        self.checkpoint_dir = self.stage_dir / "checkpoints"
        self.log_dir = self.stage_dir / "logs"

        # 创建目录
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # ============ 训练状态 ============
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_alignment_score = 0.0
        self.patience_counter = 0
        self.max_patience = 8

        memory_monitor.print_memory_stats("初始化后")

    def compute_alignment_loss(self, outputs, batch):
        """计算对齐损失"""
        losses = {}

        try:
            batch_size = outputs['visual_features'].shape[0]

            # 提取特征并归一化
            visual_features = F.normalize(outputs['visual_features'], dim=1)
            text_features = F.normalize(outputs['text_features'], dim=1)

            outputs['visual_features'] = visual_features
            outputs['text_features'] = text_features

            # 1. 对齐预测损失
            if 'alignment_logits' in outputs:
                alignment_logits = outputs['alignment_logits']
            else:
                alignment_scores = outputs['alignment_scores']
                alignment_logits = torch.log(alignment_scores / (1 - alignment_scores + 1e-8))

            if 'has_change' in batch:
                alignment_labels = batch['has_change'].unsqueeze(1).float()
                alignment_labels = alignment_labels.to(alignment_logits.device)
                alignment_loss = self.alignment_loss_fn(alignment_logits, alignment_labels)
                losses['alignment'] = alignment_loss * self.loss_weights['alignment']
            else:
                losses['alignment'] = torch.tensor(0.0, device=visual_features.device)

            # 2. 视觉-文本对齐损失
            similarity = F.cosine_similarity(visual_features, text_features, dim=1)
            v_t_similarity = similarity.mean()
            v_t_loss = 1.0 - v_t_similarity
            losses['visual_text'] = v_t_loss * self.loss_weights['visual_text']

            # 记录相似度
            if self.global_step % 50 == 0:
                self.similarity_history['visual_text'].append(v_t_similarity.item())

            # 3. 对比学习损失
            contrastive_features = F.normalize(outputs['contrastive_features'], dim=-1)
            sim_matrix = torch.matmul(contrastive_features, contrastive_features.T)

            temperature = outputs.get('temperature', torch.tensor(0.07, device=contrastive_features.device))
            sim_matrix = sim_matrix / temperature

            labels = torch.arange(batch_size, device=contrastive_features.device)
            contrastive_loss = self.contrastive_loss_fn(sim_matrix, labels)
            losses['contrastive'] = contrastive_loss * self.loss_weights['contrastive']

            # 4. 组件相关损失
            if 'component_outputs' in outputs:
                component_features = F.normalize(outputs['component_outputs']['change_features'], dim=1)

                # 组件-视觉对齐
                comp_vis_sim = F.cosine_similarity(component_features, visual_features, dim=1)
                comp_vis_similarity = comp_vis_sim.mean()
                losses['comp_visual'] = (1.0 - comp_vis_similarity) * self.loss_weights['comp_visual']

                if self.global_step % 50 == 0:
                    self.similarity_history['comp_visual'].append(comp_vis_similarity.item())

                # 组件-文本对齐
                comp_text_sim = F.cosine_similarity(component_features, text_features, dim=1)
                comp_text_similarity = comp_text_sim.mean()
                losses['comp_text'] = (1.0 - comp_text_similarity) * self.loss_weights['comp_text']

                if self.global_step % 50 == 0:
                    self.similarity_history['comp_text'].append(comp_text_similarity.item())

                # 变化类型分类
                if 'change_type_logits' in outputs['component_outputs']:
                    change_type_logits = outputs['component_outputs']['change_type_logits']
                    if 'change_type' in batch:
                        change_type = batch['change_type']
                        change_type = change_type.to(change_type_logits.device)
                        if change_type.dim() > 1 and change_type.shape[-1] > 1:
                            change_type = change_type.argmax(dim=1)
                        change_type_loss = self.change_type_loss_fn(change_type_logits, change_type)
                        losses['change_type'] = change_type_loss * self.loss_weights['change_type']
                    else:
                        losses['change_type'] = torch.tensor(0.0, device=change_type_logits.device)

            # 5. 多模态一致性损失
            if 'fusion_outputs' in outputs:
                fusion_out = outputs['fusion_outputs']
                if 'gate_values' in fusion_out:
                    gate_values = fusion_out['gate_values']
                    entropy_loss = -torch.sum(gate_values * torch.log(gate_values + 1e-8), dim=1).mean()
                    losses['gate_entropy'] = -0.1 * entropy_loss * self.loss_weights['gate_entropy']

            # 总损失
            total_loss = sum(losses.values())
            losses['total'] = total_loss

            # 记录损失历史
            self.record_loss_history(losses)

            # 动态调整损失权重
            if self.global_step % 100 == 0:
                self.dynamically_adjust_weights()

        except Exception as e:
            print(f"损失计算错误: {e}")
            device = outputs.get('visual_features', torch.zeros(1).to(self.device)).device
            total_loss = torch.tensor(1.0, device=device, requires_grad=True)
            losses = {'total': total_loss}

        return total_loss, losses

    def record_loss_history(self, losses):
        """记录损失历史"""
        for key, value in losses.items():
            if key not in self.loss_history:
                self.loss_history[key] = []

            if torch.is_tensor(value):
                self.loss_history[key].append(value.item())
            else:
                self.loss_history[key].append(value)

    def dynamically_adjust_weights(self):
        """动态调整损失权重"""
        if self.global_step == 0:
            return

        weights_changed = False
        for sim_type in ['visual_text', 'comp_visual', 'comp_text']:
            if len(self.similarity_history[sim_type]) >= 5:
                recent_sim = np.mean(self.similarity_history[sim_type][-5:])

                # 调整权重
                if recent_sim < 0.3:
                    if sim_type == 'visual_text':
                        new_weight = min(1.0, self.base_loss_weights['visual_text'] * 1.1)
                        if abs(new_weight - self.loss_weights['visual_text']) > 0.01:
                            self.loss_weights['visual_text'] = new_weight
                            weights_changed = True
                    elif sim_type == 'comp_visual':
                        new_weight = min(1.0, self.base_loss_weights['comp_visual'] * 1.1)
                        if abs(new_weight - self.loss_weights['comp_visual']) > 0.01:
                            self.loss_weights['comp_visual'] = new_weight
                            weights_changed = True

                elif recent_sim > 0.7:
                    if sim_type == 'visual_text':
                        new_weight = max(0.1, self.base_loss_weights['visual_text'] * 0.9)
                        if abs(new_weight - self.loss_weights['visual_text']) > 0.01:
                            self.loss_weights['visual_text'] = new_weight
                            weights_changed = True
                    elif sim_type == 'comp_visual':
                        new_weight = max(0.1, self.base_loss_weights['comp_visual'] * 0.9)
                        if abs(new_weight - self.loss_weights['comp_visual']) > 0.01:
                            self.loss_weights['comp_visual'] = new_weight
                            weights_changed = True

        if weights_changed and self.global_step % 500 == 0:
            print(f"调整损失权重 (Step {self.global_step}):")
            for key in ['visual_text', 'comp_visual', 'comp_text']:
                print(f"  {key}: {self.loss_weights[key]:.3f}")

    def monitor_gradients(self):
        """监控梯度"""
        grad_stats = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None and param.requires_grad:
                grad_norm = param.grad.norm().item()
                if 'text_encoder' in name:
                    grad_stats.setdefault('text', []).append(grad_norm)
                elif 'component_encoder' in name:
                    grad_stats.setdefault('component', []).append(grad_norm)
                elif 'fusion' in name or 'alignment' in name or 'contrastive' in name:
                    grad_stats.setdefault('fusion', []).append(grad_norm)

        # 记录梯度历史
        for module, norms in grad_stats.items():
            if norms:
                mean_grad = np.mean(norms)
                self.gradient_history[module].append(mean_grad)

        # 定期打印梯度统计
        if self.global_step % 200 == 0 and self.global_step > 0:
            print(f"\n梯度统计 (Step {self.global_step}):")
            for module in ['text', 'component', 'fusion']:
                if self.gradient_history[module]:
                    recent = self.gradient_history[module][-20:] if len(self.gradient_history[module]) >= 20 else \
                        self.gradient_history[module]
                    if recent:
                        avg_grad = np.mean(recent)
                        print(f"  {module}: {avg_grad:.6f}")

    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model.train()
        epoch_losses = {}

        # 预热策略
        if epoch < 2:
            self.loss_weights['visual_text'] = 0.5
            self.loss_weights['comp_visual'] = 0.5
            if epoch == 0:
                print(f"预热阶段 {epoch + 1}/2")
        elif epoch == 2:
            self.loss_weights = self.base_loss_weights.copy()
            print(f"预热结束，使用完整损失权重")

        train_loader = self.train_loader
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config.stage2_epochs}")

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

                # 确保组件数据有效
                if ref_components is not None:
                    ref_components[:, :, 0] = torch.clamp(ref_components[:, :, 0].long(), 0, 19)
                if tar_components is not None:
                    tar_components[:, :, 0] = torch.clamp(tar_components[:, :, 0].long(), 0, 19)

                # 清零梯度
                self.optimizer.zero_grad(set_to_none=True)

                # 前向传播
                outputs = self.model(ref_image, tar_image, text_tokens,
                                     ref_components, tar_components)

                total_loss, loss_dict = self.compute_alignment_loss(outputs, batch)

                # 检查损失
                if not total_loss.requires_grad:
                    continue

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    continue

                # 反向传播
                total_loss.backward()

                # 梯度累积
                if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                    # 梯度监控
                    if self.global_step % 100 == 0:
                        self.monitor_gradients()

                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    # 优化器更新
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                    self.global_step += 1

                # 累积损失
                for key, value in loss_dict.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = []
                    epoch_losses[key].append(value.item())

                # 更新进度条
                pbar.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'v_t': f"{loss_dict.get('visual_text', 0):.4f}",
                    'c_v': f"{loss_dict.get('comp_visual', 0):.4f}",
                    'step': self.global_step
                })

                # 定期清理缓存
                if batch_idx % 50 == 0:
                    memory_monitor.clear_cache()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            except Exception as e:
                if batch_idx == 0:
                    print(f"批次 {batch_idx} 错误: {e}")
                self.optimizer.zero_grad(set_to_none=True)
                continue

        # 计算平均损失
        avg_losses = {}
        for key, values in epoch_losses.items():
            if values:
                avg_losses[key] = np.mean(values)
            else:
                avg_losses[key] = 0.0

        return avg_losses

    @torch.no_grad()
    def validate(self):
        """验证过程"""
        self.model.eval()
        val_losses = []
        all_metrics = []
        similarity_stats = {
            'visual_text': [],
            'comp_visual': [],
            'comp_text': []
        }

        print(f"\n验证模型 (Step {self.global_step})...")

        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation")):
            if batch is None:
                continue

            try:
                ref_image = batch['ref_image'].to(self.device, non_blocking=True)
                tar_image = batch['tar_image'].to(self.device, non_blocking=True)
                text_tokens = batch['text_tokens'].to(self.device, non_blocking=True)
                ref_components = batch['ref_components'].to(self.device, non_blocking=True)
                tar_components = batch['tar_components'].to(self.device, non_blocking=True)

                if ref_components is not None:
                    ref_components[:, :, 0] = torch.clamp(ref_components[:, :, 0].long(), 0, 19)
                if tar_components is not None:
                    tar_components[:, :, 0] = torch.clamp(tar_components[:, :, 0].long(), 0, 19)

                outputs = self.model(ref_image, tar_image, text_tokens, ref_components, tar_components)

                total_loss, loss_dict = self.compute_alignment_loss(outputs, batch)
                val_losses.append(total_loss.item())

                # 收集相似度统计
                visual_features = outputs['visual_features']
                text_features = outputs['text_features']

                v_t_sim = F.cosine_similarity(visual_features, text_features, dim=1).mean().item()
                similarity_stats['visual_text'].append(v_t_sim)

                if 'component_outputs' in outputs:
                    component_features = outputs['component_outputs']['change_features']

                    c_v_sim = F.cosine_similarity(component_features, visual_features, dim=1).mean().item()
                    similarity_stats['comp_visual'].append(c_v_sim)

                    c_t_sim = F.cosine_similarity(component_features, text_features, dim=1).mean().item()
                    similarity_stats['comp_text'].append(c_t_sim)

                # 收集指标
                metrics = {'val_loss': total_loss.item()}
                for k, v in loss_dict.items():
                    if torch.is_tensor(v):
                        metrics[f'val_{k}'] = v.item()
                all_metrics.append(metrics)

            except Exception as e:
                continue

        if not val_losses:
            return {'val_loss': float('inf'), 'alignment_score': 0.0}

        # 计算平均指标
        avg_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics if key in m]
                if values:
                    avg_metrics[key] = np.mean(values)

        avg_val_loss = np.mean(val_losses)

        # 计算对齐分数
        alignment_similarity = 0.0
        sim_counts = 0
        for sim_type, values in similarity_stats.items():
            if values:
                alignment_similarity += np.mean(values)
                sim_counts += 1

        if sim_counts > 0:
            alignment_similarity /= sim_counts
            alignment_score = alignment_similarity
        else:
            alignment_score = 1.0 / (avg_val_loss + 1e-8)

        avg_metrics['val_loss'] = avg_val_loss
        avg_metrics['alignment_score'] = alignment_score

        # 添加相似度统计
        for sim_type, values in similarity_stats.items():
            if values:
                avg_metrics[f'{sim_type}_similarity'] = np.mean(values)

        # 打印验证结果
        print(f"\n验证结果 (Step {self.global_step}):")
        print(f"  验证损失: {avg_val_loss:.4f}")
        print(f"  对齐分数: {alignment_score:.4f}")

        if 'visual_text_similarity' in avg_metrics:
            print(f"\n  相似度:")
            print(f"    视觉-文本: {avg_metrics['visual_text_similarity']:.4f}")
            if 'comp_visual_similarity' in avg_metrics:
                print(f"    组件-视觉: {avg_metrics['comp_visual_similarity']:.4f}")
            if 'comp_text_similarity' in avg_metrics:
                print(f"    组件-文本: {avg_metrics['comp_text_similarity']:.4f}")

        print(f"\n  损失分解:")
        print(f"    对齐预测: {avg_metrics.get('val_alignment', 0):.4f}")
        print(f"    视觉-文本: {avg_metrics.get('val_visual_text', 0):.4f}")
        print(f"    对比学习: {avg_metrics.get('val_contrastive', 0):.4f}")

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
                'loss_weights': self.loss_weights,
                'config': self.config.__dict__
            }

            if suffix:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_{suffix}.pt"
            else:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}_step_{self.global_step}.pt"

            torch.save(checkpoint, checkpoint_path)

            if is_best:
                best_path = self.checkpoint_dir / "best_model.pt"
                torch.save(checkpoint, best_path)
                print(f"🎉 保存最佳模型: {best_path}")

        except Exception as e:
            print(f"保存检查点失败: {e}")

    def train(self):
        """主训练循环"""
        print(f"\n{'=' * 60}")
        print(f"开始第二阶段对齐训练，共 {self.config.stage2_epochs} 个epoch")
        print(f"{'=' * 60}")

        start_time = time.time()

        try:
            for epoch in range(self.current_epoch, self.config.stage2_epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()

                print(f"\n{'=' * 50}")
                print(f"Epoch {epoch + 1}/{self.config.stage2_epochs}")
                print(f"{'=' * 50}")

                # 训练
                train_losses = self.train_epoch(epoch)

                if train_losses:
                    print(f"\n训练完成:")
                    print(f"  总损失: {train_losses.get('total', 0):.4f}")
                    for key in ['visual_text', 'comp_visual', 'comp_text']:
                        if key in train_losses:
                            print(f"  {key}: {train_losses[key]:.4f}")

                # 验证
                val_metrics = self.validate()

                # 早停检查
                if val_metrics['alignment_score'] > self.best_alignment_score:
                    self.best_alignment_score = val_metrics['alignment_score']
                    self.best_val_loss = val_metrics['val_loss']
                    self.patience_counter = 0
                    self.save_checkpoint(is_best=True, suffix="best")
                    print(f"🎉 新的最佳模型！对齐分数: {self.best_alignment_score:.4f}")
                else:
                    self.patience_counter += 1

                    if self.patience_counter >= self.max_patience // 2:
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] *= 0.5

                # 早停
                if self.patience_counter >= self.max_patience:
                    print(f"\n⚠️ 早停触发")
                    break

                # 保存检查点
                if (epoch + 1) % 5 == 0:
                    self.save_checkpoint(suffix=f"epoch_{epoch + 1}")

                # 计算时间
                epoch_time = time.time() - epoch_start_time
                remaining_epochs = self.config.stage2_epochs - epoch - 1
                remaining_time = epoch_time * remaining_epochs

                print(f"\nEpoch {epoch + 1} 时间: {epoch_time:.1f}s")
                print(f"预计剩余时间: {remaining_time / 60:.1f}分钟")
                print(f"{'=' * 50}")

        except KeyboardInterrupt:
            print("\n训练被中断")
            self.save_checkpoint(suffix="interrupted")
        except Exception as e:
            print(f"\n训练出错: {e}")
            self.save_checkpoint(suffix="error")
        finally:
            self.save_checkpoint(suffix="final")

        # 训练总结
        total_time = time.time() - start_time
        print(f"\n{'=' * 60}")
        print("训练总结:")
        print(f"  总时间: {total_time / 60:.1f} 分钟")
        print(f"  总步数: {self.global_step}")
        print(f"  最佳对齐分数: {self.best_alignment_score:.4f}")
        print(f"{'=' * 60}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='第二阶段训练：视觉-文本-组件对齐')
    parser.add_argument('--stage1-checkpoint', type=str,
                        default=str(config.output_dir / "stage1" / "checkpoints" / "best_model.pt"),
                        help='Stage1检查点路径')
    parser.add_argument('--resume', type=str, help='从检查点恢复训练')
    parser.add_argument('--no-wandb', action='store_true', default=True, help='禁用wandb')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=None, help='batch size')

    args = parser.parse_args()

    # 更新配置
    if args.epochs:
        config.stage2_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size

    # 创建训练器
    trainer = Stage2AlignmentTrainer(args.stage1_checkpoint, use_wandb=not args.no_wandb)

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()