"""
评估指标计算
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple
import json


def compute_component_alignment_metrics(visual_features, text_features, components_list, config):
    """
    计算组件级别的视觉-文本对齐指标

    Args:
        visual_features: [B, D] 视觉特征
        text_features: [B, D] 文本特征
        components_list: List[Dict] 组件信息列表
        config: 配置对象
    """
    batch_size = visual_features.shape[0]

    # 归一化特征
    visual_norm = F.normalize(visual_features, dim=-1)
    text_norm = F.normalize(text_features, dim=-1)

    # 基础相似度矩阵
    similarity_matrix = torch.matmul(visual_norm, text_norm.T)  # [B, B]

    # 1. 整体检索指标
    v2t_pred = similarity_matrix.argmax(dim=1)
    v2t_labels = torch.arange(batch_size, device=visual_features.device)
    v2t_accuracy = (v2t_pred == v2t_labels).float().mean().item()

    t2v_pred = similarity_matrix.argmax(dim=0)
    t2v_accuracy = (t2v_pred == v2t_labels).float().mean().item()

    # 2. 组件感知对齐指标（新增）
    component_alignment_scores = []

    for i in range(batch_size):
        # 提取当前样本的组件信息
        if isinstance(components_list, torch.Tensor):
            # 处理张量格式的组件信息
            ref_comps = components_list[i]
            valid_ref_comps = (ref_comps[:, 0] > 0).sum().item()
            if valid_ref_comps > 0:
                component_score = 0.5  # 简化处理
            else:
                component_score = 0.0
        else:
            # 处理字典格式的组件信息
            if isinstance(components_list, list) and i < len(components_list):
                comp_info = components_list[i]
                if isinstance(comp_info, dict) and 'reference' in comp_info:
                    ref_comps = comp_info['reference']
                    if len(ref_comps) > 0:
                        component_score = 0.5  # 简化处理
                    else:
                        component_score = 0.0
                else:
                    component_score = 0.0
            else:
                component_score = 0.0

        component_alignment_scores.append(component_score)

    avg_component_score = np.mean(component_alignment_scores) if component_alignment_scores else 0.0

    # 3. 变化类型对齐指标
    change_type_alignment = compute_change_type_alignment(text_features, components_list)

    return {
        'v2t_accuracy': v2t_accuracy,
        't2v_accuracy': t2v_accuracy,
        'avg_similarity': similarity_matrix.diag().mean().item(),
        'component_alignment_score': avg_component_score,
        'change_type_alignment': change_type_alignment,
        'similarity_matrix': similarity_matrix.cpu().numpy()
    }


def compute_change_type_alignment(text_features, components_list):
    """
    分析文本特征中是否包含了变化类型信息
    """
    # 简化实现：检查文本特征是否区分不同变化类型
    # 实际应用中可以使用聚类等方法
    return 0.5  # 占位值


def compute_structured_alignment_metrics(pred_changes, gt_changes):
    """
    计算结构化变化的对齐指标
    """
    metrics = {
        'added_precision': 0.0,
        'added_recall': 0.0,
        'moved_precision': 0.0,
        'moved_recall': 0.0,
        'removed_precision': 0.0,
        'removed_recall': 0.0
    }

    # 这里需要实现具体的结构化变化匹配逻辑
    # 暂时返回简化结果
    return metrics


def parse_differ_text(differ_text: str) -> Dict:
    """
    解析differ_text为结构化数据
    示例输入: 'Removed TextView from position (0, 9, 144, 239); Added TextView at position (23, 116, 121, 141)'
    """
    changes = {
        'added': [],
        'removed': [],
        'moved': []
    }

    if not differ_text:
        return changes

    # 按分号分割不同变化
    change_phrases = differ_text.split(';')

    for phrase in change_phrases:
        phrase = phrase.strip()
        if not phrase:
            continue

        # 解析变化类型
        if phrase.startswith('Added'):
            change_type = 'added'
            # 提取组件类型和位置
            parts = phrase.split(' at position ')
            if len(parts) == 2:
                comp_type = parts[0].replace('Added ', '').strip()
                bbox_str = parts[1].strip('()')
                bbox = [float(x) for x in bbox_str.split(',')]
                changes['added'].append({
                    'type': comp_type,
                    'bbox': bbox
                })

        elif phrase.startswith('Removed'):
            change_type = 'removed'
            parts = phrase.split(' from position ')
            if len(parts) == 2:
                comp_type = parts[0].replace('Removed ', '').strip()
                bbox_str = parts[1].strip('()')
                bbox = [float(x) for x in bbox_str.split(',')]
                changes['removed'].append({
                    'type': comp_type,
                    'bbox': bbox
                })

        elif ' to ' in phrase and ' from ' in phrase:
            change_type = 'moved'
            # 格式: "TextView from (0, 9, 144, 239) to (23, 116, 121, 141)"
            parts = phrase.split(' from ')
            if len(parts) == 2:
                comp_type = parts[0].strip()
                positions = parts[1].split(' to ')
                if len(positions) == 2:
                    from_bbox = [float(x) for x in positions[0].strip('()').split(',')]
                    to_bbox = [float(x) for x in positions[1].strip('()').split(',')]
                    changes['moved'].append({
                        'type': comp_type,
                        'from_bbox': from_bbox,
                        'to_bbox': to_bbox
                    })

    return changes


def compute_text_representation_quality(text_features, components_list, config):
    """
    评估文本特征的质量
    """
    batch_size = text_features.shape[0]

    # 1. 特征多样性
    feature_norms = text_features.norm(dim=1)
    norm_std = feature_norms.std().item()
    norm_mean = feature_norms.mean().item()

    # 2. 聚类质量（简化）
    # 使用简单的相似度分布
    text_similarity = torch.matmul(
        F.normalize(text_features, dim=-1),
        F.normalize(text_features, dim=-1).T
    )
    similarity_std = text_similarity.std().item()

    return {
        'text_norm_mean': norm_mean,
        'text_norm_std': norm_std,
        'text_similarity_std': similarity_std,
        'text_feature_dim': text_features.shape[-1]
    }


def compute_alignment_metrics(visual_features, text_features, use_normalized=True, threshold=0.5):
    """
    计算视觉-文本对齐指标

    Args:
        visual_features: [B, D] 视觉特征
        text_features: [B, D] 文本特征
        use_normalized: 是否使用归一化特征计算相似度
        threshold: 对齐阈值

    Returns:
        metrics: 包含各项指标的字典
    """
    batch_size = visual_features.shape[0]

    if use_normalized:
        # 归一化特征
        visual_norm = F.normalize(visual_features, dim=-1)
        text_norm = F.normalize(text_features, dim=-1)
        # 相似度矩阵（余弦相似度）
        similarity_matrix = torch.matmul(visual_norm, text_norm.T)  # [B, B]
    else:
        # 直接使用原始特征的点积
        similarity_matrix = torch.matmul(visual_features, text_features.T)  # [B, B]

    # 视觉->文本检索准确率
    v2t_pred = similarity_matrix.argmax(dim=1)  # 每个视觉特征对应的最相似文本
    v2t_labels = torch.arange(batch_size, device=visual_features.device)
    v2t_correct = (v2t_pred == v2t_labels).sum().item()
    v2t_accuracy = v2t_correct / batch_size

    # 文本->视觉检索准确率 - 修复：应该检查文本检索到的视觉是否匹配
    t2v_pred = similarity_matrix.argmax(dim=0)  # 每个文本特征对应的最相似视觉
    # 这里需要使用相同的标签
    t2v_correct = (t2v_pred == torch.arange(batch_size, device=visual_features.device)).sum().item()
    t2v_accuracy = t2v_correct / batch_size

    # 平均相似度（对角线元素）
    avg_similarity = similarity_matrix.diag().mean().item()

    # 使用匈牙利算法进行最佳匹配
    similarity_np = similarity_matrix.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(-similarity_np)  # 最大化相似度

    # 计算匹配准确率
    matched_correct = sum(1 for i in range(len(row_ind)) if row_ind[i] == col_ind[i])
    hungarian_accuracy = matched_correct / batch_size if batch_size > 0 else 0.0

    # 相似度统计
    sim_mean = similarity_matrix.mean().item()
    sim_std = similarity_matrix.std().item()
    sim_diag_mean = similarity_matrix.diag().mean().item()

    # 计算非对角线元素的平均值
    if batch_size > 1:
        total_sum = similarity_matrix.sum().item()
        diag_sum = similarity_matrix.diag().sum().item()
        off_diag_sum = total_sum - diag_sum
        off_diag_count = batch_size * batch_size - batch_size
        sim_off_diag_mean = off_diag_sum / off_diag_count if off_diag_count > 0 else 0.0
    else:
        sim_off_diag_mean = 0.0

    return {
        'v2t_accuracy': v2t_accuracy,
        't2v_accuracy': t2v_accuracy,
        'hungarian_accuracy': hungarian_accuracy,
        'avg_similarity': avg_similarity,
        'sim_mean': sim_mean,
        'sim_std': sim_std,
        'sim_diag_mean': sim_diag_mean,
        'sim_off_diag_mean': sim_off_diag_mean,
        'similarity_matrix': similarity_np
    }


def compute_change_detection_metrics(pred_masks, true_masks, threshold=0.5):
    """
    计算变化检测指标

    Args:
        pred_masks: [B, H, W] 预测的mask（概率值）
        true_masks: [B, H, W] 真实的mask
        threshold: 二值化阈值

    Returns:
        metrics: 包含各项指标的字典
    """
    if isinstance(pred_masks, torch.Tensor):
        pred_masks = pred_masks.cpu().numpy()
    if isinstance(true_masks, torch.Tensor):
        true_masks = true_masks.cpu().numpy()

    # 二值化
    pred_binary = (pred_masks > threshold).astype(np.float32)
    true_binary = (true_masks > 0.5).astype(np.float32)

    # 逐样本计算指标
    batch_size = pred_masks.shape[0]
    all_precision = []
    all_recall = []
    all_f1 = []
    all_iou = []
    all_accuracy = []

    for i in range(batch_size):
        pred = pred_binary[i].flatten()
        true = true_binary[i].flatten()

        # TP, FP, FN
        tp = np.sum((pred == 1) & (true == 1))
        fp = np.sum((pred == 1) & (true == 0))
        fn = np.sum((pred == 0) & (true == 1))
        tn = np.sum((pred == 0) & (true == 0))

        # 计算指标
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)
        all_iou.append(iou)
        all_accuracy.append(accuracy)

    # 计算平均值
    avg_precision = np.mean(all_precision) if all_precision else 0.0
    avg_recall = np.mean(all_recall) if all_recall else 0.0
    avg_f1 = np.mean(all_f1) if all_f1 else 0.0
    avg_iou = np.mean(all_iou) if all_iou else 0.0
    avg_accuracy = np.mean(all_accuracy) if all_accuracy else 0.0

    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1_score': avg_f1,
        'iou': avg_iou,
        'accuracy': avg_accuracy,
        'tp': np.sum([np.sum((pred_binary[i] == 1) & (true_binary[i] == 1)) for i in range(batch_size)]),
        'fp': np.sum([np.sum((pred_binary[i] == 1) & (true_binary[i] == 0)) for i in range(batch_size)]),
        'fn': np.sum([np.sum((pred_binary[i] == 0) & (true_binary[i] == 1)) for i in range(batch_size)]),
        'tn': np.sum([np.sum((pred_binary[i] == 0) & (true_binary[i] == 0)) for i in range(batch_size)])
    }


def print_metrics(metrics, prefix=""):
    """打印评估指标"""
    if prefix:
        print(f"\n{prefix}评估指标:")

    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if key in ['v2t_accuracy', 't2v_accuracy', 'hungarian_accuracy',
                       'precision', 'recall', 'f1_score', 'iou', 'accuracy']:
                print(f"  {key}: {value:.4f}")
            elif key.endswith('loss'):
                print(f"  {key}: {value:.4f}")
            elif key.startswith('sim_'):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


def print_enhanced_metrics(metrics, prefix=""):
    """打印增强的评估指标"""
    if prefix:
        print(f"\n{prefix}评估指标:")

    # 分类打印不同类型指标
    print("  === 检索指标 ===")
    for key in ['v2t_accuracy', 't2v_accuracy', 'avg_similarity']:
        if key in metrics:
            print(f"    {key}: {metrics[key]:.4f}")

    print("  === 对齐质量 ===")
    for key in ['component_alignment_score', 'change_type_alignment']:
        if key in metrics:
            print(f"    {key}: {metrics[key]:.4f}")

    print("  === 特征质量 ===")
    for key in ['text_norm_mean', 'text_norm_std', 'text_similarity_std']:
        if key in metrics:
            print(f"    {key}: {metrics[key]:.4f}")

    print("  === 损失指标 ===")
    for key in metrics:
        if key.endswith('loss'):
            print(f"    {key}: {metrics[key]:.4f}")


if __name__ == "__main__":
    # 测试代码
    batch_size = 4
    hidden_dim = 768

    # 测试对齐指标
    visual_features = torch.randn(batch_size, hidden_dim)
    text_features = torch.randn(batch_size, hidden_dim)

    metrics = compute_alignment_metrics(visual_features, text_features)
    print_metrics(metrics, "对齐指标测试")

    # 测试变化检测指标
    pred_masks = torch.randn(batch_size, 224, 224).sigmoid()
    true_masks = torch.randint(0, 2, (batch_size, 224, 224)).float()

    metrics = compute_change_detection_metrics(pred_masks, true_masks)
    print_metrics(metrics, "变化检测指标测试")