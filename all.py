import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, accuracy_score


class GUIChangeEvaluator:
    """统一评估脚本，支持所有基线方法"""

    def __init__(self, model_name='SemAlign-GUI'):
        self.model_name = model_name
        self.results = {}

    def compute_metrics(self, y_true, y_pred, threshold=0.5):
        """计算所有评估指标"""

        # Binarize predictions
        y_pred_bin = (y_pred > threshold).astype(np.float32)
        y_true_bin = (y_true > 0).astype(np.float32)

        # Flatten arrays
        y_true_flat = y_true_bin.flatten()
        y_pred_flat = y_pred_bin.flatten()

        metrics = {
            'precision': precision_score(y_true_flat, y_pred_flat, zero_division=0),
            'recall': recall_score(y_true_flat, y_pred_flat, zero_division=0),
            'f1': f1_score(y_true_flat, y_pred_flat, zero_division=0),
            'iou': jaccard_score(y_true_flat, y_pred_flat, zero_division=0),
            'accuracy': accuracy_score(y_true_flat, y_pred_flat)
        }

        return metrics

    def evaluate_model(self, model, dataloader, device='cuda'):
        """评估单个模型"""
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3:  # For ChangeFormer
                    img1, img2, labels = batch
                    preds = model(img1.to(device), img2.to(device))
                elif len(batch) == 4:  # For UI-BERT
                    img1, img2, text, labels = batch
                    preds = model(img1.to(device), text.to(device))
                else:
                    raise ValueError("Unsupported batch format")

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.numpy())

        # Concatenate all batches
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Compute metrics
        metrics = self.compute_metrics(all_labels, all_preds)

        self.results[self.model_name] = metrics
        return metrics

    def compare_all_models(self, models_dict, dataloader):
        """比较所有模型"""
        all_results = {}

        for model_name, model in models_dict.items():
            print(f"\nEvaluating {model_name}...")
            metrics = self.evaluate_model(model, dataloader)
            all_results[model_name] = metrics

            print(f"{model_name} Results:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")

        # Create comparison table
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)

        metrics_list = ['f1', 'iou', 'accuracy']
        header = f"{'Model':<20} {'F1-Score':<12} {'IoU':<12} {'Accuracy':<12}"
        print(header)
        print("-" * 60)

        for model_name, metrics in all_results.items():
            row = f"{model_name:<20} "
            for metric in metrics_list:
                row += f"{metrics.get(metric, 0):.4f}{'':<8}"
            print(row)

        return all_results


# 主执行脚本
if __name__ == "__main__":
    # 1. 初始化模型
    changeforner = ChangeFormer()
    ui_bert = UIBERT()

    # 2. 加载数据
    from dataset import EnhancedGUIDataset

    dataset = EnhancedGUIDataset('/home/common-dir/result/out')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)

    # 3. 评估
    evaluator = GUIChangeEvaluator()

    models = {
        'ChangeFormer': changeforner,
        'UI-BERT': ui_bert,
        # 'GPT-4o' 需要通过API单独评估
    }

    results = evaluator.compare_all_models(models, dataloader)

    # 4. 保存结果
    import json

    with open('baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nEvaluation completed. Results saved to baseline_results.json")