import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import re
from typing import List, Dict, Tuple, Optional

# 导入你定义的模型类
from config import config
from model3 import Stage3PhraseContrastiveModel


class SemAlignEngine:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device
        self.config = config

        print(f"正在加载 SemAlign-GUI 模型: {model_path} ...")
        # 初始化模型架构
        self.model = Stage3PhraseContrastiveModel(
            stage2_checkpoint="",  # 推理时不需要加载stage2的checkpoint，因为我们会加载完整的stage3权重
            config=self.config
        ).to(self.device)

        # 加载训练好的权重
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.eval()

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 简单的索引库 (用于任务二)
        self.retrieval_index = []

        print("✅ 模型加载完成")

    def _preprocess_image(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert('RGB')
        return self.transform(image).unsqueeze(0).to(self.device), image

    def _parse_query_to_phrase(self, query: str) -> List[Dict]:
        """
        将自然语言查询转换为模型可理解的短语结构。
        简单的关键词匹配策略，实际应用可接入LLM进行解析。
        """
        # 默认结构
        phrase = {
            'text': query,
            'type': 'unknown',
            'component': 'unknown',
            'bbox': [0, 0, 0, 0],  # 查询时位置未知
            'is_movement': False
        }

        # 简单的规则解析
        query_lower = query.lower()
        if 'added' in query_lower or 'new' in query_lower:
            phrase['type'] = 'addition'
        elif 'removed' in query_lower or 'deleted' in query_lower:
            phrase['type'] = 'removal'
        elif 'moved' in query_lower:
            phrase['type'] = 'movement'
            phrase['is_movement'] = True

        # 组件探测
        components = ['button', 'textview', 'imageview', 'edittext', 'checkbox', 'switch']
        for comp in components:
            if comp in query_lower:
                # 简单映射回标准类名 (需与PhraseParser一致)
                phrase['component'] = comp.capitalize() if comp != 'textview' else 'TextView'
                break

        return [[phrase]]  # 返回 batch_list 格式

    # =========================================================================
    # 任务一：可解释的自动化回归测试
    # =========================================================================
    def run_regression_test(self, ref_path: str, tar_path: str, save_path: str = "report.png"):
        """
        执行任务一：输入两张图，输出Mask(定位) + 描述(解释)
        """
        ref_tensor, ref_pil = self._preprocess_image(ref_path)
        tar_tensor, tar_pil = self._preprocess_image(tar_path)

        with torch.no_grad():
            # 1. 获取 Stage 1 的视觉差异和 Mask
            # 访问路径: Stage3 -> Stage2 -> Stage1
            visual_out = self.model.stage2_model.visual_encoder(ref_tensor, tar_tensor)
            pred_logits = visual_out['pred_logits']  # [1, 224, 224]
            mask_prob = torch.sigmoid(pred_logits).squeeze().cpu().numpy()

            # 2. 获取组件变化类型 (Stage 2 Component Encoder)
            # 注意：实际推理如果没有组件树输入，我们依赖纯视觉或生成的Dummy组件
            # 这里演示如何利用视觉特征推断变化类型
            # 为了生成"描述"，我们查看 patch 级别的差异特征
            patch_features = self.model.extract_patch_features(ref_tensor, tar_tensor)  # [1, 768, 14, 14]

            # 3. 生成掩码可视化
            mask_resized = Image.fromarray((mask_prob * 255).astype(np.uint8)).resize(ref_pil.size)
            heatmap = np.array(mask_resized)

            # 4. 生成自然语言描述 (T_desc)
            # 由于模型是判别式(Contrastive)，我们通过检索预定义的类型或分析Mask重心来"生成"描述
            change_ratio = mask_prob.mean()
            if change_ratio < 0.001:
                description = "No significant visual changes detected."
                status = "PASS"
            else:
                status = "FAIL"
                # 计算重心以定位
                y_indices, x_indices = np.where(mask_prob > 0.5)
                if len(y_indices) > 0:
                    center_y, center_x = np.mean(y_indices), np.mean(x_indices)
                    # 归一化坐标
                    norm_x, norm_y = center_x / 224.0, center_y / 224.0
                    description = f"Visual discrepancy detected around center ({norm_x:.2f}, {norm_y:.2f}). "
                    description += "Likely UI component modification or rendering shift."
                else:
                    description = "Minor pixel-level noise detected."

        # 5. 绘制结果报告
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(ref_pil)
        axes[0].set_title("Reference (I_ref)")
        axes[1].imshow(tar_pil)

        # 在 Target 上叠加 Mask 轮廓
        axes[1].imshow(heatmap, cmap='jet', alpha=0.3)  # 叠加热力图
        axes[1].set_title(f"Target (I_tar) + Mask (M)")

        # 文本报告
        axes[2].text(0.1, 0.8, f"Status: {status}", fontsize=15, color='red' if status == 'FAIL' else 'green')
        axes[2].text(0.1, 0.6, "Generated Report (T_desc):", fontsize=12, fontweight='bold')
        axes[2].text(0.1, 0.4, description, fontsize=10, wrap=True)
        axes[2].axis('off')
        axes[2].set_title("Diagnostic Report")

        plt.tight_layout()
        plt.savefig(save_path)
        print(f"任务一报告已保存至: {save_path}")
        return status, description, mask_prob

    # =========================================================================
    # 任务二：基于意图的语义 GUI 检索
    # =========================================================================
    def add_to_index(self, ref_path: str, tar_path: str, sample_id: str):
        """
        构建索引库：预先计算 Patch 特征
        """
        ref_tensor, _ = self._preprocess_image(ref_path)
        tar_tensor, _ = self._preprocess_image(tar_path)

        with torch.no_grad():
            # 1. 提取 Patch 特征 (来自 Stage 3)
            # [1, 768, 14, 14] -> [1, 196, 768]
            raw_patches = self.model.extract_patch_features(ref_tensor, tar_tensor)

            # 2. 编码 Patch (映射到对比学习空间 128维)
            # [1, 196, 128]
            encoded_patches = self.model.patch_encoder(raw_patches)

            # 存储归一化的特征，方便计算余弦相似度
            encoded_patches = F.normalize(encoded_patches, dim=-1)

            self.retrieval_index.append({
                'id': sample_id,
                'patches': encoded_patches.cpu(),  # [1, 196, 128]
                'tar_path': tar_path
            })
            print(f"已索引样本: {sample_id}")

    def search_gui(self, query: str, top_k: int = 1):
        """
        执行任务二：输入文本 Query，检索并高亮区域
        """
        print(f"正在检索: '{query}' ...")

        # 1. 编码查询文本 (Q) -> Phrase Vector
        phrases_batch = self._parse_query_to_phrase(query)
        with torch.no_grad():
            # [1, 768]
            phrase_feat, _ = self.model.phrase_encoder(phrases_batch, device=self.device)
            # [1, 128] -> 投影到对比空间
            query_vec = self.model.phrase_projection(phrase_feat)
            query_vec = F.normalize(query_vec, dim=-1)  # [1, 128]

        results = []

        # 2. 在索引库中搜索 (计算 Query 与 Patch 的相似度)
        for item in self.retrieval_index:
            patch_feats = item['patches'].to(self.device)  # [1, 196, 128]

            # 计算相似度矩阵: [1, 1] @ [1, 196, 128].T -> [1, 196]
            # 这里简化处理，取每个patch与query的相似度
            sim_scores = torch.matmul(query_vec, patch_feats.squeeze(0).T)  # [1, 196]

            # 获取该图片中匹配度最高的 patch 分数作为图片得分
            max_score, max_idx = torch.max(sim_scores, dim=1)

            results.append({
                'id': item['id'],
                'score': max_score.item(),
                'best_patch_idx': max_idx.item(),
                'all_scores': sim_scores.squeeze().cpu().numpy(),
                'tar_path': item['tar_path']
            })

        # 3. 排序
        results.sort(key=lambda x: x['score'], reverse=True)
        top_results = results[:top_k]

        # 4. 可视化结果 (高亮匹配区域)
        for rank, res in enumerate(top_results):
            print(f"Rank {rank + 1}: {res['id']} (Score: {res['score']:.4f})")

            img = Image.open(res['tar_path']).convert('RGB')
            plt.figure(figsize=(8, 8))
            plt.imshow(img)

            # 绘制高亮区域 (将 14x14 网格映射回 224x224)
            patch_idx = res['best_patch_idx']
            h_idx = patch_idx // 14
            w_idx = patch_idx % 14

            # Patch 大小 16x16
            x, y = w_idx * 16, h_idx * 16

            # 调整到原图尺寸 (假设原图可能不是224)
            scale_x = img.width / 224.0
            scale_y = img.height / 224.0

            rect = plt.Rectangle(
                (x * scale_x, y * scale_y),
                16 * scale_x, 16 * scale_y,
                linewidth=3, edgecolor='lime', facecolor='none'
            )
            plt.gca().add_patch(rect)
            plt.title(f"Query: {query}\nMatch Score: {res['score']:.4f}")
            plt.axis('off')
            plt.savefig(f"search_result_{rank}.png")
            plt.show()


# =========================================================================
# 主程序入口示例
# =========================================================================
if __name__ == "__main__":
    # 假设你的模型保存在这里
    MODEL_PATH = "stage3_phrase_xxx/stage3_model.pth"

    # 1. 初始化引擎
    # 注意：首次运行请确保路径正确
    try:
        engine = SemAlignEngine(model_path=MODEL_PATH)

        # ============ 测试任务一 ============
        print("\n--- Running Task 1: Regression Testing ---")
        # 请替换为实际图片路径
        engine.run_regression_test(
            ref_path="data/login_v1.png",
            tar_path="data/login_v2.png",
            save_path="regression_report.png"
        )

        # ============ 测试任务二 ============
        print("\n--- Running Task 2: Semantic Retrieval ---")
        # 1. 建立索引 (添加一些样本)
        engine.add_to_index("data/login_v1.png", "data/login_v2.png", "sample_login")
        engine.add_to_index("data/settings_v1.png", "data/settings_v2.png", "sample_settings")

        # 2. 执行查询
        query = "Show me where the login button moved"
        engine.search_gui(query, top_k=1)

    except FileNotFoundError:
        print("请确保模型文件和测试图片路径正确。")
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback

        traceback.print_exc()