import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

# ==========================================
# 1. 数据处理 (提取组件特征与边界框)
# ==========================================
class UIBERT_ComponentDataset(Dataset):
    def __init__(self, h5_path, split='train'):
        """
        与 ChangeFormer 直接吃全图不同，UI-BERT 需要解析后的组件级数据。
        假设数据集包含每个界面的：图像、文本描述、组件边界框列表、以及组件级别的 0/1 变化标签。
        """
        super().__init__()
        self.split = split
        # 模拟加载数据：实际中你会从 HDF5 或 JSON 中读取解析好的结构化组件
        # 数据集应返回：(界面特征, 文本特征, 组件边界框, 组件级真实变化标签)
        self.length = 1000 # 假设 1000 个样本
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 模拟生成占位数据
        # 假设每个界面有 20 个解析好的组件 (N=20)
        num_components = 20
        
        # 模拟 UI-BERT 需要的输入 (视觉 token, 文本 token 等)
        # 这里用 dummy tensor 代替
        dummy_img = torch.randn(3, 224, 224)
        dummy_text = torch.randint(0, 3000, (32,))
        
        # 边界框格式: [x1, y1, x2, y2], 缩放到 224x224 尺寸内
        bboxes = torch.rand(num_components, 4) * 224 
        bboxes[:, 2:] += bboxes[:, :2] # 确保 x2>x1, y2>y1
        bboxes = torch.clamp(bboxes, 0, 224)
        
        # 组件级标签: 1 表示该组件发生了变化，0 表示未变
        comp_labels = torch.randint(0, 2, (num_components, 1)).float()
        
        return dummy_img, dummy_text, bboxes, comp_labels

# ==========================================
# 2. 模型适配 (冻结骨干 + 两层 MLP 分类头)
# ==========================================
class AdaptedUIBERT(nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()
        
        # 模拟加载官方 UI-BERT 预训练模型
        # 实际代码: self.uibert = UIBERTModel.from_pretrained(...)
        self.uibert = nn.Linear(224, hidden_dim) # 伪代码占位：模拟输出隐层特征
        
        # 论文关键点 1：冻结视觉和文本编码器，防止灾难性遗忘
        for param in self.parameters():
            param.requires_grad = False
            
        # 论文关键点 2：添加两层 MLP 分类头 (被激活以进行训练)
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1) # 输出该组件是否变化的 Logit
        )
        
    def forward(self, img, text, bboxes):
        # 1. 骨干网络提取特征 (不计算梯度)
        with torch.no_grad():
            # 真实 UI-BERT 会融合图像、文本和布局，输出每个组件的上下文特征
            # Shape: (Batch, Num_Components, Hidden_Dim)
            B, N, _ = bboxes.shape
            comp_embeddings = torch.randn(B, N, 768).to(bboxes.device) 
            
        # 2. 仅通过 MLP 分类头进行前向传播
        # Shape: (Batch, Num_Components, 1)
        logits = self.mlp_head(comp_embeddings)
        return logits

# ==========================================
# 3. 空间渲染逻辑 (核心适配机制)
# ==========================================
def render_spatial_masks(bboxes, pred_probs, threshold=0.5, img_size=224):
    """
    论文核心后处理：将分类为"变化"的边界框渲染为像素级掩码
    Overlapping spatial regions are deterministically resolved by computing the pixel-wise maximum.
    """
    B, N, _ = bboxes.shape
    # 初始化全 0 画布
    masks = torch.zeros((B, 1, img_size, img_size), device=bboxes.device)
    
    for b in range(B):
        for n in range(N):
            # 如果 MLP 判定该组件发生了变化
            if pred_probs[b, n, 0] > threshold:
                x1, y1, x2, y2 = bboxes[b, n].int()
                
                # 边界保护
                x1, y1 = max(0, x1.item()), max(0, y1.item())
                x2, y2 = min(img_size, x2.item()), min(img_size, y2.item())
                
                if x2 > x1 and y2 > y1:
                    # 将该组件区域填充为 1
                    # 后续覆盖的 1 会自然实现像素级最大值 (逻辑或) 的效果
                    masks[b, 0, y1:y2, x1:x2] = 1.0
                    
    return masks

# ==========================================
# 4. 训练循环 (仅训练 MLP 分类头)
# ==========================================
def train_uibert_head():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    BATCH_SIZE = 64
    EPOCHS = 50
    # 论文中明确指定的分类头学习率
    PEAK_LR = 5e-4 
    GRAD_CLIP = 1.0
    WARMUP_RATIO = 0.1
    
    loader_train = DataLoader(UIBERT_ComponentDataset('dummy.h5'), batch_size=BATCH_SIZE, shuffle=True)
    
    model = AdaptedUIBERT().to(device)
    
    # 仅将 MLP 的参数传入优化器
    optimizer = AdamW(model.mlp_head.parameters(), lr=PEAK_LR, weight_decay=0.01)
    
    # 组件级别的二分类损失
    criterion = nn.BCEWithLogitsLoss()
    
    # 调度器配置
    total_steps = len(loader_train) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(total_steps - warmup_steps))
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        pbar = tqdm(loader_train, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for img, text, bboxes, comp_labels in pbar:
            img, text = img.to(device), text.to(device)
            bboxes, comp_labels = bboxes.to(device), comp_labels.to(device)
            
            # 前向传播预测组件变化概率
            logits = model(img, text, bboxes)
            
            # 计算组件级损失
            loss = criterion(logits, comp_labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.mlp_head.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            # 评估时：使用预测结果渲染像素掩码 (此处仅展示调用逻辑，真实评估在 eval 阶段)
            pred_probs = torch.sigmoid(logits)
            # rendered_masks = render_spatial_masks(bboxes, pred_probs)
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.6f}"})

    torch.save(model.mlp_head.state_dict(), 'uibert_mlp_head.pth')
    print("UI-BERT MLP classification head training complete.")

if __name__ == '__main__':
    train_uibert_head()
