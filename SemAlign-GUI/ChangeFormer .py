import os
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

# ==========================================
# 1. 数据处理与加载 (对应论文的 HDF5 格式与 224x224 缩放)
# ==========================================
class CoD_GUIDataset(Dataset):
    def __init__(self, h5_path, split='train'):
        """
        论文提到多模态数据被序列化为 HDF5 格式。
        假设 HDF5 结构为: /train/img_ref, /train/img_tar, /train/mask
        """
        super().__init__()
        self.h5_path = h5_path
        self.split = split
        
        # 预处理：统一缩放至 224x224 并归一化 (论文标准)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # 仅用于掩码的预处理 (不进行均值归一化，保持 0/1 二值)
        self.mask_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

        # 获取数据集长度
        with h5py.File(self.h5_path, 'r') as f:
            self.length = len(f[self.split]['mask'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 每次读取时打开 HDF5，兼容多线程 DataLoader
        with h5py.File(self.h5_path, 'r') as f:
            img_ref = f[self.split]['img_ref'][idx] # Shape: (H, W, 3)
            img_tar = f[self.split]['img_tar'][idx] # Shape: (H, W, 3)
            mask = f[self.split]['mask'][idx]       # Shape: (H, W), 0 or 1
            
        # 应用 Transforms
        img_ref = self.transform(img_ref)
        img_tar = self.transform(img_tar)
        mask = self.mask_transform(mask) # Shape: (1, 224, 224)
        
        return img_ref, img_tar, mask

# ==========================================
# 2. 模型初始化 (加载官方预训练权重)
# ==========================================
def get_changeformer_model(device):
    """
    这里是对接官方 ChangeFormer 的入口。
    实际使用时，你需要从 ChangeFormer 官方仓库导入模型类。
    例如: from models.ChangeFormer import ChangeFormerV6
    """
    # 假设这是官方的 ChangeFormer 初始化
    # model = ChangeFormerV6(embed_dim=256, output_nc=2) 
    
    # 为了让这个脚本独立运行，这里用一个伪代码占位。
    # 论文提到：它直接输出 224x224 的像素级二值掩码
    class DummyChangeFormer(nn.Module):
        def __init__(self):
            super().__init__()
            # 模拟特征提取和解码器
            self.conv = nn.Conv2d(6, 1, kernel_size=3, padding=1)
        def forward(self, ref, tar):
            # 真实模型会通过 Siamese Transformer 处理
            x = torch.cat([ref, tar], dim=1)
            return self.conv(x) # 输出 (B, 1, 224, 224) 的 logits
            
    model = DummyChangeFormer()
    
    # 论文：初始化加载官方预训练权重
    # pretrained_dict = torch.load('path_to_pretrained_changeformer.pth')
    # model.load_state_dict(pretrained_dict)
    
    return model.to(device)

# ==========================================
# 3. 核心训练循环 (50 Epochs, AdamW, 梯度裁剪)
# ==========================================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 论文超参数设置
    BATCH_SIZE = 64
    EPOCHS = 50
    PEAK_LR = 1e-4
    WEIGHT_DECAY = 0.01
    GRAD_CLIP = 1.0
    WARMUP_RATIO = 0.1
    
    # 初始化数据
    dataset_train = CoD_GUIDataset('cod_gui_dataset.h5', split='train')
    loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # 初始化模型
    model = get_changeformer_model(device)
    
    # 损失函数：因为是输出二值掩码，使用 BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()
    
    # 优化器：AdamW (wd=0.01)
    optimizer = AdamW(model.parameters(), lr=PEAK_LR, weight_decay=WEIGHT_DECAY)
    
    # 学习率调度器：带有 10% 线性 Warm-up 的余弦退火
    total_steps = len(loader_train) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(total_steps - warmup_steps))
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    # 开始微调
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        pbar = tqdm(loader_train, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for img_ref, img_tar, masks in pbar:
            img_ref = img_ref.to(device)
            img_tar = img_tar.to(device)
            masks = masks.to(device) # Shape: (B, 1, 224, 224)
            
            # 前向传播 (传入参考图和目标图)
            logits = model(img_ref, img_tar) 
            
            # 计算像素级损失
            loss = criterion(logits, masks)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 论文中提到的梯度裁剪阈值 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.6f}"})
            
        print(f"Epoch [{epoch+1}/{EPOCHS}] Average Loss: {total_loss/len(loader_train):.4f}")

    # 保存微调后的模型权重
    torch.save(model.state_dict(), 'changeformer_cod_gui_finetuned.pth')
    print("Training complete. Model saved.")

if __name__ == '__main__':
    train()
