import json
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional
import warnings


@dataclass
class ModelConfig:
    """模型配置类，包含所有训练参数"""

    # 路径配置
    model_root: str = "/home/common-dir/models"
    data_root: str = "/home/common-dir/result/out"
    output_dir: str = "/home/common-dir/result/training_output"

    # 模型配置
    image_model: str = "vit-large-patch16-224"
    text_model: str = "Qwen2.5-1.5B"

    # 数据配置 - 针对第二阶段调整
    image_size: Tuple[int, int] = (224, 224)
    max_text_len: int = 64  # 减少文本长度，因为differ_text通常较短
    max_components: int = 20  # 减少最大组件数，实际数据中组件不多

    # 训练配置
    batch_size: int = 16  # 第二阶段使用较小的batch size
    grad_accum_steps: int = 4
    learning_rate: float = 3e-4  # 第二阶段使用较小的学习率
    warmup_steps: int = 500
    max_steps: int = 10000

    # 内存优化配置
    mixed_precision: bool = True
    gradient_checkpointing: bool = False  # 第二阶段通常不需要
    pin_memory: bool = True
    num_workers: int = 4 # 增加工作进程数

    # 模型维度配置 - 关键修改
    visual_dim: int = 1024  # ViT-Large输出维度
    text_dim: int = 768  # 简化文本编码器维度
    hidden_dim: int = 768  # 统一隐藏维度，减小内存使用

    # 阶段特定配置
    stage1_epochs: int = 10
    stage2_epochs: int = 40  # 增加第二阶段轮数
    stage3_epochs: int = 50

    # 在 config.py 的 __post_init__ 方法中添加
    def __post_init__(self):
        """初始化后处理"""
        # CUDA检查
        self.cuda_available = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda_available else 'cpu')

        if not self.cuda_available:
            warnings.warn("CUDA不可用，使用CPU训练")
            self.batch_size = max(2, self.batch_size // 2)
            self.grad_accum_steps = max(1, self.grad_accum_steps // 2)
            self.mixed_precision = False
            self.pin_memory = False
            self.num_workers = 0

        # 打印设备信息
        print(f"设备: {self.device}")
        if self.cuda_available:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

        # 创建目录
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 维度验证
        print(f"\n维度配置验证:")
        print(f"  视觉原始维度: {self.visual_dim}")
        print(f"  文本维度: {self.text_dim}")
        print(f"  隐藏维度: {self.hidden_dim}")
        print(f"  组件特征维度: {self.hidden_dim // 2}")
        print(f"  组件变化特征维度: {2 * (self.hidden_dim // 2) + 1}")

        # 检查维度一致性
        assert self.hidden_dim == self.text_dim, \
            f"hidden_dim({self.hidden_dim})必须等于text_dim({self.text_dim})"

        # 创建各阶段目录
        for stage in ["stage1", "stage2_alignment", "stage3"]:
            stage_dir = self.output_dir / stage
            stage_dir.mkdir(exist_ok=True)
            (stage_dir / "checkpoints").mkdir(exist_ok=True)
            (stage_dir / "logs").mkdir(exist_ok=True)
    def save(self, path: str):
        """保存配置到文件"""
        config_dict = self.__dict__.copy()
        config_dict.pop('device', None)
        config_dict.pop('cuda_available', None)
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str):
        """从文件加载配置"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def auto_adjust_batch_size(self):
        """根据可用内存自动调整batch size"""
        if not self.cuda_available:
            return

        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3

            # 第二阶段内存估计
            # 图像: 2 * 224*224*3*4 = 1.2MB
            # 组件: 2 * 10*13*4 = 1KB
            # 模型参数: ~100MB
            sample_memory = 1.2 + 0.001 + 0.1  # MB

            # 计算安全batch size
            safe_memory = total_memory * 0.7  # 使用70%的GPU内存
            estimated_batch = int((safe_memory * 1024) / sample_memory)

            # 调整batch size
            adjusted_batch = max(2, min(estimated_batch, self.batch_size))
            adjusted_batch = (adjusted_batch // self.grad_accum_steps) * self.grad_accum_steps

            if adjusted_batch != self.batch_size:
                print(f"根据GPU内存自动调整batch size: {self.batch_size} -> {adjusted_batch}")
                self.batch_size = adjusted_batch

        except Exception as e:
            print(f"自动调整batch size失败: {e}")


# 创建全局配置
config = ModelConfig()