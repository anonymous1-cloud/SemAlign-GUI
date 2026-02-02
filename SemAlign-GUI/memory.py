import torch
import psutil
import gc
from typing import Dict
import time


class MemoryMonitor:
    """内存使用监控器"""

    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.history = []

    def get_memory_stats(self) -> Dict:
        """获取当前内存使用情况"""
        stats = {}

        # CPU内存
        cpu_mem = psutil.Process().memory_info()
        stats['cpu_rss_mb'] = cpu_mem.rss / (1024 ** 2)
        stats['cpu_vms_mb'] = cpu_mem.vms / (1024 ** 2)

        # GPU内存
        if self.cuda_available:
            stats['gpu_allocated_mb'] = torch.cuda.memory_allocated() / (1024 ** 2)
            stats['gpu_reserved_mb'] = torch.cuda.memory_reserved() / (1024 ** 2)
            stats['gpu_max_allocated_mb'] = torch.cuda.max_memory_allocated() / (1024 ** 2)

            try:
                # 尝试获取更详细的GPU信息
                gpu_props = torch.cuda.get_device_properties(0)
                stats['gpu_total_mb'] = gpu_props.total_memory / (1024 ** 2)
                stats['gpu_utilization'] = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
            except:
                pass

        return stats

    def print_memory_stats(self, prefix: str = ""):
        """打印内存使用情况"""
        stats = self.get_memory_stats()

        print(f"{prefix}内存使用情况:")
        print(f"  CPU RSS: {stats['cpu_rss_mb']:.1f} MB")
        print(f"  CPU VMS: {stats['cpu_vms_mb']:.1f} MB")

        if self.cuda_available:
            print(f"  GPU分配: {stats['gpu_allocated_mb']:.1f} MB")
            print(f"  GPU保留: {stats['gpu_reserved_mb']:.1f} MB")
            print(f"  GPU最大分配: {stats['gpu_max_allocated_mb']:.1f} MB")
            if 'gpu_total_mb' in stats:
                utilization = (stats['gpu_allocated_mb'] / stats['gpu_total_mb']) * 100
                print(f"  GPU使用率: {utilization:.1f}%")

    def clear_cache(self):
        """清理缓存"""
        gc.collect()
        if self.cuda_available:
            torch.cuda.empty_cache()

    def check_memory_safe(self, required_mb: float = 500) -> bool:
        """检查是否有足够内存"""
        stats = self.get_memory_stats()

        if not self.cuda_available:
            # CPU内存检查
            available_cpu = psutil.virtual_memory().available / (1024 ** 2)
            return available_cpu > required_mb

        # GPU内存检查
        if 'gpu_total_mb' in stats:
            gpu_available = stats['gpu_total_mb'] - stats['gpu_allocated_mb']
            return gpu_available > required_mb

        return True

    def record_peak_memory(self):
        """记录峰值内存并重置"""
        if self.cuda_available:
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            torch.cuda.reset_peak_memory_stats()
            return peak
        return 0.0


# 全局内存监控器
memory_monitor = MemoryMonitor()