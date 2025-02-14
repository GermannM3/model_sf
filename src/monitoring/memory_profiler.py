import torch
import gc
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class TensorMemoryInfo:
    size: tuple
    dtype: torch.dtype
    device: torch.device
    memory_bytes: int

class MemoryProfiler:
    def __init__(self):
        self.tensor_snapshots = []
        
    def take_snapshot(self):
        """Создает снимок состояния памяти"""
        snapshot = {}
        for obj in gc.get_objects():
            if torch.is_tensor(obj):
                tensor = obj
                info = TensorMemoryInfo(
                    size=tensor.size(),
                    dtype=tensor.dtype,
                    device=tensor.device,
                    memory_bytes=tensor.element_size() * tensor.nelement()
                )
                snapshot[id(tensor)] = info
        self.tensor_snapshots.append(snapshot)
        
    def compare_snapshots(self, snapshot1: Dict, snapshot2: Dict) -> Dict[str, int]:
        """Сравнивает два снимка памяти"""
        diff = {
            'new_tensors': len(snapshot2) - len(snapshot1),
            'memory_diff': sum(t.memory_bytes for t in snapshot2.values()) -
                         sum(t.memory_bytes for t in snapshot1.values())
        }
        return diff
        
    def get_memory_usage_by_dtype(self, snapshot: Dict) -> Dict[str, int]:
        """Анализирует использование памяти по типам данных"""
        usage = {}
        for info in snapshot.values():
            dtype_name = str(info.dtype)
            usage[dtype_name] = usage.get(dtype_name, 0) + info.memory_bytes
        return usage
        
    def get_largest_tensors(self, snapshot: Dict, top_k: int = 10) -> List[TensorMemoryInfo]:
        """Возвращает список самых больших тензоров"""
        sorted_tensors = sorted(
            snapshot.values(),
            key=lambda x: x.memory_bytes,
            reverse=True
        )
        return sorted_tensors[:top_k] 