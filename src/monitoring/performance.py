import time
import torch
import psutil
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class PerformanceMetrics:
    inference_time: float
    memory_usage: float
    gpu_utilization: Optional[float]
    throughput: float  # tokens/second

class PerformanceAnalyzer:
    def __init__(self):
        self.history: List[PerformanceMetrics] = []
        
    def measure_inference_time(self, model: torch.nn.Module, input_tensor: torch.Tensor) -> float:
        start_time = time.perf_counter()
        with torch.no_grad():
            _ = model(input_tensor)
        end_time = time.perf_counter()
        return end_time - start_time
        
    def get_gpu_utilization(self) -> Optional[float]:
        if torch.cuda.is_available():
            return torch.cuda.utilization()
        return None
        
    def get_memory_usage(self) -> Dict[str, float]:
        memory_stats = {
            'ram': psutil.Process().memory_info().rss / 1024**2,  # MB
            'cuda': torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        }
        return memory_stats
        
    def profile_model(self, model: torch.nn.Module, input_tensor: torch.Tensor) -> PerformanceMetrics:
        inference_time = self.measure_inference_time(model, input_tensor)
        memory = self.get_memory_usage()
        gpu_util = self.get_gpu_utilization()
        
        # Вычисляем throughput (токенов в секунду)
        num_tokens = input_tensor.shape[0] * input_tensor.shape[1]
        throughput = num_tokens / inference_time
        
        metrics = PerformanceMetrics(
            inference_time=inference_time,
            memory_usage=memory['ram'],
            gpu_utilization=gpu_util,
            throughput=throughput
        )
        
        self.history.append(metrics)
        return metrics 