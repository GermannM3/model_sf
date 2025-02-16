import torch
import torch.cuda.profiler as profiler
from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class GPUStats:
    memory_allocated: float  # MB
    memory_cached: float    # MB
    utilization: float      # %
    temperature: float      # °C
    power_usage: float      # W

class GPUProfiler:
    def __init__(self):
        self.stats_history: List[GPUStats] = []
        
    def start_profiling(self):
        if torch.cuda.is_available():
            profiler.start()
            
    def stop_profiling(self):
        if torch.cuda.is_available():
            profiler.stop()
            
    def get_gpu_stats(self) -> Optional[GPUStats]:
        if not torch.cuda.is_available():
            return None
            
        stats = GPUStats(
            memory_allocated=torch.cuda.memory_allocated() / 1024**2,
            memory_cached=torch.cuda.memory_reserved() / 1024**2,
            utilization=torch.cuda.utilization(),
            temperature=torch.cuda.get_device_properties(0).temperature,
            power_usage=torch.cuda.get_device_properties(0).power_usage
        )
        
        self.stats_history.append(stats)
        return stats
        
    def profile_operation(self, func, *args, **kwargs):
        """Профилирует конкретную операцию"""
        self.start_profiling()
        result = func(*args, **kwargs)
        self.stop_profiling()
        
        stats = self.get_gpu_stats()
        return result, stats
        
    def get_memory_timeline(self) -> Dict[str, List[float]]:
        """Возвращает историю использования памяти"""
        return {
            'allocated': [s.memory_allocated for s in self.stats_history],
            'cached': [s.memory_cached for s in self.stats_history]
        } 