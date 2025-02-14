import torch
import time
from typing import Dict, List
from collections import defaultdict
import numpy as np
from dataclasses import dataclass

@dataclass
class LayerProfile:
    forward_time: float
    backward_time: float
    memory_usage: float
    flops: int
    params: int

class LayerProfiler:
    def __init__(self):
        self.layer_profiles = defaultdict(list)
        self.hooks = []
        
    def start_profiling(self, model: torch.nn.Module):
        """Начинает профилирование слоев"""
        def forward_hook(name):
            def hook(module, input, output):
                start_time = time.perf_counter()
                
                # Профилируем прямой проход
                result = module(input[0])
                
                forward_time = time.perf_counter() - start_time
                memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                
                self.layer_profiles[name].append(LayerProfile(
                    forward_time=forward_time,
                    backward_time=0.0,  # Будет обновлено в backward_hook
                    memory_usage=memory,
                    flops=self._estimate_flops(module, input[0].shape),
                    params=sum(p.numel() for p in module.parameters())
                ))
                
                return result
            return hook
            
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                start_time = time.perf_counter()
                
                # Профилируем обратный проход
                result = module.backward(grad_input[0])
                
                backward_time = time.perf_counter() - start_time
                self.layer_profiles[name][-1].backward_time = backward_time
                
                return result
            return hook
            
        # Регистрируем хуки для всех слоев
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)):
                self.hooks.append(module.register_forward_hook(forward_hook(name)))
                self.hooks.append(module.register_backward_hook(backward_hook(name)))
                
    def stop_profiling(self):
        """Останавливает профилирование"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def get_layer_stats(self) -> Dict[str, Dict[str, float]]:
        """Возвращает статистики по слоям"""
        stats = {}
        for name, profiles in self.layer_profiles.items():
            if profiles:
                stats[name] = {
                    'mean_forward_time': np.mean([p.forward_time for p in profiles]),
                    'mean_backward_time': np.mean([p.backward_time for p in profiles]),
                    'mean_memory': np.mean([p.memory_usage for p in profiles]),
                    'total_flops': profiles[0].flops,
                    'total_params': profiles[0].params
                }
        return stats
        
    def _estimate_flops(self, module: torch.nn.Module, input_shape: torch.Size) -> int:
        """Оценивает количество операций с плавающей точкой"""
        if isinstance(module, torch.nn.Linear):
            return input_shape[0] * input_shape[1] * module.out_features * 2
        elif isinstance(module, torch.nn.LayerNorm):
            return input_shape[0] * input_shape[1] * 5  # mean, var, norm, scale, bias
        return 0 