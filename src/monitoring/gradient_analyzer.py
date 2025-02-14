import torch
import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class GradientAnalyzer:
    def __init__(self):
        self.grad_history = []
        self.grad_norms = []
        
    def compute_grad_stats(self, model: torch.nn.Module) -> Dict[str, float]:
        """Вычисляет статистики градиентов"""
        grad_norms = []
        grad_means = []
        grad_vars = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                grad_norms.append(grad.norm().item())
                grad_means.append(grad.mean().item())
                grad_vars.append(grad.var().item())
                
        stats = {
            'mean_norm': np.mean(grad_norms),
            'max_norm': np.max(grad_norms),
            'mean_value': np.mean(grad_means),
            'variance': np.mean(grad_vars)
        }
        
        self.grad_norms.append(stats['mean_norm'])
        return stats
        
    def check_vanishing_exploding(self, threshold_min: float = 1e-4, threshold_max: float = 1e2) -> Dict[str, bool]:
        """Проверяет наличие проблем с градиентами"""
        recent_norms = self.grad_norms[-100:] if len(self.grad_norms) > 100 else self.grad_norms
        
        return {
            'vanishing': np.mean(recent_norms) < threshold_min,
            'exploding': np.mean(recent_norms) > threshold_max
        }
        
    def plot_gradient_flow(self, model: torch.nn.Module):
        """Визуализирует поток градиентов через слои"""
        named_parameters = [(name, param) for name, param in model.named_parameters() if param.grad is not None]
        
        plt.figure(figsize=(12, 6))
        for name, param in named_parameters:
            plt.plot(param.grad.abs().mean().item(), label=name)
            
        plt.yscale('log')
        plt.xlabel('Layers')
        plt.ylabel('Gradient Magnitude (log scale)')
        plt.title('Gradient Flow')
        plt.legend()
        return plt.gcf() 