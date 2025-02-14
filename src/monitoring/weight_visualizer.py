import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

class WeightVisualizer:
    def __init__(self):
        self.weight_history = {}
        
    def capture_weights(self, model: torch.nn.Module):
        """Сохраняет текущие веса модели"""
        for name, param in model.named_parameters():
            if 'weight' in name:
                self.weight_history[name] = param.data.detach().cpu()
                
    def plot_weight_distributions(self, layer_name: Optional[str] = None):
        """Визуализирует распределение весов"""
        plt.figure(figsize=(12, 6))
        
        if layer_name and layer_name in self.weight_history:
            weights = self.weight_history[layer_name]
            sns.histplot(weights.flatten().numpy(), bins=50)
            plt.title(f'Weight Distribution - {layer_name}')
        else:
            for name, weights in self.weight_history.items():
                sns.histplot(
                    weights.flatten().numpy(),
                    bins=50,
                    label=name,
                    alpha=0.5
                )
            plt.title('Weight Distributions Across Layers')
            plt.legend()
            
        plt.xlabel('Weight Value')
        plt.ylabel('Count')
        return plt.gcf()
        
    def compute_weight_stats(self) -> Dict[str, Dict[str, float]]:
        """Вычисляет статистики весов"""
        stats = {}
        for name, weights in self.weight_history.items():
            stats[name] = {
                'mean': float(weights.mean()),
                'std': float(weights.std()),
                'norm': float(weights.norm()),
                'sparsity': float((weights == 0).float().mean())
            }
        return stats
        
    def plot_weight_matrix(self, layer_name: str):
        """Визуализирует матрицу весов"""
        if layer_name not in self.weight_history:
            return None
            
        weights = self.weight_history[layer_name]
        plt.figure(figsize=(10, 10))
        sns.heatmap(weights.numpy(), cmap='viridis')
        plt.title(f'Weight Matrix - {layer_name}')
        return plt.gcf() 