import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional

class AttentionVisualizer:
    def __init__(self):
        self.attention_maps = []
        
    def capture_attention(self, model, input_ids: torch.Tensor) -> List[torch.Tensor]:
        """Получает карты внимания из модели"""
        attention_maps = []
        
        def hook_fn(module, input, output):
            attention_maps.append(output.detach())
            
        # Регистрируем хуки для слоев внимания
        hooks = []
        for layer in model.layers:
            hooks.append(layer.attention.register_forward_hook(hook_fn))
            
        # Прямой проход
        with torch.no_grad():
            model(input_ids)
            
        # Удаляем хуки
        for hook in hooks:
            hook.remove()
            
        return attention_maps
        
    def plot_attention_map(
        self,
        attention_matrix: torch.Tensor,
        tokens: Optional[List[str]] = None,
        layer_idx: int = 0,
        head_idx: int = 0
    ):
        """Визуализирует карту внимания"""
        att_map = attention_matrix[0, head_idx].cpu().numpy()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            att_map,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='viridis'
        )
        plt.title(f'Attention Map - Layer {layer_idx}, Head {head_idx}')
        plt.tight_layout()
        return plt.gcf()
        
    def analyze_attention_patterns(self, attention_maps: List[torch.Tensor]):
        """Анализирует паттерны внимания"""
        patterns = {}
        
        for layer_idx, layer_attention in enumerate(attention_maps):
            # Находим самые сильные связи
            max_attention = torch.max(layer_attention, dim=-1)[0]
            mean_attention = torch.mean(layer_attention, dim=1)
            
            patterns[f'layer_{layer_idx}'] = {
                'max_attention': max_attention.cpu().numpy(),
                'mean_attention': mean_attention.cpu().numpy(),
                'attention_entropy': self._compute_attention_entropy(layer_attention)
            }
            
        return patterns
        
    def _compute_attention_entropy(self, attention: torch.Tensor) -> float:
        """Вычисляет энтропию внимания"""
        attention = attention.cpu().numpy()
        entropy = -np.sum(attention * np.log(attention + 1e-10), axis=-1)
        return float(np.mean(entropy)) 