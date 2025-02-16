import torch
import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class NeuronAnalyzer:
    def __init__(self):
        self.activation_history = defaultdict(list)
        self.activation_stats = {}
        
    def register_hooks(self, model: torch.nn.Module):
        """Регистрирует хуки для сбора активаций"""
        def hook_fn(name):
            def hook(module, input, output):
                self.activation_history[name].append(output.detach().cpu())
            return hook
            
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.ReLU, torch.nn.GELU)):
                module.register_forward_hook(hook_fn(name))
                
    def compute_activation_stats(self) -> Dict[str, Dict[str, float]]:
        """Вычисляет статистики активаций нейронов"""
        stats = {}
        for name, activations in self.activation_history.items():
            if activations:
                concat_activations = torch.cat(activations, dim=0)
                stats[name] = {
                    'mean': float(concat_activations.mean()),
                    'std': float(concat_activations.std()),
                    'dead_neurons': float((concat_activations == 0).float().mean()),
                    'saturation': float((concat_activations > 0.95).float().mean())
                }
        return stats
        
    def plot_activation_distributions(self, layer_name: Optional[str] = None):
        """Визуализирует распределение активаций"""
        plt.figure(figsize=(12, 6))
        
        if layer_name and layer_name in self.activation_history:
            activations = torch.cat(self.activation_history[layer_name], dim=0)
            sns.histplot(activations.flatten().numpy(), bins=50)
            plt.title(f'Activation Distribution - {layer_name}')
        else:
            for name, activations in self.activation_history.items():
                if activations:
                    sns.histplot(
                        torch.cat(activations, dim=0).flatten().numpy(),
                        bins=50,
                        label=name,
                        alpha=0.5
                    )
            plt.title('Activation Distributions Across Layers')
            plt.legend()
            
        plt.xlabel('Activation Value')
        plt.ylabel('Count')
        return plt.gcf()
        
    def analyze_neuron_specialization(self, layer_name: str, top_k: int = 5):
        """Анализирует специализацию нейронов"""
        if layer_name not in self.activation_history:
            return None
            
        activations = torch.cat(self.activation_history[layer_name], dim=0)
        mean_activations = activations.mean(dim=0)
        
        # Находим нейроны с самой высокой средней активацией
        top_neurons = torch.topk(mean_activations, top_k)
        
        return {
            'top_neurons': top_neurons.indices.tolist(),
            'top_activations': top_neurons.values.tolist()
        } 