import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

class ReceptiveFieldVisualizer:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.activation_maps = {}
        
    def register_hooks(self):
        """Регистрирует хуки для сбора карт активаций"""
        def hook_fn(name):
            def hook(module, input, output):
                self.activation_maps[name] = output.detach()
            return hook
            
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                module.register_forward_hook(hook_fn(name))
                
    def compute_receptive_field(
        self,
        layer_name: str,
        input_size: Tuple[int, int],
        neuron_idx: int
    ) -> np.ndarray:
        """Вычисляет рецептивное поле для конкретного нейрона"""
        # Создаем тензор градиентов для выбранного нейрона
        grad_output = torch.zeros_like(self.activation_maps[layer_name])
        grad_output[0, neuron_idx] = 1
        
        # Вычисляем градиенты по входу
        input_grad = torch.autograd.grad(
            self.activation_maps[layer_name],
            self.model.embedding.weight,
            grad_output
        )[0]
        
        return input_grad.abs().sum(dim=1).reshape(input_size).cpu().numpy()
        
    def visualize_receptive_field(self, receptive_field: np.ndarray):
        """Визуализирует рецептивное поле"""
        plt.figure(figsize=(10, 10))
        plt.imshow(receptive_field, cmap='viridis')
        plt.colorbar()
        plt.title('Receptive Field')
        return plt.gcf() 