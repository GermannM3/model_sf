import torch
import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients, Saliency, DeepLift

class SensitivityAnalyzer:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.integrated_gradients = IntegratedGradients(model)
        self.saliency = Saliency(model)
        self.deeplift = DeepLift(model)
        
    def compute_input_sensitivity(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Вычисляет чувствительность модели к входным данным"""
        attributions = {
            'integrated_gradients': self.integrated_gradients.attribute(input_tensor),
            'saliency': self.saliency.attribute(input_tensor),
            'deeplift': self.deeplift.attribute(input_tensor)
        }
        return attributions
        
    def visualize_sensitivity(self, attributions: Dict[str, torch.Tensor]):
        """Визуализирует карты чувствительности"""
        fig, axes = plt.subplots(1, len(attributions), figsize=(15, 5))
        
        for ax, (method, attr) in zip(axes, attributions.items()):
            im = ax.imshow(attr.sum(dim=1).abs().cpu().numpy())
            ax.set_title(f'{method} Attribution')
            plt.colorbar(im, ax=ax)
            
        plt.tight_layout()
        return fig 