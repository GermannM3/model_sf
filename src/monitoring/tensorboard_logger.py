from torch.utils.tensorboard import SummaryWriter
import torch
from typing import Dict, Any
from pathlib import Path

class TensorBoardLogger:
    def __init__(self, log_dir: str = "runs"):
        self.writer = SummaryWriter(log_dir)
        
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Логирует метрики"""
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)
            
    def log_model_graph(self, model: torch.nn.Module, input_shape: tuple):
        """Логирует граф модели"""
        dummy_input = torch.randn(input_shape)
        self.writer.add_graph(model, dummy_input)
        
    def log_images(self, name: str, images: torch.Tensor, step: int):
        """Логирует изображения"""
        self.writer.add_images(name, images, step)
        
    def log_histograms(self, model: torch.nn.Module, step: int):
        """Логирует гистограммы весов и градиентов"""
        for name, param in model.named_parameters():
            self.writer.add_histogram(f'{name}/weights', param.data, step)
            if param.grad is not None:
                self.writer.add_histogram(f'{name}/gradients', param.grad, step)
                
    def close(self):
        self.writer.close() 