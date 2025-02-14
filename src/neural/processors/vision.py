import torch
import torch.nn as nn
from typing import List

class ImageProcessor:
    def __init__(self):
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def process_image(self, image_data: bytes) -> List[float]:
        # Заглушка для обработки изображений
        return [0.5, 0.2, 0.8]
        
    def __call__(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.conv1(input_tensor)
        
    @classmethod
    def default(cls) -> 'ImageProcessor':
        return cls() 