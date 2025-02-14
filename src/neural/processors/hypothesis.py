import torch
import torch.nn as nn
from typing import Optional

class HypothesisGenerator:
    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
    def generate(self, input_data: torch.Tensor) -> torch.Tensor:
        return self.model(input_data)
        
    @classmethod
    def default(cls) -> 'HypothesisGenerator':
        return cls() 