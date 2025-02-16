from dataclasses import dataclass
from typing import Optional
import torch
from src.neural.processors.vision import ImageProcessor
from src.neural.processors.nlp import LanguageProcessor
from src.neural.processors.hypothesis import HypothesisGenerator

class Pipeline:
    def __init__(self):
        self.vision = ImageProcessor()
        self.nlp = LanguageProcessor()
        self.hypothesis = HypothesisGenerator()
        
    @classmethod
    def default(cls) -> 'Pipeline':
        return cls() 