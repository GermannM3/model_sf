from dataclasses import dataclass
from typing import Dict, List
import time
import torch
from prometheus_client import Counter, Gauge, Histogram

@dataclass
class TrainingMetrics:
    loss: float
    accuracy: float
    learning_rate: float
    epoch: int
    batch_size: int
    time_taken: float

class MetricsCollector:
    def __init__(self):
        # Prometheus метрики
        self.training_loss = Gauge('kenga_training_loss', 'Training loss value')
        self.inference_time = Histogram('kenga_inference_time', 'Time for inference')
        self.requests_total = Counter('kenga_requests_total', 'Total requests processed')
        
    def record_training_metrics(self, metrics: TrainingMetrics):
        self.training_loss.set(metrics.loss)
        
    def record_inference(self, duration: float):
        self.inference_time.observe(duration)
        
    def increment_requests(self):
        self.requests_total.inc() 