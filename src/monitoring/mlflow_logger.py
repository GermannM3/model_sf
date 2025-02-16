import mlflow
import torch
from typing import Dict, Any, Optional
from pathlib import Path

class MLflowLogger:
    def __init__(self, experiment_name: str):
        mlflow.set_experiment(experiment_name)
        self.run = None
        
    def start_run(self, run_name: Optional[str] = None):
        self.run = mlflow.start_run(run_name=run_name)
        
    def log_params(self, params: Dict[str, Any]):
        mlflow.log_params(params)
        
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        mlflow.log_metrics(metrics, step=step)
        
    def log_model(self, model: torch.nn.Module, artifact_path: str):
        mlflow.pytorch.log_model(model, artifact_path)
        
    def log_figure(self, figure, artifact_path: str):
        mlflow.log_figure(figure, artifact_path)
        
    def end_run(self):
        if self.run:
            mlflow.end_run()
            self.run = None 