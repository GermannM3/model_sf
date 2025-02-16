import wandb
import torch
from typing import Dict, Any, Optional
from pathlib import Path

class WandBLogger:
    def __init__(self, project_name: str, config: Dict[str, Any]):
        self.run = wandb.init(
            project=project_name,
            config=config
        )
        
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        wandb.log(metrics, step=step)
        
    def log_model(self, model: torch.nn.Module, artifact_name: str):
        torch.save(model.state_dict(), "model.pt")
        artifact = wandb.Artifact(
            artifact_name,
            type="model",
            description="Trained Kenga model"
        )
        artifact.add_file("model.pt")
        wandb.log_artifact(artifact)
        
    def log_attention_map(self, figure, caption: str):
        wandb.log({"attention_map": wandb.Image(figure, caption=caption)})
        
    def finish(self):
        wandb.finish() 