import torch
from typing import Optional
from src.neural.models.kenga import KengaModel, KengaConfig
from src.utils.logging import log_event

class AutonomousTrainer:
    def __init__(self, model: KengaModel, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
    async def train_step(self, batch: torch.Tensor) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        
        batch = batch.to(self.device)
        loss = self.model.self_learn(batch)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    async def evaluate(self, batch: torch.Tensor) -> float:
        self.model.eval()
        with torch.no_grad():
            batch = batch.to(self.device)
            loss = self.model.self_learn(batch)
        return loss.item()
        
    def save_checkpoint(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        log_event(f"Model checkpoint saved to {path}")
        
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        log_event(f"Model checkpoint loaded from {path}") 