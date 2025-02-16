import os
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from src.neural.models.kenga_s import KengaSModel, KengaSConfig

class KengaSTrainer:
    def __init__(self, model: KengaSModel, train_dataset: Dataset, val_dataset: Dataset, config: dict):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.model_path = self.config.get("model_path", "models/kengaS.pth")
        self.best_val_loss = float('inf')
        self.device = self.config.get("device", "cpu")
        self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.config.get("lr", 1e-4))
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.config.get("batch_size", 32), shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.config.get("batch_size", 32), shuffle=False)
        self.train_loss_history = []
        self.val_loss_history = []
        self.load_checkpoint()
    
    def train(self):
        epochs = self.config.get("epochs", 10)
        start_epoch = self.config.get("start_epoch", 0)
        current_epoch = start_epoch

        try:
            for epoch in range(start_epoch, epochs):
                current_epoch = epoch
                self.model.train()
                train_loss = 0
                for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                    input_ids = batch["input_ids"].to(self.device)
                    self.optimizer.zero_grad()
                    loss = self.model.grpo_learn(input_ids)
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()
                train_loss /= len(self.train_dataloader)
                val_loss = self.validate()
                self.train_loss_history.append(train_loss)
                self.val_loss_history.append(val_loss)
                print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                self.save_checkpoint(epoch+1, val_loss)
        except KeyboardInterrupt:
            print("Training interrupted by user!")
        finally:
            self.save_final_checkpoint(current_epoch+1)
    
    def validate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                loss = self.model.grpo_learn(input_ids)
                val_loss += loss.item()
        val_loss /= len(self.val_dataloader)
        return val_loss
    
    def save_checkpoint(self, epoch, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "best_val_loss": self.best_val_loss
            }
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            torch.save(checkpoint, self.model_path)
            print(f"Checkpoint saved to {self.model_path}")
    
    def load_checkpoint(self):
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.config["start_epoch"] = checkpoint["epoch"]
            self.best_val_loss = checkpoint["best_val_loss"]
            print(f"Loaded checkpoint from {self.model_path}")
    
    def save_final_checkpoint(self, epoch):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "best_val_loss": self.best_val_loss
        }
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(checkpoint, self.model_path)
        print(f"Final checkpoint saved to {self.model_path}") 