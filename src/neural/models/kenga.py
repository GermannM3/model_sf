import torch
import torch.nn as nn
from typing import Optional, Tuple
from dataclasses import dataclass

@dataclass
class KengaConfig:
    vocab_size: int = 50000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    dropout: float = 0.1

class KengaAttention(nn.Module):
    def __init__(self, config: KengaConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        return context

class KengaLayer(nn.Module):
    def __init__(self, config: KengaConfig):
        super().__init__()
        self.attention = KengaAttention(config)
        self.intermediate = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout)
        )
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_output = self.attention(x)
        x = self.layer_norm1(x + attention_output)
        intermediate_output = self.intermediate(x)
        x = self.layer_norm2(x + intermediate_output)
        return x

class KengaModel(nn.Module):
    def __init__(self, config: KengaConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([KengaLayer(config) for _ in range(config.num_layers)])
        self.dropout = nn.Dropout(config.dropout)
        
        # Модули для самообучения
        self.contrastive_head = nn.Linear(config.hidden_size, config.hidden_size)
        self.prediction_head = nn.Linear(config.hidden_size, config.vocab_size)
        
    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(input_ids)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x)
            
        # Получаем эмбеддинги для контрастивного обучения
        contrastive_output = self.contrastive_head(x[:, 0, :])
        
        # Получаем предсказания следующего токена
        prediction_output = self.prediction_head(x)
        
        return contrastive_output, prediction_output
        
    def self_learn(self, batch: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
        """Самообучение на основе контрастивного подхода"""
        embeddings, predictions = self(batch)
        
        # Нормализуем эмбеддинги
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        
        # Вычисляем матрицу схожести
        similarity = torch.matmul(embeddings, embeddings.t()) / temperature
        
        # Создаем метки (положительные примеры на диагонали)
        labels = torch.arange(similarity.size(0), device=similarity.device)
        
        # Вычисляем контрастивную потерю
        loss = torch.nn.CrossEntropyLoss()(similarity, labels)
        
        return loss 