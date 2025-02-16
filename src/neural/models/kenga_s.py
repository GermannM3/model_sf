from dataclasses import dataclass
import torch
import torch.nn as nn
import os

@dataclass
class KengaSConfig:
    vocab_size: int = 50000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    dropout: float = 0.1
    grpo_reward_weight: float = 0.5   # Вес награды GRPO

class KengaSAttention(nn.Module):
    def __init__(self, config: KengaSConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_heads
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        batch_size = x.size(0)
        q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        return context

class KengaSLayer(nn.Module):
    def __init__(self, config: KengaSConfig):
        super().__init__()
        self.attention = KengaSAttention(config)
        self.intermediate = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout)
        )
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
    
    def forward(self, x):
        attn_output = self.attention(x)
        x = self.layer_norm1(x + attn_output)
        inter_output = self.intermediate(x)
        x = self.layer_norm2(x + inter_output)
        return x

class KengaSModel(nn.Module):
    def __init__(self, config: KengaSConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([KengaSLayer(config) for _ in range(config.num_layers)])
        # Две головы: для ответа и для рассуждения (chain-of-thought)
        self.answer_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.reasoning_head = nn.Linear(config.hidden_size, config.vocab_size)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        # Берем представление первого токена для обеих задач
        answer_logits = self.answer_head(x[:, 0, :])
        reasoning_logits = self.reasoning_head(x[:, 0, :])
        return {"answer_logits": answer_logits, "reasoning_logits": reasoning_logits}
    
    def grpo_learn(self, input_ids):
        """
        Реализация GRPO-обучения (dummy-версия): комбинируем кросс-энтропийный лосс для двух голов.
        В реальной системе здесь вычисляются награды и производится оптимизация по групповой относительной политике.
        """
        outputs = self.forward(input_ids)
        answer_logits = outputs["answer_logits"]
        reasoning_logits = outputs["reasoning_logits"]
        # Используем dummy-таргет (нулевые индексы) для демонстрации
        target = torch.zeros(answer_logits.size(0), dtype=torch.long, device=answer_logits.device)
        loss_fn = nn.CrossEntropyLoss()
        answer_loss = loss_fn(answer_logits, target)
        reasoning_loss = loss_fn(reasoning_logits, target)
        loss = (1 - self.config.grpo_reward_weight) * answer_loss + self.config.grpo_reward_weight * reasoning_loss
        return loss 