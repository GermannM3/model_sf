# Kenga Model Documentation

## Overview

Kenga is a self-learning transformer-based model designed for autonomous learning from various data sources. It combines contrastive learning with next token prediction to develop rich representations of input data.

## Architecture

### Components

1. **KengaAttention**
   - Multi-head attention mechanism
   - Supports parallel attention computation
   - Configurable number of attention heads

2. **KengaLayer**
   - Complete transformer layer
   - Includes attention and feed-forward networks
   - Layer normalization and residual connections

3. **KengaModel**
   - Full model architecture
   - Embedding layer for input processing
   - Multiple transformer layers
   - Dual heads for contrastive and predictive learning

### Configuration

```python
@dataclass
class KengaConfig:
    vocab_size: int = 50000      # Размер словаря
    hidden_size: int = 768       # Размер скрытого состояния
    num_layers: int = 12         # Количество слоев
    num_heads: int = 12          # Количество голов внимания
    intermediate_size: int = 3072 # Размер промежуточного слоя
    dropout: float = 0.1         # Вероятность dropout
```

## Training

### Self-Learning Process

1. **Contrastive Learning**
   - Creates embeddings for input sequences
   - Maximizes similarity between related samples
   - Minimizes similarity between unrelated samples

2. **Next Token Prediction**
   - Predicts next tokens in sequence
   - Helps model understand sequential patterns
   - Supports language modeling capabilities

### Example Usage

```python
from src.neural.models.kenga import KengaModel, KengaConfig
from src.neural.trainers.autonomous_trainer import AutonomousTrainer

# Initialize model
config = KengaConfig()
model = KengaModel(config)

# Create trainer
trainer = AutonomousTrainer(model)

# Training loop
async def train():
    batch = get_training_batch()
    loss = await trainer.train_step(batch)
    print(f"Training loss: {loss}")
```

## Performance Optimization

- Uses torch.nn.functional for efficient operations
- Supports batch processing for parallel computation
- Optimized attention mechanism implementation 