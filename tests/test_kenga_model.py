import pytest
import torch
from src.neural.models.kenga import KengaModel, KengaConfig

@pytest.fixture
def model():
    config = KengaConfig(vocab_size=1000, hidden_size=128, num_layers=2)
    return KengaModel(config)

def test_model_forward():
    config = KengaConfig(vocab_size=1000, hidden_size=128, num_layers=2)
    model = KengaModel(config)
    
    batch_size = 4
    seq_length = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    
    contrastive_output, prediction_output = model(input_ids)
    
    assert contrastive_output.shape == (batch_size, config.hidden_size)
    assert prediction_output.shape == (batch_size, seq_length, config.vocab_size)

@pytest.mark.asyncio
async def test_model_self_learning():
    config = KengaConfig(vocab_size=1000, hidden_size=128, num_layers=2)
    model = KengaModel(config)
    
    batch_size = 4
    seq_length = 10
    batch = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    
    loss = model.self_learn(batch)
    assert isinstance(loss.item(), float)
    assert loss.item() > 0 