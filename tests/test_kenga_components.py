import pytest
import torch
from src.neural.models.kenga import KengaAttention, KengaLayer, KengaConfig

@pytest.fixture
def config():
    return KengaConfig(
        vocab_size=1000,
        hidden_size=128,
        num_layers=2,
        num_heads=4
    )

def test_attention_mechanism(config):
    attention = KengaAttention(config)
    batch_size = 4
    seq_length = 10
    x = torch.randn(batch_size, seq_length, config.hidden_size)
    
    output = attention(x)
    assert output.shape == (batch_size, seq_length, config.hidden_size)
    
def test_layer_forward(config):
    layer = KengaLayer(config)
    batch_size = 4
    seq_length = 10
    x = torch.randn(batch_size, seq_length, config.hidden_size)
    
    output = layer(x)
    assert output.shape == (batch_size, seq_length, config.hidden_size)
    
@pytest.mark.parametrize("batch_size,seq_length", [
    (1, 5),
    (4, 10),
    (8, 20),
])
def test_model_different_sizes(config, batch_size, seq_length):
    from src.neural.models.kenga import KengaModel
    
    model = KengaModel(config)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    
    contrastive, predictions = model(input_ids)
    assert contrastive.shape == (batch_size, config.hidden_size)
    assert predictions.shape == (batch_size, seq_length, config.vocab_size) 