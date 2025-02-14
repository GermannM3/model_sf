import torch
from src.neural.models.kenga import KengaModel, KengaConfig
from src.neural.trainers.autonomous_trainer import AutonomousTrainer
from src.neural.data.collector import DataCollector

async def basic_usage_example():
    # Инициализация модели
    config = KengaConfig(
        vocab_size=10000,
        hidden_size=256,
        num_layers=4,
        num_heads=8
    )
    model = KengaModel(config)
    
    # Пример прямого прохода
    batch_size = 8
    seq_length = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    contrastive_output, predictions = model(input_ids)
    
    print(f"Contrastive output shape: {contrastive_output.shape}")
    print(f"Predictions shape: {predictions.shape}")

async def training_example():
    # Настройка обучения
    config = KengaConfig()
    model = KengaModel(config)
    trainer = AutonomousTrainer(model)
    collector = DataCollector()
    
    # Цикл обучения
    for epoch in range(3):
        batch = await collector.get_training_batch()
        loss = await trainer.train_step(batch)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        # Сохранение чекпоинта
        if epoch % 10 == 0:
            trainer.save_checkpoint(f"checkpoints/epoch_{epoch}.pt")

async def inference_example():
    # Загрузка обученной модели
    config = KengaConfig()
    model = KengaModel(config)
    trainer = AutonomousTrainer(model)
    trainer.load_checkpoint("checkpoints/latest.pt")
    
    # Инференс
    input_text = "Example input text"
    input_tensor = torch.tensor([[1, 2, 3, 4, 5]])  # Пример токенизации
    contrastive_output, predictions = model(input_tensor)
    
    print("Model predictions:", predictions.argmax(dim=-1))

if __name__ == "__main__":
    import asyncio
    asyncio.run(basic_usage_example())
    asyncio.run(training_example())
    asyncio.run(inference_example()) 