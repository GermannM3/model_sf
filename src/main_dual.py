import asyncio
import os
import torch
from src.neural.models.kenga import KengaConfig, KengaModel
from src.neural.training import KengaTrainer
from src.neural.data import TextDataset
from src.utils.tokenizer import Tokenizer

from src.neural.models.kenga_s import KengaSConfig, KengaSModel
from src.neural.training_kenga_s import KengaSTrainer

def generate_tokens(files, tokenizer):
    tokens = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    t = tokenizer.encode(line)
                    if t:
                        tokens.append(t)
    return tokens

async def main():
    # Создаем необходимые директории и файлы с данными
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    train_files = ["data/train.txt"]
    val_files = ["data/val.txt"]
    for file in train_files + val_files:
        if not os.path.exists(file):
            with open(file, "w", encoding="utf-8") as f:
                f.write("This is a sample sentence.\nThis is another sentence.\n")
    
    tokenizer = Tokenizer()
    tokenizer.train(train_files)
    tokenizer.save("models", "kenga")
    
    train_tokens = generate_tokens(train_files, tokenizer)
    val_tokens = generate_tokens(val_files, tokenizer)
    
    train_dataset = TextDataset(train_tokens)
    val_dataset = TextDataset(val_tokens)
    
    # Инициализация модели Kenga (MML)
    config = KengaConfig()
    config.vocab_size = tokenizer.vocab_size
    model_mml = KengaModel(config)
    trainer_mml = KengaTrainer(model_mml, train_dataset, val_dataset, {
        "lr": 1e-4,
        "batch_size": 4,
        "epochs": 5,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_path": "models/kenga.pth"
    })
    
    # Инициализация модели KengaS (SML)
    config_s = KengaSConfig()
    config_s.vocab_size = tokenizer.vocab_size
    model_sml = KengaSModel(config_s)
    trainer_sml = KengaSTrainer(model_sml, train_dataset, val_dataset, {
        "lr": 1e-4,
        "batch_size": 4,
        "epochs": 5,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_path": "models/kengaS.pth"
    })
    
    # Параллельное обучение обеих моделей с использованием потоков
    import threading
    t1 = threading.Thread(target=trainer_mml.train)
    t2 = threading.Thread(target=trainer_sml.train)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
    # После обучения делаем тестовый инференс
    model_mml.eval()
    model_sml.eval()
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, 10))
    mml_output = model_mml(input_ids)
    sml_output = model_sml(input_ids)
    print("Kenga (MML) Output:", mml_output)
    print("KengaS (SML) Output:", sml_output)

if __name__ == "__main__":
    asyncio.run(main()) 