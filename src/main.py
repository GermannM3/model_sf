import asyncio
from typing import Optional
import uvicorn  # type: ignore
from fastapi import FastAPI
from src.bot.telegram import TelegramBot
from src.web.interface import WebServer
from src.network.p2p import P2PNetwork
from src.autonomous.learning import AutonomousLearner
from src.neural.cortex import DefaultCortex
from src.neural.models.kenga import KengaConfig, KengaModel
from src.neural.training import KengaTrainer
from src.neural.data import TextDataset
from src.utils.tokenizer import Tokenizer  # Импортируем токенизатор
import requests
import re

def fetch_and_append_sources(file_path: str):
    urls = [
        "https://en.wikipedia.org/wiki/Natural_language_processing",
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://en.wikipedia.org/wiki/News"
    ]
    extra_content = ""
    for url in urls:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                # Простейшее удаление HTML-тегов
                text = re.sub('<[^<]+?>', '', response.text)
                extra_content += text + "\n\n"
        except Exception as e:
            print(f"Не удалось получить данные с {url}: {e}")
    if extra_content:
         with open(file_path, "a", encoding="utf-8") as f:
               f.write("\n=== EXTRA CONTENT START ===\n")
               f.write(extra_content)
               f.write("\n=== EXTRA CONTENT END ===\n")

async def main():
    # Пока не используем следующие модули:
    # cortex = DefaultCortex()  # пока не используем
    # telegram_bot = TelegramBot()  # пока не используем
    # web_server = WebServer(cortex)  # пока не используем
    # p2p = await P2PNetwork.create()  # пока не используем
    # learner = AutonomousLearner(cortex)  # пока не используем

    # Пример: файлы для обучения
    train_files = ["data/train.txt"]
    val_files = ["data/val.txt"]

    import os
    os.makedirs("data", exist_ok=True)
    # Если файлы не существуют, создаём их с образцами данных
    for file in train_files + val_files:
        if not os.path.exists(file):
            with open(file, "w", encoding="utf-8") as f:
                f.write("This is a sample sentence.\nThis is another sentence.")

    # Дополнительные источники из интернета: новости, Wikipedia и др.
    fetch_and_append_sources("data/train.txt")

    tokenizer = Tokenizer()
    tokenizer.train(train_files)
    tokenizer.save("models", "kenga")  # Сохраняем токенизатор

    train_tokens = []
    for file in train_files:
        with open(file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    tokens = tokenizer.encode(line)
                    if tokens:
                        train_tokens.append(tokens)

    val_tokens = []
    for file in val_files:
        with open(file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    tokens = tokenizer.encode(line)
                    if tokens:
                        val_tokens.append(tokens)

    train_dataset = TextDataset(train_tokens)
    val_dataset = TextDataset(val_tokens)

    # Конфигурация модели
    config = KengaConfig()
    config.vocab_size = tokenizer.vocab_size  # Обновляем vocab_size согласно токенизатору

    model = KengaModel(config)
    train_config = {
        "lr": 1e-4,
        "batch_size": 4,
        "epochs": 5,
        "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
        "model_path": "models/kenga.pth"  # Путь для сохранения модели
    }
    trainer = KengaTrainer(model, train_dataset, val_dataset, train_config)
    trainer.train()

    # Пример использования других модулей:
    # await asyncio.gather(
    #     telegram_bot.run(),
    #     web_server.start(),
    #     p2p.listen(),
    #     learner.start_cycle()
    # )

if __name__ == "__main__":
    asyncio.run(main()) 