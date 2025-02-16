# Kenga AI Chat & Model

Данный проект включает в себя несколько компонентов:
- **Тренировка модели** (скрипт `src/main_dual.py`)
- **Веб-интерфейс для инференса (чат)** – сервер FastAPI с маршрутами `/chat` и `/graphs` (скрипты: `src/inference_server.py` и `src/run_all.py`)
- **Телеграм‑бот** (скрипт `src/telegram_bot.py`)

Проект реализован с использованием FastAPI, uvicorn и multiprocessing для одновременного запуска различных процессов.

---

## Установка

1. **Клонировать репозиторий**
   Если вы ещё не скачали проект:
   ```bash
   git clone https://gitlab.com/kenga1/kenga-ai.git
   cd kenga-ai
   ```

2. **Установить зависимости**  
   Убедитесь, что у вас установлен Python 3.11 (или выше) и выполните:
   ```bash
   pip install -r requirements.txt
   ```

---

## Запуск на ПК

### 1. Запуск тренировки модели

Для запуска обучения модели напрямую выполните:
```bash
python -m src.main_dual
```
*Либо*, чтобы запустить все компоненты (тренировку, веб-интерфейс и Telegram‑бот) одновременно, используйте:
```bash
python -m src.run_all
```

### 2. Запуск веб-интерфейса

Если вы хотите запустить только сервер инференса:
```bash
python -m src.inference_server
```
После запуска сервера откройте в браузере следующие ссылки:
- **Чат:** [http://localhost:8000/chat](http://localhost:8000/chat)
- **Графики обучения:** [http://localhost:8000/graphs](http://localhost:8000/graphs)

### 3. Запуск Telegram‑бота

Чтобы запустить Telegram‑бота отдельно:
```bash
python -m src.telegram_bot
```
**Важно!** Перед запуском бота убедитесь, что задана переменная окружения `TELEGRAM_BOT_TOKEN` (например, через файл `.env`).

---

## Google Colab

Вы также можете протестировать проект через Jupyter Notebook в Google Colab. Для этого откройте следующий ноутбук:

[Открыть в Google Colab](https://colab.research.google.com/github/kenga1/kenga-ai/blob/main/notebooks/demo.ipynb)

Убедитесь, что файл `notebooks/demo.ipynb` существует в репозитории и содержит демонстрационный пример использования модели и веб-интерфейса.

---

## Структура проекта

```
kenga-ai/
├── src/
│   ├── __init__.py
│   ├── run_all.py                # Запуск всех компонентов одновременно
│   ├── main_dual.py              # Логика тренировки модели
│   ├── telegram_bot.py           # Запуск Telegram‑бота
│   ├── inference_server.py       # Запуск инференс‑сервера (чат + графики)
│   ├── templates/
│   │   ├── chat.html             # Шаблон веб-интерфейса для чата
│   │   └── graphs.html           # Шаблон страницы графиков обучения
│   └── static/
│       ├── style_caa.css         # Стили caa для чата и графиков
│       └── graphs/
│           └── training_graph.png  # График обучения (сгенерированный в процессе обучения)
├── requirements.txt
└── README.md
```

---

## Примечания

- **Зависимости:** Убедитесь, что в `requirements.txt` присутствуют все необходимые пакеты (например, `fastapi`, `uvicorn`, `python-dotenv` и прочие).
- **Переменные окружения:** Для корректной работы Telegram‑бота задайте `TELEGRAM_BOT_TOKEN` (например, через файл `.env`).
- **Запуск всех компонентов:** Запуск через `python -m src.run_all` использует multiprocessing для одновременного старта тренинга, инференса и Telegram‑бота.

---

MIT License - see LICENSE file for details. 
