"""
Скрипт для одновременного запуска:
- Тренинга модели (например, main_dual.py)
- Инференс-сервера (HTTP сервер, например FastAPI)
- Telegram-бота (чат-бот для общения с моделями)
"""

import multiprocessing
import sys
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse

def run_training():
    # Запускаем тренировку модели, импортируя main_dual
    # Убедитесь, что функция main() в main_dual.py не блокирует процесс бесконечным циклом,
    # или адаптируйте её запуск согласно вашим требованиям
    import src.main_dual as main_dual
    import asyncio
    asyncio.run(main_dual.main())

def run_telegram():
    # Запускаем Telegram-бота
    import src.telegram_bot as telegram_bot
    telegram_bot.main()

def run_inference():
    # Запускаем простой HTTP-сервер для инференса
    # Здесь используется FastAPI и uvicorn для демонстрации
    app = FastAPI()
    templates = Jinja2Templates(directory="src/templates")
    from fastapi.staticfiles import StaticFiles
    app.mount("/static", StaticFiles(directory="src/static"), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index_page(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

    @app.get("/chat", response_class=HTMLResponse)
    async def chat_page(request: Request):
        return templates.TemplateResponse("chat.html", {"request": request})

    @app.post("/chat")
    async def chat_endpoint(payload: dict):
        user_message = payload.get("message", "")
        # Вызываем функцию модели для получения реального ответа
        from src.model import get_model_response
        reply = get_model_response(user_message)
        return JSONResponse(content={"reply": reply})

    @app.get("/graphs", response_class=HTMLResponse)
    async def graphs_page(request: Request):
        return templates.TemplateResponse("graphs.html", {"request": request})

    @app.get("/metrics", response_class=JSONResponse)
    async def metrics():
        # В production-решении здесь следует вернуть реальные метрики обучения
        sample_metrics = {
            "labels": ["Эпоха 1", "Эпоха 2", "Эпоха 3", "Эпоха 4", "Эпоха 5"],
            "loss": [0.9, 0.7, 0.6, 0.4, 0.3]
        }
        return JSONResponse(content=sample_metrics)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    processes = []
    
    p_train = multiprocessing.Process(target=run_training)
    p_telegram = multiprocessing.Process(target=run_telegram)
    p_inference = multiprocessing.Process(target=run_inference)

    processes.extend([p_train, p_telegram, p_inference])

    for p in processes:
        p.start()

    try:
        # Ждём, чтобы все процессы завершились (обычно они работают бесконечно)
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("Остановка процессов...")
        for p in processes:
            p.terminate()
        sys.exit(0) 