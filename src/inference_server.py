from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory="src/templates")

@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/chat")
async def chat_endpoint(payload: dict):
    user_message = payload.get("message", "")
    from src.model import get_model_response
    reply = get_model_response(user_message)
    return JSONResponse(content={"reply": reply})

@app.get("/metrics", response_class=JSONResponse)
async def metrics():
    # Здесь можно заменить на реальные данные из файлов логов или базы данных
    sample_metrics = {
        "labels": ["Эпоха 1", "Эпоха 2", "Эпоха 3", "Эпоха 4", "Эпоха 5"],
        "loss": [0.9, 0.7, 0.6, 0.4, 0.3]
    }
    return JSONResponse(content=sample_metrics)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 