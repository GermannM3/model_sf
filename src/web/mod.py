from fastapi import FastAPI
from src.web.api import router as api_router
from src.web.interface import WebServer

def create_app() -> FastAPI:
    app = FastAPI()
    app.include_router(api_router, prefix="/api")
    return app 