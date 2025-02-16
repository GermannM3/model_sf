from dataclasses import dataclass
from typing import Optional
from fastapi import FastAPI, Request

@dataclass
class WebhookData:
    event: str
    payload: dict

class WebhookHandler:
    def __init__(self):
        self.app = FastAPI()
        self.setup_routes()
        
    def setup_routes(self):
        @self.app.post("/webhook")
        async def handle_webhook(request: Request):
            data = await request.json()
            return await self.process_webhook(WebhookData(
                event=data.get("event", ""),
                payload=data.get("payload", {})
            ))
            
    async def process_webhook(self, data: WebhookData):
        # Заглушка для обработки вебхуков
        return {"status": "processed", "event": data.event} 