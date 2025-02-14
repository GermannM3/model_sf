from dataclasses import dataclass
from fastapi import FastAPI, WebSocket
from src.neural.cortex import Cortex

@dataclass
class Status:
    status: str
    version: str

class WebServer:
    def __init__(self, cortex: Cortex):
        self.cortex = cortex
        self.app = FastAPI()
        self.setup_routes()

    def setup_routes(self):
        @self.app.get("/status")
        async def status():
            return self.cortex.status()

        @self.app.post("/api/ask")
        async def process_query(query: str):
            return await self.cortex.process(query)

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            while True:
                data = await websocket.receive_text()
                response = await self.cortex.process(data)
                await websocket.send_text(response)

    async def start(self):
        import uvicorn
        config = uvicorn.Config(self.app, host="0.0.0.0", port=3000)
        server = uvicorn.Server(config)
        await server.serve() 