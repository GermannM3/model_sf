import asyncio
from dataclasses import dataclass
from typing import Optional
import aiohttp
from aiohttp import web

@dataclass
class NetworkError(Exception):
    message: str

class P2PNetwork:
    def __init__(self):
        self.peers = set()
        self.app = web.Application()
        self.app.router.add_post('/peer', self.handle_peer)
        
    @classmethod
    async def create(cls) -> 'P2PNetwork':
        network = cls()
        return network
        
    async def handle_peer(self, request: web.Request) -> web.Response:
        peer_data = await request.json()
        self.peers.add(peer_data['address'])
        return web.Response(text="Peer registered")
        
    async def listen(self):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', 8000)
        await site.start()
        
        while True:
            await asyncio.sleep(60)  # Проверка пиров каждую минуту
            await self.discover_peers()
            
    async def discover_peers(self):
        async with aiohttp.ClientSession() as session:
            for peer in self.peers.copy():
                try:
                    async with session.get(f'http://{peer}/status') as response:
                        if response.status != 200:
                            self.peers.remove(peer)
                except:
                    self.peers.remove(peer) 