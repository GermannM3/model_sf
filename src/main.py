import asyncio
from typing import Optional
import uvicorn
from fastapi import FastAPI
from src.bot.telegram import TelegramBot
from src.web.interface import WebServer
from src.network.p2p import P2PNetwork
from src.autonomous.learning import AutonomousLearner
from src.neural.cortex import DefaultCortex

async def main():
    cortex = DefaultCortex()
    telegram_bot = TelegramBot()
    web_server = WebServer(cortex)
    p2p = await P2PNetwork.create()
    learner = AutonomousLearner(cortex)

    await asyncio.gather(
        telegram_bot.run(),
        web_server.start(),
        p2p.listen(),
        learner.start_cycle()
    )

if __name__ == "__main__":
    asyncio.run(main()) 