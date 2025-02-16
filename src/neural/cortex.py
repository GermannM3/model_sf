from dataclasses import dataclass
import torch
from typing import Optional
from src.common.status import Status

@dataclass
class CortexError(Exception):
    message: str

class Cortex:
    def __init__(self):
        self.memory = None
        self.processors = None

    async def process(self, input_text: str) -> str:
        return input_text  # Заглушка

    def status(self) -> Status:
        return Status(
            status="ok",
            version="0.1.0"
        )

    async def memory_usage(self) -> float:
        return 42.0  # Заглушка

class DefaultCortex(Cortex):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 