from fastapi import APIRouter, Depends
from typing import Optional
from src.neural.cortex import Cortex

router = APIRouter()

def get_cortex() -> Cortex:
    return Cortex()

@router.post("/ask")
async def handle_ask(query: str, cortex: Cortex = Depends(get_cortex)):
    return await cortex.process(query)

@router.get("/status")
async def get_status(cortex: Cortex = Depends(get_cortex)):
    return cortex.status() 