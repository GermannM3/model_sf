import pytest
from src.neural.cortex import Cortex, DefaultCortex

@pytest.fixture
def cortex():
    return DefaultCortex()

async def test_cortex_process(cortex):
    result = await cortex.process("test input")
    assert isinstance(result, str)

def test_cortex_status(cortex):
    status = cortex.status()
    assert status.status == "ok"
    assert isinstance(status.version, str)

async def test_cortex_memory_usage(cortex):
    usage = await cortex.memory_usage()
    assert isinstance(usage, float)
    assert 0 <= usage <= 100 