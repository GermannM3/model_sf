import aiohttp
from typing import List, Optional
from bs4 import BeautifulSoup
import torch
from src.utils.crypto import hash_data

class DataCollector:
    def __init__(self):
        self.visited_urls = set()
        self.collected_data = []
        
    async def collect_from_web(self, url: str) -> Optional[str]:
        if url in self.visited_urls:
            return None
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        text = soup.get_text()
                        self.visited_urls.add(url)
                        return text
        except Exception as e:
            print(f"Error collecting data from {url}: {e}")
        return None
        
    def preprocess_text(self, text: str) -> torch.Tensor:
        # Заглушка для предобработки текста
        return torch.tensor([ord(c) for c in text[:100]])
        
    async def get_training_batch(self, batch_size: int = 32) -> torch.Tensor:
        # Заглушка для создания батча данных
        return torch.randn(batch_size, 100) 