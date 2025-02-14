from typing import Optional, Dict, Any
import torch

class HierarchicalMemory:
    def __init__(self):
        self.short_term = {}
        self.long_term = {}
        
    def store(self, key: str, value: Any):
        self.short_term[key] = value
        
    def recall(self, key: str) -> Optional[Any]:
        return self.short_term.get(key) or self.long_term.get(key)
        
    def consolidate(self):
        # Перемещаем важные данные из краткосрочной в долгосрочную память
        for key, value in self.short_term.items():
            if self._is_important(value):
                self.long_term[key] = value
        self.short_term.clear()
        
    def _is_important(self, value: Any) -> bool:
        # Заглушка для определения важности данных
        return True 