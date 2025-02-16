# src/utils/tokenizer.py
from tokenizers import ByteLevelBPETokenizer
from pathlib import Path
from typing import List, Dict

class Tokenizer:
    def __init__(self, vocab_file: str = None, merges_file: str = None):
        if vocab_file and merges_file:
            self.tokenizer = ByteLevelBPETokenizer(vocab_file, merges_file)
        else:
            self.tokenizer = ByteLevelBPETokenizer()

    def train(self, files: List[str], vocab_size: int = 50000, min_frequency: int = 1,
              special_tokens: List[str] = ["<s>", "</s>", "<pad>", "<unk>", "<mask>"]):
        """Обучает токенизатор на данных файлах."""
        self.tokenizer.train(files, vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=special_tokens)

    def save(self, directory: str, prefix: str = "kenga"):
        """Сохраняет модель токенизатора."""
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_model(directory, prefix)

    def encode(self, text: str) -> List[int]:
        """Кодирует текст в список токенов."""
        return self.tokenizer.encode(text).ids

    def decode(self, tokens: List[int]) -> str:
        """Декодирует список токенов обратно в текст."""
        return self.tokenizer.decode(tokens)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def get_vocab(self) -> Dict[str, int]:
        return self.tokenizer.get_vocab() 