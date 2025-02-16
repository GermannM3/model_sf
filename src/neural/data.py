import torch
from torch.utils.data import Dataset
from typing import List

class TextDataset(Dataset):
    def __init__(self, tokens: List[List[int]], max_length: int = 512):
        self.tokens = tokens
        self.max_length = max_length

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        # Обрезка и добавление паддинга (0 – токен паддинга)
        input_ids = self.tokens[idx][:self.max_length]
        padding_length = self.max_length - len(input_ids)
        input_ids = input_ids + [0] * padding_length
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long)} 