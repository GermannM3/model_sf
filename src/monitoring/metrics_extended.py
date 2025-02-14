from dataclasses import dataclass
from typing import Dict, List
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

@dataclass
class ExtendedMetrics:
    loss: float
    perplexity: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    attention_entropy: float
    embedding_similarity: float

class ExtendedMetricsCollector:
    def __init__(self):
        self.metrics_history: List[ExtendedMetrics] = []
        
    def compute_perplexity(self, loss: float) -> float:
        return float(torch.exp(torch.tensor(loss)))
        
    def compute_token_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        pred_tokens = predictions.argmax(dim=-1).flatten()
        true_tokens = targets.flatten()
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_tokens.cpu(),
            pred_tokens.cpu(),
            average='weighted'
        )
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
    def compute_embedding_similarity(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor
    ) -> float:
        similarity = torch.cosine_similarity(embeddings1, embeddings2)
        return float(similarity.mean())
        
    def collect_metrics(
        self,
        loss: float,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        attention_maps: List[torch.Tensor],
        embeddings: torch.Tensor
    ) -> ExtendedMetrics:
        token_metrics = self.compute_token_metrics(predictions, targets)
        
        metrics = ExtendedMetrics(
            loss=loss,
            perplexity=self.compute_perplexity(loss),
            accuracy=float((predictions.argmax(dim=-1) == targets).float().mean()),
            precision=token_metrics['precision'],
            recall=token_metrics['recall'],
            f1_score=token_metrics['f1'],
            attention_entropy=self._compute_attention_entropy(attention_maps),
            embedding_similarity=self.compute_embedding_similarity(
                embeddings[:, 0, :],
                embeddings[:, -1, :]
            )
        )
        
        self.metrics_history.append(metrics)
        return metrics
        
    def _compute_attention_entropy(self, attention_maps: List[torch.Tensor]) -> float:
        entropies = []
        for att_map in attention_maps:
            att_probs = torch.softmax(att_map, dim=-1)
            entropy = -torch.sum(att_probs * torch.log(att_probs + 1e-10), dim=-1)
            entropies.append(float(entropy.mean()))
        return np.mean(entropies) 