import torch
import numpy as np
from typing import List, Optional
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap

class EmbeddingVisualizer:
    def __init__(self):
        self.embeddings_history = []
        
    def reduce_dimensions(
        self,
        embeddings: torch.Tensor,
        method: str = 'tsne',
        n_components: int = 2
    ) -> np.ndarray:
        """Уменьшает размерность эмбеддингов для визуализации"""
        embeddings_np = embeddings.detach().cpu().numpy()
        
        if method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42)
        elif method == 'pca':
            reducer = PCA(n_components=n_components)
        elif method == 'umap':
            reducer = umap.UMAP(n_components=n_components)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        return reducer.fit_transform(embeddings_np)
        
    def plot_embeddings(
        self,
        embeddings: torch.Tensor,
        labels: Optional[List[str]] = None,
        method: str = 'tsne'
    ):
        """Визуализирует эмбеддинги"""
        reduced_embeddings = self.reduce_dimensions(embeddings, method)
        
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            c=range(len(reduced_embeddings)) if labels is None else None
        )
        
        if labels is not None:
            for i, label in enumerate(labels):
                plt.annotate(label, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))
                
        plt.title(f'Embeddings Visualization ({method.upper()})')
        plt.colorbar(scatter)
        return plt.gcf()
        
    def analyze_embedding_space(self, embeddings: torch.Tensor) -> Dict[str, float]:
        """Анализирует свойства пространства эмбеддингов"""
        # Нормализуем эмбеддинги
        normalized_embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        
        # Вычисляем попарные косинусные сходства
        similarity_matrix = torch.matmul(normalized_embeddings, normalized_embeddings.t())
        
        return {
            'mean_similarity': similarity_matrix.mean().item(),
            'similarity_std': similarity_matrix.std().item(),
            'max_similarity': similarity_matrix.max().item(),
            'min_similarity': similarity_matrix.min().item()
        } 