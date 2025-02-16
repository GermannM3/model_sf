import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

class AdvancedVisualizer:
    def create_3d_embedding_plot(self, embeddings: np.ndarray, labels: List[str]) -> go.Figure:
        """Создает 3D визуализацию эмбеддингов"""
        tsne = TSNE(n_components=3, random_state=42)
        embeddings_3d = tsne.fit_transform(embeddings)
        
        fig = go.Figure(data=[go.Scatter3d(
            x=embeddings_3d[:, 0],
            y=embeddings_3d[:, 1],
            z=embeddings_3d[:, 2],
            mode='markers+text',
            text=labels,
            marker=dict(
                size=8,
                color=np.arange(len(embeddings)),
                colorscale='Viridis',
                opacity=0.8
            )
        )])
        
        fig.update_layout(
            title='3D Embedding Visualization',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        
        return fig
        
    def create_attention_heatmap(self, attention_weights: np.ndarray, tokens: List[str]) -> go.Figure:
        """Создает интерактивную тепловую карту внимания"""
        fig = go.Figure(data=go.Heatmap(
            z=attention_weights,
            x=tokens,
            y=tokens,
            colorscale='RdBu',
            text=np.round(attention_weights, 3),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Attention Weights Heatmap',
            xaxis_title='Target Tokens',
            yaxis_title='Source Tokens'
        )
        
        return fig
        
    def create_layer_activation_plot(self, activations: Dict[str, np.ndarray]) -> go.Figure:
        """Создает визуализацию активаций по слоям"""
        fig = go.Figure()
        
        for layer_name, layer_activations in activations.items():
            fig.add_trace(go.Violin(
                y=layer_activations.flatten(),
                name=layer_name,
                box_visible=True,
                meanline_visible=True
            ))
            
        fig.update_layout(
            title='Layer Activations Distribution',
            xaxis_title='Layer',
            yaxis_title='Activation Value'
        )
        
        return fig
        
    def create_training_dashboard(self, metrics: Dict[str, List[float]]) -> go.Figure:
        """Создает интерактивную панель мониторинга обучения"""
        fig = go.Figure()
        
        for metric_name, values in metrics.items():
            fig.add_trace(go.Scatter(
                y=values,
                name=metric_name,
                mode='lines+markers'
            ))
            
        fig.update_layout(
            title='Training Metrics Dashboard',
            xaxis_title='Step',
            yaxis_title='Value',
            hovermode='x unified'
        )
        
        return fig 