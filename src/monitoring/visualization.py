import plotly.graph_objects as go
from typing import List, Dict
import pandas as pd

class MetricsVisualizer:
    def __init__(self):
        self.metrics_history: List[Dict] = []
        
    def add_metrics(self, metrics: Dict):
        self.metrics_history.append(metrics)
        
    def plot_loss_curve(self) -> go.Figure:
        df = pd.DataFrame(self.metrics_history)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['loss'], mode='lines', name='Training Loss'))
        fig.update_layout(title='Training Loss Over Time', xaxis_title='Step', yaxis_title='Loss')
        return fig
        
    def plot_accuracy(self) -> go.Figure:
        df = pd.DataFrame(self.metrics_history)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['accuracy'], mode='lines', name='Accuracy'))
        fig.update_layout(title='Model Accuracy Over Time', xaxis_title='Step', yaxis_title='Accuracy')
        return fig 