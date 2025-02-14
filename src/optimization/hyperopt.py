from typing import Dict, Any, Callable
import optuna
from dataclasses import dataclass
import torch
import numpy as np
from src.neural.models.kenga import KengaConfig

@dataclass
class OptimizationConfig:
    n_trials: int = 100
    timeout: int = 3600  # seconds
    metric: str = 'loss'
    direction: str = 'minimize'

class HyperparameterOptimizer:
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.study = optuna.create_study(direction=config.direction)
        self.best_params = None
        
    def optimize(self, objective_fn: Callable):
        """Запускает оптимизацию гиперпараметров"""
        self.study.optimize(
            objective_fn,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout
        )
        self.best_params = self.study.best_params
        return self.best_params
        
    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Предлагает значения гиперпараметров"""
        return {
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
            'hidden_size': trial.suggest_int('hidden_size', 128, 1024, step=128),
            'num_layers': trial.suggest_int('num_layers', 2, 12),
            'num_heads': trial.suggest_int('num_heads', 4, 16),
            'dropout': trial.suggest_uniform('dropout', 0.1, 0.5),
            'batch_size': trial.suggest_int('batch_size', 16, 128, step=16)
        }
        
    def create_model_config(self, params: Dict[str, Any]) -> KengaConfig:
        """Создает конфигурацию модели из оптимизированных параметров"""
        return KengaConfig(
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            num_heads=params['num_heads'],
            dropout=params['dropout']
        )
        
    def plot_optimization_history(self):
        """Визуализирует историю оптимизации"""
        return optuna.visualization.plot_optimization_history(self.study)
        
    def plot_param_importances(self):
        """Визуализирует важность параметров"""
        return optuna.visualization.plot_param_importances(self.study) 