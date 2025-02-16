from typing import Dict, Any
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.optuna import OptunaSearch
from dataclasses import dataclass
from src.neural.models.kenga import KengaConfig

@dataclass
class RayTuneConfig:
    num_samples: int = 10
    max_epochs: int = 100
    gpus_per_trial: float = 0.5
    cpus_per_trial: int = 2

class RayTuner:
    def __init__(self, config: RayTuneConfig):
        self.config = config
        ray.init()
        
    def setup_search_space(self) -> Dict[str, Any]:
        """Определяет пространство поиска гиперпараметров"""
        return {
            "learning_rate": tune.loguniform(1e-5, 1e-2),
            "hidden_size": tune.choice([128, 256, 512, 768, 1024]),
            "num_layers": tune.choice([2, 4, 8, 12]),
            "num_heads": tune.choice([4, 8, 12, 16]),
            "dropout": tune.uniform(0.1, 0.5),
            "batch_size": tune.choice([16, 32, 64, 128])
        }
        
    def create_scheduler(self):
        """Создает планировщик для ранней остановки"""
        return ASHAScheduler(
            max_t=self.config.max_epochs,
            grace_period=10,
            reduction_factor=2
        )
        
    def run_optimization(self, trainable, search_space: Dict[str, Any]):
        """Запускает распределенную оптимизацию"""
        analysis = tune.run(
            trainable,
            config=search_space,
            num_samples=self.config.num_samples,
            scheduler=self.create_scheduler(),
            search_alg=OptunaSearch(),
            resources_per_trial={
                "cpu": self.config.cpus_per_trial,
                "gpu": self.config.gpus_per_trial
            }
        )
        
        return analysis
        
    def get_best_config(self, analysis) -> KengaConfig:
        """Получает лучшую конфигурацию"""
        best_trial = analysis.get_best_trial("loss", "min", "last")
        return KengaConfig(
            hidden_size=best_trial.config["hidden_size"],
            num_layers=best_trial.config["num_layers"],
            num_heads=best_trial.config["num_heads"],
            dropout=best_trial.config["dropout"]
        ) 