from typing import Dict, List, Optional
import json
from datetime import datetime
from pathlib import Path
import mlflow
from dataclasses import dataclass, asdict

@dataclass
class ExperimentConfig:
    name: str
    description: str
    model_config: Dict
    training_config: Dict
    tags: List[str]

@dataclass
class ExperimentRun:
    id: str
    config: ExperimentConfig
    status: str
    metrics: Dict[str, float]
    artifacts: List[str]
    start_time: datetime
    end_time: Optional[datetime] = None

class ExperimentManager:
    def __init__(self, storage_dir: str = "experiments"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"file://{self.storage_dir}/mlruns")
        
    def create_experiment(self, config: ExperimentConfig) -> ExperimentRun:
        """Создает новый эксперимент"""
        mlflow.set_experiment(config.name)
        
        with mlflow.start_run() as run:
            # Логируем конфигурацию
            mlflow.log_params(config.model_config)
            mlflow.log_params(config.training_config)
            
            # Создаем запись эксперимента
            experiment = ExperimentRun(
                id=run.info.run_id,
                config=config,
                status="created",
                metrics={},
                artifacts=[],
                start_time=datetime.now()
            )
            
            # Сохраняем метаданные
            self._save_experiment_metadata(experiment)
            
            return experiment
            
    def update_metrics(self, experiment_id: str, metrics: Dict[str, float]):
        """Обновляет метрики эксперимента"""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        with mlflow.start_run(run_id=experiment_id):
            mlflow.log_metrics(metrics)
            
        experiment.metrics.update(metrics)
        self._save_experiment_metadata(experiment)
        
    def add_artifact(self, experiment_id: str, artifact_path: str, artifact_name: Optional[str] = None):
        """Добавляет артефакт к эксперименту"""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        with mlflow.start_run(run_id=experiment_id):
            mlflow.log_artifact(artifact_path, artifact_name)
            
        experiment.artifacts.append(artifact_path)
        self._save_experiment_metadata(experiment)
        
    def get_experiment(self, experiment_id: str) -> Optional[ExperimentRun]:
        """Получает информацию об эксперименте"""
        metadata_path = self.storage_dir / f"{experiment_id}.json"
        if not metadata_path.exists():
            return None
            
        with open(metadata_path) as f:
            data = json.load(f)
            return ExperimentRun(**data)
            
    def list_experiments(self) -> List[ExperimentRun]:
        """Возвращает список всех экспериментов"""
        experiments = []
        for metadata_file in self.storage_dir.glob("*.json"):
            with open(metadata_file) as f:
                data = json.load(f)
                experiments.append(ExperimentRun(**data))
        return experiments
        
    def _save_experiment_metadata(self, experiment: ExperimentRun):
        """Сохраняет метаданные эксперимента"""
        metadata_path = self.storage_dir / f"{experiment.id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(experiment), f, default=str) 