import asyncio
import torch
from pathlib import Path
from src.neural.models.kenga import KengaModel, KengaConfig
from src.neural.trainers.autonomous_trainer import AutonomousTrainer
from src.neural.data.collector import DataCollector
from src.monitoring.metrics import MetricsCollector
from src.monitoring.visualization import MetricsVisualizer
from src.monitoring.performance import PerformanceAnalyzer
from src.monitoring.attention_viz import AttentionVisualizer
from src.monitoring.mlflow_logger import MLflowLogger
from src.monitoring.wandb_logger import WandBLogger
from src.monitoring.metrics_extended import ExtendedMetricsCollector
from src.monitoring.gpu_profiler import GPUProfiler
from src.monitoring.gradient_analyzer import GradientAnalyzer
from src.monitoring.embedding_visualizer import EmbeddingVisualizer
from src.optimization.hyperopt import HyperparameterOptimizer, OptimizationConfig
from src.monitoring.neuron_analyzer import NeuronAnalyzer
from src.monitoring.weight_visualizer import WeightVisualizer
from src.optimization.ray_tuner import RayTuner, RayTuneConfig
from src.monitoring.layer_profiler import LayerProfiler

async def train_model(
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    checkpoint_dir: str = "checkpoints",
    log_interval: int = 10,
    use_mlflow: bool = True,
    use_wandb: bool = True,
    optimize_hyperparams: bool = False,
    use_ray_tune: bool = False
):
    if optimize_hyperparams:
        # Оптимизация гиперпараметров
        opt_config = OptimizationConfig(n_trials=50)
        optimizer = HyperparameterOptimizer(opt_config)
        
        async def objective(trial):
            params = optimizer.suggest_params(trial)
            config = optimizer.create_model_config(params)
            # Инициализация
            model = KengaModel(config)
            trainer = AutonomousTrainer(model)
            collector = DataCollector()
            metrics_collector = MetricsCollector()
            visualizer = MetricsVisualizer()
            
            # Создаем директорию для чекпоинтов
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            
            # Инициализация логгеров
            mlflow_logger = MLflowLogger("kenga_experiment") if use_mlflow else None
            wandb_logger = WandBLogger("kenga", config=config.__dict__) if use_wandb else None
            
            # Инициализация анализаторов
            performance_analyzer = PerformanceAnalyzer()
            attention_visualizer = AttentionVisualizer()
            extended_metrics = ExtendedMetricsCollector()
            
            # Основной цикл обучения
            for epoch in range(epochs):
                total_loss = 0
                num_batches = 0
                
                # Обучение на эпохе
                for _ in range(100):  # 100 батчей на эпоху
                    batch = await collector.get_training_batch(batch_size)
                    loss = await trainer.train_step(batch)
                    total_loss += loss
                    num_batches += 1
                    
                    # Логирование метрик
                    if num_batches % log_interval == 0:
                        avg_loss = total_loss / num_batches
                        metrics_collector.record_training_metrics({
                            'loss': avg_loss,
                            'epoch': epoch,
                            'batch': num_batches
                        })
                        
                        print(f"Epoch {epoch}, Batch {num_batches}, Loss: {avg_loss:.4f}")
                
                # Сохранение чекпоинта
                if epoch % 10 == 0:
                    checkpoint_path = f"{checkpoint_dir}/epoch_{epoch}.pt"
                    trainer.save_checkpoint(checkpoint_path)
                
                # Визуализация метрик
                if epoch % 5 == 0:
                    loss_fig = visualizer.plot_loss_curve()
                    loss_fig.write_html(f"metrics/loss_epoch_{epoch}.html")
                
                # Анализ производительности
                perf_metrics = performance_analyzer.profile_model(model, batch)
                
                # Визуализация внимания
                attention_maps = attention_visualizer.capture_attention(model, batch)
                attention_fig = attention_visualizer.plot_attention_map(
                    attention_maps[0],
                    layer_idx=0,
                    head_idx=0
                )
                
                # Расширенные метрики
                ext_metrics = extended_metrics.collect_metrics(
                    loss,
                    predictions,
                    targets,
                    attention_maps,
                    embeddings
                )
                
                # Логирование в MLflow
                if mlflow_logger:
                    mlflow_logger.log_metrics({
                        'loss': loss,
                        'perplexity': ext_metrics.perplexity,
                        'f1_score': ext_metrics.f1_score,
                        'inference_time': perf_metrics.inference_time
                    }, step=epoch)
                    mlflow_logger.log_figure(attention_fig, f"attention_map_epoch_{epoch}.png")
                
                # Логирование в W&B
                if wandb_logger:
                    wandb_logger.log_metrics({
                        'loss': loss,
                        'perplexity': ext_metrics.perplexity,
                        'f1_score': ext_metrics.f1_score,
                        'gpu_memory': perf_metrics.memory_usage
                    }, step=epoch)
                    wandb_logger.log_attention_map(attention_fig, f"Attention Map - Epoch {epoch}")
                
                # Логирование метрик
                metrics_collector.record_training_metrics({
                    'loss': avg_loss,
                    'epoch': epoch,
                    'batch': num_batches
                })
                
                # Сохранение чекпоинта
                if epoch % 10 == 0:
                    checkpoint_path = f"{checkpoint_dir}/epoch_{epoch}.pt"
                    trainer.save_checkpoint(checkpoint_path)
                
                # Визуализация метрик
                if epoch % 5 == 0:
                    loss_fig = visualizer.plot_loss_curve()
                    loss_fig.write_html(f"metrics/loss_epoch_{epoch}.html")
                
                # Завершение логирования
                if mlflow_logger:
                    mlflow_logger.end_run()
                if wandb_logger:
                    wandb_logger.finish()
                
                return final_loss
            
        best_params = optimizer.optimize(objective)
        config = optimizer.create_model_config(best_params)
    else:
        config = KengaConfig()
    
    # Инициализация
    model = KengaModel(config)
    trainer = AutonomousTrainer(model)
    collector = DataCollector()
    metrics_collector = MetricsCollector()
    visualizer = MetricsVisualizer()
    
    # Создаем директорию для чекпоинтов
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Инициализация логгеров
    mlflow_logger = MLflowLogger("kenga_experiment") if use_mlflow else None
    wandb_logger = WandBLogger("kenga", config=config.__dict__) if use_wandb else None
    
    # Инициализация анализаторов
    performance_analyzer = PerformanceAnalyzer()
    attention_visualizer = AttentionVisualizer()
    extended_metrics = ExtendedMetricsCollector()
    
    # Инициализация новых анализаторов
    gpu_profiler = GPUProfiler()
    gradient_analyzer = GradientAnalyzer()
    embedding_visualizer = EmbeddingVisualizer()
    
    # Инициализация анализаторов
    neuron_analyzer = NeuronAnalyzer()
    weight_visualizer = WeightVisualizer()
    layer_profiler = LayerProfiler()
    
    # Регистрируем хуки для анализа
    neuron_analyzer.register_hooks(model)
    layer_profiler.start_profiling(model)
    
    # Основной цикл обучения
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        # Обучение на эпохе
        for _ in range(100):  # 100 батчей на эпоху
            batch = await collector.get_training_batch(batch_size)
            loss = await trainer.train_step(batch)
            total_loss += loss
            num_batches += 1
            
            # Логирование метрик
            if num_batches % log_interval == 0:
                avg_loss = total_loss / num_batches
                metrics_collector.record_training_metrics({
                    'loss': avg_loss,
                    'epoch': epoch,
                    'batch': num_batches
                })
                
                print(f"Epoch {epoch}, Batch {num_batches}, Loss: {avg_loss:.4f}")
        
        # Сохранение чекпоинта
        if epoch % 10 == 0:
            checkpoint_path = f"{checkpoint_dir}/epoch_{epoch}.pt"
            trainer.save_checkpoint(checkpoint_path)
        
        # Визуализация метрик
        if epoch % 5 == 0:
            loss_fig = visualizer.plot_loss_curve()
            loss_fig.write_html(f"metrics/loss_epoch_{epoch}.html")
        
        # Анализ производительности
        perf_metrics = performance_analyzer.profile_model(model, batch)
        
        # Визуализация внимания
        attention_maps = attention_visualizer.capture_attention(model, batch)
        attention_fig = attention_visualizer.plot_attention_map(
            attention_maps[0],
            layer_idx=0,
            head_idx=0
        )
        
        # Расширенные метрики
        ext_metrics = extended_metrics.collect_metrics(
            loss,
            predictions,
            targets,
            attention_maps,
            embeddings
        )
        
        # Логирование в MLflow
        if mlflow_logger:
            mlflow_logger.log_metrics({
                'loss': loss,
                'perplexity': ext_metrics.perplexity,
                'f1_score': ext_metrics.f1_score,
                'inference_time': perf_metrics.inference_time
            }, step=epoch)
            mlflow_logger.log_figure(attention_fig, f"attention_map_epoch_{epoch}.png")
        
        # Логирование в W&B
        if wandb_logger:
            wandb_logger.log_metrics({
                'loss': loss,
                'perplexity': ext_metrics.perplexity,
                'f1_score': ext_metrics.f1_score,
                'gpu_memory': perf_metrics.memory_usage
            }, step=epoch)
            wandb_logger.log_attention_map(attention_fig, f"Attention Map - Epoch {epoch}")
        
        # Логирование метрик
        metrics_collector.record_training_metrics({
            'loss': avg_loss,
            'epoch': epoch,
            'batch': num_batches
        })
        
        # Сохранение чекпоинта
        if epoch % 10 == 0:
            checkpoint_path = f"{checkpoint_dir}/epoch_{epoch}.pt"
            trainer.save_checkpoint(checkpoint_path)
        
        # Визуализация метрик
        if epoch % 5 == 0:
            loss_fig = visualizer.plot_loss_curve()
            loss_fig.write_html(f"metrics/loss_epoch_{epoch}.html")
        
        # Завершение логирования
        if mlflow_logger:
            mlflow_logger.end_run()
        if wandb_logger:
            wandb_logger.finish()
        
        # Профилирование GPU
        gpu_stats = gpu_profiler.get_gpu_stats()
        if gpu_stats:
            wandb_logger.log_metrics({
                'gpu_memory': gpu_stats.memory_allocated,
                'gpu_utilization': gpu_stats.utilization
            }, step=epoch)
        
        # Анализ градиентов
        grad_stats = gradient_analyzer.compute_grad_stats(model)
        grad_issues = gradient_analyzer.check_vanishing_exploding()
        
        if any(grad_issues.values()):
            print(f"Warning: Gradient issues detected: {grad_issues}")
            
        # Визуализация эмбеддингов
        if epoch % 10 == 0:
            emb_fig = embedding_visualizer.plot_embeddings(embeddings)
            wandb_logger.log_metrics({'embeddings': wandb.Image(emb_fig)}, step=epoch)
        
        # Анализ активаций нейронов
        activation_stats = neuron_analyzer.compute_activation_stats()
        if epoch % 10 == 0:
            act_fig = neuron_analyzer.plot_activation_distributions()
            wandb_logger.log_metrics({'activations': wandb.Image(act_fig)}, step=epoch)
            
        # Анализ весов
        weight_visualizer.capture_weights(model)
        weight_stats = weight_visualizer.compute_weight_stats()
        if epoch % 10 == 0:
            weight_fig = weight_visualizer.plot_weight_distributions()
            wandb_logger.log_metrics({'weights': wandb.Image(weight_fig)}, step=epoch)
            
        # Профилирование слоев
        layer_stats = layer_profiler.get_layer_stats()
        wandb_logger.log_metrics({
            f'layer_time/{name}': stats['mean_forward_time']
            for name, stats in layer_stats.items()
        }, step=epoch)

if __name__ == "__main__":
    asyncio.run(train_model()) 