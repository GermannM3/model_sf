import plotly.express as px
from src.monitoring.metrics import MetricsCollector
from src.monitoring.visualization import MetricsVisualizer

async def main():
    collector = MetricsCollector()
    visualizer = MetricsVisualizer()
    
    # Получаем метрики
    metrics = collector.get_all_metrics()
    
    # Создаем визуализации
    loss_fig = visualizer.plot_loss_curve()
    accuracy_fig = visualizer.plot_accuracy()
    
    # Сохраняем графики
    loss_fig.write_html("metrics/loss_curve.html")
    accuracy_fig.write_html("metrics/accuracy.html")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 