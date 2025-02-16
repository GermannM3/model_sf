// Скрипт для инициализации и обновления графика с использованием Chart.js
document.addEventListener('DOMContentLoaded', async function() {
    const canvas = document.getElementById('trainingChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    // Функция для получения метрик через API /metrics
    async function fetchMetrics() {
        try {
            const response = await fetch('/metrics');
            const data = await response.json();
            return data;
        } catch (err) {
            console.error("Ошибка получения метрик:", err);
            return null;
        }
    }

    // Инициализируем график с начальными данными
    const initialMetrics = await fetchMetrics();
    const initialLabels = initialMetrics ? initialMetrics.labels : [];
    const initialLoss = initialMetrics ? initialMetrics.loss : [];
    
    const chartData = {
        labels: initialLabels,
        datasets: [{
            label: 'Loss',
            data: initialLoss,
            fill: false,
            borderColor: 'rgba(37, 99, 235, 1)',
            backgroundColor: 'rgba(37, 99, 235, 0.5)',
            tension: 0.1
        }]
    };

    const config = {
        type: 'line',
        data: chartData,
        options: {
            scales: {
                y: { beginAtZero: true }
            }
        }
    };

    const trainingChart = new Chart(ctx, config);

    // Функция для обновления графика – вызывается каждые 10 секунд
    setInterval(async () => {
        const newMetrics = await fetchMetrics();
        if (newMetrics && newMetrics.labels && newMetrics.loss) {
            trainingChart.data.labels = newMetrics.labels;
            trainingChart.data.datasets[0].data = newMetrics.loss;
            trainingChart.update();
        }
    }, 10000);  // обновление каждые 10 секунд
}); 