"use client";

import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
);

export const options = {
    responsive: true,
    plugins: {
        legend: {
            position: 'top' as const,
            labels: {
                color: 'white'
            }
        },
        title: {
            display: true,
            text: 'Model Gelişim Grafiği (MAE & Loss)',
            color: 'white'
        },
    },
    scales: {
        y: {
            ticks: { color: '#ccc' },
            grid: { color: '#333' }
        },
        x: {
            ticks: { color: '#ccc' },
            grid: { color: '#333' }
        }
    }
};

export default function TrainingHistoryChart({ history }: { history: any[] }) {
    if (!history || history.length === 0) return <div className="text-gray-400">Veri yok...</div>;

    const data = {
        labels: history.map(h => new Date(h.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })),
        datasets: [
            {
                label: 'Hata Payı (MAE €)',
                data: history.map(h => h.mae_score),
                borderColor: 'rgb(255, 99, 132)',
                backgroundColor: 'rgba(255, 99, 132, 0.5)',
                yAxisID: 'y',
            },
            {
                label: 'Loss',
                data: history.map(h => h.final_loss),
                borderColor: 'rgb(53, 162, 235)',
                backgroundColor: 'rgba(53, 162, 235, 0.5)',
                yAxisID: 'y1',
            },
        ],
    };

    return <Line options={options} data={data} />;
}
