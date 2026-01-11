"use client";

import { useEffect, useState } from "react";
import TrainingHistoryChart from "@/components/TrainingHistoryChart";
import PredictionPlayground from "@/components/PredictionPlayground";
import { Activity, Brain, Database, Timer } from "lucide-react";

export default function Home() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  const fetchData = async () => {
    try {
      const res = await fetch("/api/training-logs");
      const json = await res.json();
      setData(json);
      setLoading(false);
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    fetchData();
    // 60 saniyede bir otomatik yenile
    const interval = setInterval(fetchData, 60000);
    return () => clearInterval(interval);
  }, []);

  if (loading) return <div className="flex h-screen items-center justify-center text-purple-500">Sistem Yükleniyor...</div>;

  const latest = data?.latest || {};
  const history = data?.history || [];
  const modelJson = latest.model_json || null;

  return (
    <main className="min-h-screen p-8 max-w-7xl mx-auto space-y-8">
      {/* Header */}
      <header className="flex justify-between items-center mb-12">
        <div>
          <h1 className="text-4xl font-extrabold bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-pink-600 neon-text">
            AI Scout Master
          </h1>
          <p className="text-gray-400 mt-2">Otonom Oyuncu Değerleme Sistemi</p>
        </div>
        <div className="flex items-center gap-2 text-xs text-gray-500 glass px-4 py-2 rounded-full">
          <span className="relative flex h-3 w-3">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
          </span>
          Sistem Aktif • Son Güncelleme: {new Date(latest.timestamp || Date.now()).toLocaleTimeString()}
        </div>
      </header>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <StatCard
          icon={<Brain className="text-purple-400" />}
          title="Model Durumu"
          value="Eğitildi"
          sub="MLP / PyTorch"
        />
        <StatCard
          icon={<Database className="text-blue-400" />}
          title="Eğitim Verisi"
          value="11,850+"
          sub="Oyuncu Kaydı"
        />
        <StatCard
          icon={<Activity className="text-green-400" />}
          title="Başarım (MAE)"
          value={`€${(latest.mae_score || 0).toLocaleString()}`}
          sub="Ortalama Hata"
        />
        <StatCard
          icon={<Timer className="text-yellow-400" />}
          title="Sonraki Eğitim"
          value="1 Saat Sonra"
          sub="Otomatik Döngü"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Sol Kolon: Grafik */}
        <div className="lg:col-span-2 glass-card h-[400px]">
          <TrainingHistoryChart history={history} />
        </div>

        {/* Sağ Kolon: Tahmin */}
        <div className="glass-card">
          <PredictionPlayground modelJson={modelJson} />
        </div>
      </div>
    </main>
  );
}

function StatCard({ icon, title, value, sub }: any) {
  return (
    <div className="glass-card flex items-center gap-4">
      <div className="p-3 glass rounded-lg">{icon}</div>
      <div>
        <div className="text-gray-400 text-xs uppercase tracking-wider">{title}</div>
        <div className="text-xl font-bold text-white">{value}</div>
        <div className="text-gray-500 text-xs">{sub}</div>
      </div>
    </div>
  )
}
