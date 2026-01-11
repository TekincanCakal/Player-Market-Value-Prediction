'use client';

import { useEffect, useState } from 'react';

// Format currency function
const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-DE', { style: 'currency', currency: 'EUR', maximumFractionDigits: 0 }).format(value);
};

export default function PlayersPage() {
    const [players, setPlayers] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetch('/api/players')
            .then((res) => res.json())
            .then((data) => {
                if (Array.isArray(data)) {
                    setPlayers(data);
                } else {
                    console.error("API response is not an array:", data);
                    // Hata durumunda boş liste dönebiliriz veya UI'da hata gösterebiliriz
                    setPlayers([]);
                }
                setLoading(false);
            })
            .catch((err) => {
                console.error(err);
                setLoading(false);
            });
    }, []);

    if (loading) return <div className="p-8 text-white">Yükleniyor...</div>;

    return (
        <div className="p-8 min-h-screen">
            <h1 className="text-3xl font-bold mb-6 text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">
                Oyuncu Veritabanı ve Tahminler
            </h1>

            <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl overflow-hidden shadow-2xl">
                <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse">
                        <thead>
                            <tr className="bg-white/5 text-gray-300 border-b border-white/10">
                                <th className="p-4 font-semibold text-sm">İsim</th>
                                <th className="p-4 font-semibold text-sm">Yaş</th>
                                <th className="p-4 font-semibold text-sm">Reyting</th>
                                <th className="p-4 font-semibold text-sm">Potansiyel</th>
                                <th className="p-4 font-semibold text-sm text-right">Gerçek Değer</th>
                                <th className="p-4 font-semibold text-sm text-right">Model Tahmini</th>
                                <th className="p-4 font-semibold text-sm text-right">Fark</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-white/5 text-gray-200">
                            {players.map((player, idx) => {
                                const diff = player.Value_Diff;
                                const diffColor = diff > 0
                                    ? 'text-emerald-400'
                                    : diff < 0
                                        ? 'text-rose-400'
                                        : 'text-gray-400';

                                return (
                                    <tr key={idx} className="hover:bg-white/5 transition-colors">
                                        <td className="p-4 font-medium">{player.Name}</td>
                                        <td className="p-4">{player.Age}</td>
                                        <td className="p-4">
                                            <span className={`px-2 py-1 rounded-md text-xs font-bold ${player.Overall >= 85 ? 'bg-emerald-500/20 text-emerald-400' : 'bg-blue-500/20 text-blue-400'}`}>
                                                {player.Overall}
                                            </span>
                                        </td>
                                        <td className="p-4 text-gray-400">{player.Potential}</td>
                                        <td className="p-4 text-right font-mono text-blue-300">{formatCurrency(player.Value)}</td>
                                        <td className="p-4 text-right font-mono text-purple-300">{formatCurrency(player.Predicted_Value)}</td>
                                        <td className={`p-4 text-right font-mono font-bold ${diffColor}`}>
                                            {diff > 0 ? '+' : ''}{formatCurrency(diff)}
                                        </td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}
