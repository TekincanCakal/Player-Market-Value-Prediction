"use client";

import { useState, useEffect, useMemo } from 'react';

export default function PredictionPlayground({ modelJson }: { modelJson: any }) {
    const [inputs, setInputs] = useState<any>({
        Age: 25,
        Overall: 75,
        Potential: 80,
        Wage: 50000,
        Contract_Duration: 3
    });
    const [prediction, setPrediction] = useState<number | null>(null);

    // One-Hot Sütunlarını Kategorilere Ayır
    const categories = useMemo(() => {
        if (!modelJson?.input_columns) return {};
        const cols = modelJson.input_columns as string[];
        const teams = cols.filter(c => c.startsWith('Team_')).map(c => c.replace('Team_', ''));
        const positions = cols.filter(c => c.startsWith('Position_')).map(c => c.replace('Position_', ''));

        return { teams, positions };
    }, [modelJson]);

    const [selectedTeam, setSelectedTeam] = useState(categories.teams?.[0] || '');
    const [selectedPos, setSelectedPos] = useState(categories.positions?.[0] || '');

    const handlePredict = () => {
        if (!modelJson) return;

        // 1. Girdi Vektörünü Oluştur (Hepsi 0)
        const inputCols = modelJson.input_columns;
        let vector = new Array(inputCols.length).fill(0);

        // 2. Sayısal Değerleri Yerleştir
        // Bu feature'ların isimlerini eğitimden biliyoruz:
        // ['Age', 'Overall', 'Potential', 'wage', 'age_squared' ...]
        // Basitlik için sadece ana featureları update edelim, diğerlerini ortalama (0 scaled) bırakalım.

        // NOT: Gerçek modelde tüm sütunların indexini bulup değer atamalıyız.
        // Burada basitleştirilmiş mantık: Input columns listesinde adı geçen feature'a değeri ata.

        const numericMap: any = {
            'Age': inputs.Age,
            'Age_Squared': inputs.Age * inputs.Age,
            'Overall': inputs.Overall,
            'Potential': inputs.Potential,
            'Wage': inputs.Wage,
            'Contract_Duration': inputs.Contract_Duration
        };

        // Kategorikler
        numericMap[`Team_${selectedTeam}`] = 1;
        numericMap[`Position_${selectedPos}`] = 1;

        // Vektörü Doldur
        for (let i = 0; i < inputCols.length; i++) {
            const colName = inputCols[i];

            // Eğer numerik haritamızda varsa
            if (numericMap[colName] !== undefined) {
                // Scaling öncesi ham değer
                const rawVal = numericMap[colName];

                // Ölçekle: (Value - Mean) / Scale
                // Dikkat: x_scaler_mean bir liste, i. eleman bu sütunun mean'i
                const mean = modelJson.x_scaler_mean[i];
                const scale = modelJson.x_scaler_scale[i];

                vector[i] = (rawVal - mean) / scale;
            } else {
                // Bilinmeyen veya girilmeyen değerler için Mean (0'a scale edilmiş) kullan
                // scaler transform: (x - u) / s. Eğer x = u ise sonuç 0.
                // Biz 0 gönderirsek model ortalama bir oyuncu gibi davranır.
                vector[i] = 0;
            }
        }

        // 3. İleri Yayılım (Forward Pass)
        // Layer 1
        let h1 = denseLayer(vector, modelJson.model_weights['fc1.weight'], modelJson.model_weights['fc1.bias']);
        h1 = relu(h1);

        // Layer 2
        let h2 = denseLayer(h1, modelJson.model_weights['fc2.weight'], modelJson.model_weights['fc2.bias']);
        h2 = relu(h2);

        // Layer 3
        let h3 = denseLayer(h2, modelJson.model_weights['fc3.weight'], modelJson.model_weights['fc3.bias']);
        h3 = relu(h3);

        // Output
        let out = denseLayer(h3, modelJson.model_weights['fc4.weight'], modelJson.model_weights['fc4.bias']);

        // 4. Inverse Scale & Inverse Log
        // out bir dizi [val], tek çıktı var
        const resultScaled = out[0];
        const resultLog = (resultScaled * modelJson.y_scaler_scale[0]) + modelJson.y_scaler_mean[0];
        const finalValue = Math.expm1(resultLog); // Log dönüşümünü geri al

        setPrediction(finalValue);
    };

    // Math Utils
    function denseLayer(input: number[], weights: number[][], bias: number[]) {
        // weights shape: [out_features, in_features]
        // result[i] = dot(weights[i], input) + bias[i]
        const output = [];
        for (let i = 0; i < weights.length; i++) {
            let sum = 0;
            const w_row = weights[i];
            for (let j = 0; j < w_row.length; j++) {
                sum += w_row[j] * input[j];
            }
            output.push(sum + bias[i]);
        }
        return output;
    }

    function relu(arr: number[]) {
        return arr.map(x => Math.max(0, x));
    }

    if (!modelJson) return <div>Model yükleniyor...</div>;

    return (
        <div className="bg-gray-800 p-6 rounded-xl border border-gray-700 shadow-lg">
            <h2 className="text-xl font-bold mb-4 text-purple-400">🔮 Canlı Fiyat Tahmincisi</h2>

            <div className="grid grid-cols-2 gap-4">
                <div>
                    <label className="block text-xs text-gray-400">Yaş</label>
                    <input type="number" value={inputs.Age} onChange={e => setInputs({ ...inputs, Age: +e.target.value })} className="w-full bg-gray-900 border border-gray-700 rounded p-2 text-white" />
                </div>
                <div>
                    <label className="block text-xs text-gray-400">Overall (Reyting)</label>
                    <input type="number" value={inputs.Overall} onChange={e => setInputs({ ...inputs, Overall: +e.target.value })} className="w-full bg-gray-900 border border-gray-700 rounded p-2 text-white" />
                </div>
                <div>
                    <label className="block text-xs text-gray-400">Potential</label>
                    <input type="number" value={inputs.Potential} onChange={e => setInputs({ ...inputs, Potential: +e.target.value })} className="w-full bg-gray-900 border border-gray-700 rounded p-2 text-white" />
                </div>
                <div>
                    <label className="block text-xs text-gray-400">Maaş (€)</label>
                    <input type="number" value={inputs.Wage} step={1000} onChange={e => setInputs({ ...inputs, Wage: +e.target.value })} className="w-full bg-gray-900 border border-gray-700 rounded p-2 text-white" />
                </div>
                <div className="col-span-2">
                    <label className="block text-xs text-gray-400">Takım</label>
                    <select value={selectedTeam} onChange={e => setSelectedTeam(e.target.value)} className="w-full bg-gray-900 border border-gray-700 rounded p-2 text-white">
                        {categories.teams?.slice(0, 100).map((t: string) => <option key={t} value={t}>{t}</option>)}
                    </select>
                </div>
            </div>

            <button onClick={handlePredict} className="w-full mt-4 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-500 hover:to-blue-500 text-white font-bold py-2 rounded transition-all">
                HESAPLA
            </button>

            {prediction !== null && (
                <div className="mt-4 text-center animate-pulse">
                    <div className="text-gray-400 text-sm">Tahmini Piyasa Değeri</div>
                    <div className="text-3xl font-extrabold text-green-400">
                        €{prediction.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                    </div>
                </div>
            )}
        </div>
    );
}
