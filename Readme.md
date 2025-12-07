# ⚽ Oyuncu Piyasa Değeri Tahmini (PyTorch Derin Öğrenme)

Bu proje, futbolcu istatistiklerini (EA FC/FIFA verileri) kullanarak oyuncuların güncel piyasa değerlerini yüksek doğrulukla tahmin etmek amacıyla PyTorch ile geliştirilmiş bir **Çok Katmanlı Algılayıcı (MLP)** sinir ağı modelini sunar.

## 🚀 Proje Performans Özeti

Modelimiz, agresif özellik mühendisliği ve dikkatli hiperparametre ayarı (Logaritmik Dönüşüm, Düşük Öğrenme Oranı) sayesinde kararlı ve güçlü sonuçlar elde etmiştir.

| Metrik | Sonuç | Yorum |
| :--- | :--- | :--- |
| **Model** | Yapay Sinir Ağı (MLP) | Yüksek boyutlu verilerde etkin. |
| **Veri Seti Boyutu** | 11,880 Oyuncu Kaydı | Eğitim için kullanılan veri miktarı. |
| **Girdi Özellik Sayısı** | 1029 | One-Hot Encoding sonrası özellik sayısı. |
| **Test Seti MAE** | **€475,925.00** | Tahminlerin gerçek değerden ortalama sapmasıdır. (Optimizasyon sonrası en kararlı sonuç). |
| **Cihaz Kullanımı** | NVIDIA CUDA (GPU) | Eğitim, yüksek hız için GPU üzerinde yapılmıştır. |

---

## 🛠️ Kurulum ve Çalıştırma

### 1. Kütüphane Kurulumu

Projenin çalışması için temel veri bilimi ve CUDA destekli PyTorch kütüphanelerinin yüklü olması gerekmektedir.

```bash
(venv) python --version
Python 3.11.9
# Temel kütüphaneler
pip install pandas numpy scikit-learn

# CUDA destekli PyTorch kurulumu
# (Sürümünüzü kontrol etmeyi unutmayın)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
````

### 2\. Çalıştırma

Proje dosyasını (`data_clean_and_train.py`) terminalde çalıştırmak için:

```bash
python data_clean_and_train.py
```

-----

## ⚙️ Veri Ön İşleme ve Özellik Mühendisliği

Veri setindeki 65 ham sütun, aşağıdaki adımlarla 1029 özellikli sayısal vektöre dönüştürülmüştür:

1.  **Temel Temizlik:** Para birimi (`Value`, `Wage`, `Release clause`) ve `Height` (boy) bilgileri sayısal formata dönüştürülmüştür.
2.  **Özellik Türetme:** `Age_Squared` ve `Contract_Duration` gibi modelin daha iyi öğrenmesini sağlayan yeni özellikler oluşturulmuştur.
3.  **Kategorik Kodlama (One-Hot Encoding):** `Position`, `Team` ve `Best position` sütunları One-Hot Encoding ile sayısal vektörlere çevrilerek girdi boyutu büyük oranda artırılmıştır.
4.  **Girdi Ölçekleme (X):** Tüm girdi özellikleri, eğitime uygun hale getirmek için **StandardScaler** ile ölçeklenmiştir.
5.  **Hedef Ölçekleme (Y):** Negatif tahminleri önlemek ve dağılımı normale yaklaştırmak için hedef değişken (`Value`) önce **Logaritmik Dönüşüm** (`np.log1p`), ardından **StandardScaler** ile ölçeklenmiştir.

-----

## 🧠 Yapay Sinir Ağı (MLP) Mimarisi

Kullanılan model, 1029 girdiyi işleyebilen, 3 gizli katmanlı bir MLP'dir. Yüksek özellik sayısından kaynaklanabilecek aşırı öğrenmeyi engellemek için **Dropout** mekanizması entegre edilmiştir.

### Model Mimarisi

| Katman | Nöron Sayısı | Aktivasyon / İşlem |
| :--- | :--- | :--- |
| **Giriş** | 1029 | - |
| **Gizli 1** | 256 | ReLU + **Dropout (0.1)** |
| **Gizli 2** | 128 | ReLU + **Dropout (0.1)** |
| **Gizli 3** | 64 | ReLU |
| **Çıkış** | 1 | Lineer |

### Eğitim Parametreleri

| Parametre | Değer |
| :--- | :--- |
| **Kayıp Fonksiyonu** | `nn.MSELoss()` |
| **Optimizasyon** | `optim.Adam` |
| **Öğrenme Oranı** | `lr=0.0001` |
| **Epoch Sayısı** | 1200 |


### Eğitim Çıktıları
CUDA GeForce RTX 5090 NVIDIA Blackwell

⏳ Model 1200 epoch boyunca eğitiliyor...
Epoch [50/1200], Loss: 0.005253
Epoch [100/1200], Loss: 0.002578
Epoch [150/1200], Loss: 0.004656
Epoch [200/1200], Loss: 0.004160
Epoch [250/1200], Loss: 0.001515
Epoch [300/1200], Loss: 0.004120
Epoch [350/1200], Loss: 0.001913
Epoch [400/1200], Loss: 0.001689
Epoch [450/1200], Loss: 0.001307
Epoch [500/1200], Loss: 0.000845
Epoch [550/1200], Loss: 0.003875
Epoch [600/1200], Loss: 0.001092
Epoch [650/1200], Loss: 0.006634
Epoch [700/1200], Loss: 0.000506
Epoch [750/1200], Loss: 0.001713
Epoch [800/1200], Loss: 0.001089
Epoch [850/1200], Loss: 0.000593
Epoch [900/1200], Loss: 0.000227
Epoch [950/1200], Loss: 0.000537
Epoch [1000/1200], Loss: 0.000404
Epoch [1050/1200], Loss: 0.000369
Epoch [1100/1200], Loss: 0.000547
Epoch [1150/1200], Loss: 0.001922
Epoch [1200/1200], Loss: 0.000518
✅ Eğitim tamamlandı.

==================================================
🚀 OPTİMİZE PYTORCH MODEL DEĞERLENDİRME SONUÇLARI
Test Seti Üzerindeki Ortalama Mutlak Hata (MAE): €447,587.75
==================================================

📝 Sinir Ağının Örnek Tahminleri (İlk 5):
   Gerçek Değer  Tahmin Edilen Değer          Hata
4  6.500001e+06         4.411458e+06  2.088543e+06
1  2.400000e+06         2.384980e+06  1.502000e+04
3  1.400000e+06         1.720836e+06  3.208361e+05
0  5.250002e+05         5.635123e+05  3.851206e+04
2  8.999996e+04         1.061797e+05  1.617977e+04