import pandas as pd
import numpy as np
import re
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from dotenv import load_dotenv
import os # Added for os.getenv

load_dotenv()

# (Tahmin bloğu dosya sonuna taşındı)
# ==============================================================================
# 1. AYARLAR VE VERİYİ YÜKLEME
# ==============================================================================
# Veri setinizin en son başarılı olduğu dosya adını kullanıyoruz.
FILE_NAME = 'players_data.json' 
CURRENT_YEAR = datetime.date.today().year

# PyTorch için cihazı tanımla (CUDA kontrolü)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 PyTorch cihazı ayarlandı: {device}")

try:
    df = pd.read_json(FILE_NAME)
    print(f"✅ '{FILE_NAME}' başarıyla yüklendi. Başlangıç Boyutu: {df.shape}")
    
    # Sütun isimlerini düzeltme (Scraping çıktısına göre)
    df.rename(columns={'name': 'Name', 'Overall rating': 'Overall'}, inplace=True)

except FileNotFoundError:
    print(f"❌ Hata: '{FILE_NAME}' dosyası bulunamadı.")
    exit()

# ==============================================================================
# 2. ÖZELLİK TEMİZLEME FONKSİYONLARI
# ==============================================================================

def clean_currency(value):
    """'€', 'M' (milyon) ve 'K' (bin) içeren değerleri sayısal formata dönüştürür."""
    if isinstance(value, str):
        if not value: return np.nan
        value = value.replace('€', '').strip()
        if 'M' in value:
            return float(value.replace('M', '')) * 1_000_000
        elif 'K' in value:
            return float(value.replace('K', '')) * 1_000
        try:
            return float(value)
        except ValueError:
            return np.nan
    return value

def clean_height_cm(height_str):
    """Santimetre (cm) değerini çıkarır."""
    if isinstance(height_str, str) and 'cm' in height_str:
        match = re.search(r'(\d+)cm', height_str)
        if match:
            return float(match.group(1))
    return np.nan

def get_contract_end_year(contract_info):
    """Sözleşme bitiş yılını çıkarır."""
    if isinstance(contract_info, str) and '~' in contract_info:
        try:
            return int(contract_info.split('~')[-1].strip())
        except ValueError:
            return np.nan
    return np.nan

# ==============================================================================
# 3. VERİ TEMİZLEME VE DÖNÜŞÜM İŞLEMLERİ
# ==============================================================================

# Para Birimi, Boy ve Sözleşme işlemleri
currency_cols = ['Value', 'Wage', 'Release clause']
for col in currency_cols:
    df[col] = df[col].astype(str).apply(clean_currency)
df['Height_cm'] = df['Height'].apply(clean_height_cm)
df.drop('Height', axis=1, inplace=True)
df[['Team', 'Contract_Info']] = df['Team & Contract'].str.split('\n', expand=True)
df['Contract_End_Year'] = df['Contract_Info'].apply(get_contract_end_year)
df.drop(['Team & Contract', 'Contract_Info'], axis=1, inplace=True)

# İstatistik sütunlarını sayısallaştırma
cols_to_exclude_from_stat_conversion = ['Name', 'Position', 'Team', 'Best position', 'ID', 'Weight']
stat_cols = df.select_dtypes(include=['object']).columns.tolist()
stat_cols = [col for col in stat_cols if col not in cols_to_exclude_from_stat_conversion] 
for col in stat_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce') 

# Gereksiz ve Tekrarlı Sütunları Kaldırma
cols_to_drop = ['', 'ID', 'Growth', 'Defending / Pace', 'Dribbling / Reflexes', 
                'Pace / Diving', 'Passing / Kicking', 'Shooting / Handling', 
                'Base stats', 'International reputation']
existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
df.drop(existing_cols_to_drop, axis=1, inplace=True)
print("✅ Temel temizlik adımları tamamlandı.")


# ==============================================================================
# 4. ÖZELLİK MÜHENDİSLİĞİ (FEATURE ENGINEERING)
# ==============================================================================

df['Age_Squared'] = df['Age'] ** 2
df['Contract_Duration'] = df['Contract_End_Year'] - CURRENT_YEAR
df['Contract_Duration'] = df['Contract_Duration'].apply(lambda x: max(0, x))
print("✅ Yeni özellikler oluşturuldu.")


# ==============================================================================
# 5. EKSİK DEĞER YÖNETİMİ VE TEKİL SÜTUN TEMİZLİĞİ
# ==============================================================================

# Sayısal Doldurma (Medyan) ve Tamamen NaN Sütunları Kaldırma
numerical_cols = df.select_dtypes(include=np.number).columns
for col in numerical_cols:
    if df[col].isnull().all():
        df.drop(col, axis=1, inplace=True)
    else:
        df[col] = df[col].fillna(df[col].median())

# Kategorik Doldurma ('Unknown')
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if col not in ['Name', 'Weight']: 
        df[col] = df[col].fillna('Unknown')

# Kategorik Kodlama (One-Hot Encoding)
cols_to_encode = ['Position', 'Team', 'Best position']
existing_cols_to_encode = [col for col in cols_to_encode if col in df.columns]
# One-Hot Encoding (dtype=int ile sayısal olmasını garanti et)
df_final = pd.get_dummies(df, columns=existing_cols_to_encode, drop_first=True, dtype=int)

# Son Temizlik
if 'Name' in df_final.columns:
    df_final.drop('Name', axis=1, inplace=True)
if 'Weight' in df_final.columns:
    df_final.drop('Weight', axis=1, inplace=True)

print(f"✅ Veri temizliği ve kodlama bitti. Son Boyut: {df_final.shape}")


# ==============================================================================
# 6. TRAIN/TEST VE PYTORCH TENSOR'LARA DÖNÜŞÜM (LOG DÖNÜŞÜMLÜ)
# ==============================================================================

X = df_final.drop('Value', axis=1)
Y = df_final[['Value']] 

# Tüm X verisini float'a çevir (Scaler hatasını önlemek için kritik)
X = X.astype(float)

variance = X.var()
constant_columns = variance[variance == 0].index.tolist()
if constant_columns:
    X.drop(columns=constant_columns, inplace=True)

# Eğitim ve Test Setlerine Bölme
X_train_df, X_test_df, Y_train_df, Y_test_df = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# --- X Özelliklerini Ölçeklendirme ---
# Tüm sütunları ölçekle (Frontend ile 1-1 eşleşme için şart)
x_scaler = StandardScaler()
X_train_df[:] = x_scaler.fit_transform(X_train_df)
X_test_df[:] = x_scaler.transform(X_test_df)
print("✅ X Özellikleri ölçeklendirildi.")


# --- Y Hedef Değişkeni Logaritmik Dönüşüm ve Ölçeklendirme ---
# Log dönüşümü (np.log1p) ile dağılımı normalize et ve negatif tahminleri engelle
y_scaler = StandardScaler()
Y_train_log = np.log1p(Y_train_df) 
Y_test_log = np.log1p(Y_test_df)

Y_train_scaled = y_scaler.fit_transform(Y_train_log)
Y_test_scaled = y_scaler.transform(Y_test_log)
print("✅ Y Hedef Değişkeni Logaritmik Dönüşüm ve Ölçekleme yapıldı.")


def enforce_float(df):
    """DataFrame'deki tüm sütunları float'a zorlar."""
    df = df.apply(pd.to_numeric, errors='coerce')
    return df.fillna(df.mean())

X_train_df = enforce_float(X_train_df.copy())
X_test_df = enforce_float(X_test_df.copy())


# NumPy'a ve sonra PyTorch Tensor'lara dönüştürme
X_train_tensor = torch.tensor(X_train_df.values.astype(np.float32)).to(device)
Y_train_tensor = torch.tensor(Y_train_scaled.astype(np.float32)).to(device)
X_test_tensor = torch.tensor(X_test_df.values.astype(np.float32)).to(device)
Y_test_tensor = torch.tensor(Y_test_scaled.astype(np.float32)).to(device)

# DataLoader oluşturma
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) # Batch size artırıldı

print("✅ Veriler PyTorch Tensor'lara dönüştürüldü ve DataLoader hazırlandı.")


# ==============================================================================
# 7. PYTORCH MODEL TANIMI (DROPOUT DÜZELTİLDİ)
# ==============================================================================

class PlayerValueModel(nn.Module):
    def __init__(self, input_size):
        super(PlayerValueModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256) 
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        # Dropout %30'dan %10'a düşürüldü
        self.dropout = nn.Dropout(0.1) 
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Model başlatma
input_size = X_train_tensor.shape[1]
model = PlayerValueModel(input_size).to(device)

# Kayıp Fonksiyonu ve Optimizasyon
criterion = nn.MSELoss() 
# Öğrenme Oranı 0.0005'ten 0.0001'e düşürüldü (Kararlılık için)
optimizer = optim.Adam(model.parameters(), lr=0.0001) 

print(f"\n🧠 Yapay Sinir Ağı ({input_size} Giriş) Modeli Hazırlandı.")


# ==============================================================================
# 8. MODEL EĞİTİM DÖNGÜSÜ
# ==============================================================================

NUM_EPOCHS = 1200 # Epoch sayısı biraz daha artırıldı

print(f"\n⏳ Model {NUM_EPOCHS} epoch boyunca eğitiliyor...")

for epoch in range(NUM_EPOCHS):
    model.train() 
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.6f}')

print("✅ Eğitim tamamlandı.")


# ==============================================================================
# 9. MODEL DEĞERLENDİRME (LOG DÖNÜŞÜMÜ GERİ ALINDI)
# ==============================================================================

model.eval() # Değerlendirme moduna geç
with torch.no_grad():
    Y_pred_tensor = model(X_test_tensor)
    
    # Ölçeklenmiş tahminleri CPU'ya taşı
    Y_pred_scaled = Y_pred_tensor.cpu().numpy()
    Y_test_scaled = Y_test_tensor.cpu().numpy()

# 1. Adım: Ölçeklemeyi orijinal logaritmik aralığa geri al
Y_pred_unscaled_log = y_scaler.inverse_transform(Y_pred_scaled)
Y_test_unscaled_log = y_scaler.inverse_transform(Y_test_scaled)

# 2. Adım: Log dönüşümünü (np.expm1) geri alarak orijinal Euro değerine dön
Y_pred = np.expm1(Y_pred_unscaled_log)
Y_test = np.expm1(Y_test_unscaled_log)


# Performansı Değerlendirme (MAE)
mae = mean_absolute_error(Y_test, Y_pred)

print("\n" + "="*50)
print("🚀 OPTİMİZE PYTORCH MODEL DEĞERLENDİRME SONUÇLARI")
print(f"Test Seti Üzerindeki Ortalama Mutlak Hata (MAE): €{mae:,.2f}")
print("==================================================")

# Örnek tahminleri göster
sample_predictions = pd.DataFrame({
    'Gerçek Değer': Y_test.flatten(), 
    'Tahmin Edilen Değer': Y_pred.flatten()
})
sample_predictions['Hata'] = abs(sample_predictions['Gerçek Değer'] - sample_predictions['Tahmin Edilen Değer'])
print("\n📝 Sinir Ağının Örnek Tahminleri (İlk 5):")
print(sample_predictions.head().sort_values(by='Gerçek Değer', ascending=False))


# ==============================================================================
# 10. VERCEL ENTEGRASYONU: DB LOGLAMA VE JSON EXPORT
# ==============================================================================

import json
import os
import psycopg2
from datetime import datetime

def export_model_to_json(model, x_scaler, y_scaler, input_columns):
    """
    Modelin ağırlıklarını (Weights) ve Scaler parametrelerini 
    JavaScript tarafında kullanılmak üzere JSON formatına çevirir.
    """
    state_dict = model.state_dict()
    
    # PyTorch Tensorlarını listeye çevir
    weights = {}
    for key, value in state_dict.items():
        weights[key] = value.cpu().numpy().tolist()
        
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "model_weights": weights,
        "x_scaler_mean": x_scaler.mean_.tolist(),
        "x_scaler_scale": x_scaler.scale_.tolist(),
        "y_scaler_mean": y_scaler.mean_.tolist(),
        "y_scaler_scale": y_scaler.scale_.tolist(),
        "input_columns": input_columns.tolist(),
        "mae_score": float(mae)
    }
    
    return json.dumps(export_data)

def save_to_postgres(json_data, mae_score, loss):
    """
    Eğitim sonuçlarını ve model JSON'ını Vercel Postgres'e kaydeder.
    """
    db_url = os.environ.get("POSTGRES_URL")
    if not db_url:
        print("⚠️ POSTGRES_URL bulunamadı, DB'ye kayıt yapılmadı. (Sadece yerel test mi?)")
        # Yerel test için dosyaya yaz
        with open("model_export.json", "w", encoding="utf-8") as f:
            f.write(json_data)
        print("✅ Model 'model_export.json' olarak yerel diske kaydedildi.")
        return

    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        
        # Tabloları oluştur (Yoksa)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS training_logs (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                mae_score FLOAT,
                final_loss FLOAT,
                model_json JSONB
            );
        """)
        
        # Veriyi ekle
        cur.execute("""
            INSERT INTO training_logs (mae_score, final_loss, model_json)
            VALUES (%s, %s, %s)
        """, (mae_score, loss, json_data))
        
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Eğitim sonuçları ve Model JSON Vercel Postgres'e başarıyla kaydedildi!")
        
    except Exception as e:
        print(f"❌ DB Kayıt Hatası: {e}")

# JSON Verisini Hazırla
print("🔄 Model JSON formatına dönüştürülüyor...")
# Input columns sıralamasını kaydetmek önemli (Frontend'de aynı sırayla özellik vektörü oluşturmak için)
input_cols = X.columns 

model_json = export_model_to_json(model, x_scaler, y_scaler, input_cols)

# DB'ye Kaydet
# Not: 'loss' değişkeni eğitim döngüsünden gelen son loss değeridir.
final_loss = loss.item() if 'loss' in locals() else 0.0
save_to_postgres(model_json, mae, final_loss)

# ==============================================================================
# 11. TAHMİNLERİ OLUŞTUR VE VERİTABANINA KAYDET (YENİ ÖZELLİK)
# ==============================================================================
print("Tüm oyuncular için tahminler oluşturuluyor...")

# Tüm veri seti üzerinde tahmin yap (X_scaled kullanacağız)
# X zaten float olarak ayarlanmıştı.
# Tüm veriyi tekrar scale et (Eğitimde fit edilen scaler ile)

X_all_scaled = x_scaler.transform(X)
X_tensor_all = torch.tensor(X_all_scaled, dtype=torch.float32).to(device)

model.eval()
with torch.no_grad():
    predictions_log = model(X_tensor_all).cpu().numpy()

# Ters Log Dönüşümü (Gerçek Değer Tahmini)
# Önce Scaler'ın tersini al (Z-score -> Log Value)
predictions_rescaled = y_scaler.inverse_transform(predictions_log)
# Sonra Log'un tersini al (Log Value -> Euro)
predictions_eur = np.expm1(predictions_rescaled)

# Sonuçları DataFrame'e ekle
# Orjinal DataFrame'den isim ve diğer bilgileri alabilmek için, indexleri kullanmamız lazım.
# df_final üzerinde çalışmıştık ama isimleri atmıştık.
# En başta 'df' değişkeninde orjinal veri duruyor.

# df ile X aynı indexe sahip olmalı.
df_results = df.copy() 
# Temizlik sırasında satır atıldıysa indexler kaymış olabilir, ancak biz dropna yaptık.
# En garantisi: df_final'in indexlerini kullanmak.
df_results = df_results.loc[df_final.index]

df_results['Predicted_Value'] = predictions_eur
df_results['Value_Diff'] = df_results['Predicted_Value'] - df_results['Value']

# Kaydedilecek sütunları seç
# 'Name' sütunu df'de mevcut.
columns_to_save = ['Name', 'Age', 'Overall', 'Potential', 'Value', 'Predicted_Value', 'Value_Diff']
# Eğer 'Nationality' veya 'Team' gibi ek bilgiler istersek buraya ekleyebiliriz.

# Sadece bu sütunları al
df_save = df_results[columns_to_save].copy()

# Değerleri tam sayıya yuvarla
df_save['Value'] = df_save['Value'].astype(int)
df_save['Predicted_Value'] = df_save['Predicted_Value'].astype(int)
df_save['Value_Diff'] = df_save['Value_Diff'].astype(int)

# Veritabanına Kaydet (players tablosu)
try:
    from sqlalchemy import create_engine, text
    
    # SQLAlchemy motoru oluştur (Postgres)
    # POSTGRES_URL: postgresql://admin:admin@localhost:5432/football_db
    # SQLAlchemy için 'postgresql+psycopg2://...' formatı gerekebilir veya 'postgresql://' yeterli.
    db_url = os.getenv("POSTGRES_URL").replace("postgresql://", "postgresql+psycopg2://")
    engine = create_engine(db_url)
    
    # Tabloyu (varsa) üzerine yaz
    df_save.to_sql('players', engine, if_exists='replace', index=False)
    
    # Primary Key ekle (Opsiyonel ama iyi olur)
    # with engine.connect() as con:
    #     con.execute(text('ALTER TABLE players ADD PRIMARY KEY ("Name");'))
        
    print(f"✅ {len(df_save)} oyuncu verisi 'players' tablosuna kaydedildi.")
    
except Exception as e:
    print(f"❌ Oyuncu verileri kaydedilemedi: {e}")