import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# .env'den bilgileri al veya varsayılanları kullan
DB_URL = os.getenv("POSTGRES_URL", "postgresql://admin:admin@localhost:5432/football_db")

def init_db():
    print(f"🔌 Veritabanına bağlanılıyor: {DB_URL}")
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        
        print("🛠️ Tablolar oluşturuluyor...")
        
        # Training Logs Tablosu
        cur.execute("""
            CREATE TABLE IF NOT EXISTS training_logs (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                mae_score FLOAT,
                final_loss FLOAT,
                model_json JSONB
            );
        """)
        
        print("✅ 'training_logs' tablosu hazır.")
        
        conn.commit()
        cur.close()
        conn.close()
        print("🎉 Veritabanı kurulumu tamamlandı!")
        
    except Exception as e:
        print(f"❌ Hata: {e}")

if __name__ == "__main__":
    init_db()
