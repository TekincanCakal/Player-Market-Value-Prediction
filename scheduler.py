import time
import os
import subprocess
from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

def job_function():
    print(f"⏰ [Scheduler] Eğitim görevi tetiklendi: {datetime.now()}")
    
    # 1. Scraping
    print("🕷️ Scraping başlatılıyor...")
    # scraping.py'nin olduğu dizin
    subprocess.run(["python", "scraping.py"])
    
    # 2. Training
    print("🧠 Eğitim başlatılıyor...")
    try:
        # data_clean_and_train.py dosyasını çalıştır.
        # Bu script zaten .env dosyasındaki veya ortam değişkenlerindeki POSTGRES_URL'i okuyacak şekilde ayarlı (veya ayarlayacağız).
        env = os.environ.copy()
        result = subprocess.run(["python", "data_clean_and_train.py"], env=env, text=True, capture_output=True, encoding='utf-8')
        
        print("✅ Eğitim tamamlandı.")
        print("--- Çıktı ---")
        print(result.stdout)
        if result.stderr:
            print("--- Hatalar ---")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")

if __name__ == "__main__":
    scheduler = BlockingScheduler()
    
    # Her saat başı çalışacak şekilde ayarla (minute=0)
    # Test için: her dakika başı çalışsın istersen 'interval', minutes=1 yapabiliriz.
    # Gerçek senaryo: 'cron', hour='*' (Her saat)
    
    print("🚀 Scheduler başlatıldı. Her saat başı (XX:00) çalışacak.")
    print("İlk çalıştırmayı hemen yapıyorum...")
    job_function() # Başlangıçta bir kere çalıştır

    scheduler.add_job(job_function, 'cron', minute=0)
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("🛑 Scheduler durduruldu.")
