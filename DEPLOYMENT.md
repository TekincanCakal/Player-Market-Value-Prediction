# 🚀 Dağıtım (Deployment) Rehberi

Projeniz Vercel (Frontend & DB) ve GitHub Actions (Otomatik Eğitim) üzerinde çalışacak şekilde hazırlandı. Canlıya almak için aşağıdaki adımları izleyin.

## 1. GitHub'a Gönderin
Kodları GitHub deponuza yükleyin:
```bash
git add .
git commit -m "Full stack otomasyon sistemi eklendi"
git push origin main
```

## 2. Vercel Kurulumu (Frontend & DB)
1.  [Vercel.com](https://vercel.com/new)'a gidin ve bu projeyi (`dashboard` klasörü değil, **root** klasörü seçerek import edebilirsin, ancak `dashboard`'ı root olarak ayarlaman gerekir. Daha kolayı: `dashboard` klasörünü ayrı proje olarak değil, monorepo mantığıyla tek repoda tutmaktır.)
    *   **ÖNEMLİ:** Vercel'de projeyi import ederken **"Root Directory"** kısmını `dashboard` olarak seçin! (Edit'e basıp `dashboard` seçin).
    *   **Framework Preset:** Next.js (Otomatik seçilir).
2.  Projeyi Deploy edin.
3.  Deploy bittikten sonra Vercel panelinde **"Storage"** sekmesine gidin.
4.  **"Create Database"** -> **"Postgres"** seçin ve oluşturun (Ücretsiz plan).
5.  Veritabanı oluştuktan sonra sol menüden **"Settings"** -> **"Environment Variables"** kısmına gidin.
6.  Buradaki değerleri (`POSTGRES_URL`, `POSTGRES_USER` vb.) bir yere not edin. (Otomatik eklenmiş olabilir, "Show Secret" diyip kopyalayın).

## 3. GitHub Actions (Otomasyon İşçisi)
Modelin otomatik eğitilmesi için GitHub'a veritabanı şifrelerini vermemiz lazım.
1.  GitHub Reponuzda **Settings** -> **Secrets and variables** -> **Actions** kısmına gidin.
2.  **"New repository secret"** diyerek aşağıdaki anahtarları (Vercel'den aldığınız değerlerle) ekleyin:
    *   `POSTGRES_URL` (Genellikle bu tek başına yeterlidir ama scriptte ayrı ayrı da tanımlanmış olabilir, `data_clean_and_train.py` sadece `POSTGRES_URL` kullanacak şekilde ayarlandı.)
    *   **Dikkat:** `data_clean_and_train.py` dosyasında kod `os.environ.get("POSTGRES_URL")` kullanıyor. GitHub Secret adı da `POSTGRES_URL` olmalı.

## 4. Test Edin
1.  GitHub'da **"Actions"** sekmesine gidin.
2.  `Hourly Data Scraping and Training` iş akışını göreceksiniz.
3.  Sol taraftan seçip **"Run workflow"** diyerek manuel tetikleyin.
4.  Başarılı olursa (Yeşil tik), Vercel'deki sitenizi yenileyin. Verilerin geldiğini göreceksiniz!
