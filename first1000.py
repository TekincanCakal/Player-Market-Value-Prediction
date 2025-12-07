import json

# JSON dosyasını yükleme
with open('players_data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# İlk 1000 veriyi ayırma
first_1000_data = data[:1000]

# Yeni JSON dosyasına kaydetme
with open('first_1000_data.json', 'w', encoding='utf-8') as output_file:
    json.dump(first_1000_data, output_file, ensure_ascii=False, indent=4)

print("İlk 1000 veri 'first_1000_data.json' dosyasına kaydedildi.")
