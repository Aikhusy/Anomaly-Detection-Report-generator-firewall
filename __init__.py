from DBConnect import GlobalHandler as DBConnect 


DBConnect()

import json
from datetime import datetime

# Baca file JSON
with open("HashedDbInfo.json", "r") as file:
    data = json.load(file)

# Tambahkan field baru untuk logging
data["LogTime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
data["LogLevel"] = "INFO"  # Bisa diubah sesuai kebutuhan
data["Message"] = "Konfigurasi database diperbarui"

# Simpan kembali ke file JSON
with open("config.json", "w") as file:
    json.dump(data, file, indent=4)

print("Field logging berhasil ditambahkan.")
