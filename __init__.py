import json
from datetime import datetime
from Process import ExportToPDF as export



def update_config(log_level="INFO", message="Konfigurasi database diperbarui"):
    """Memperbarui file config.json dengan informasi terbaru."""
    try:
        with open("config.json", "r") as file:
            data = json.load(file)

        data["LastExport"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data["LogLevel"] = log_level
        data["Message"] = message

        with open("config.json", "w") as file:
            json.dump(data, file, indent=4)

        print("Konfigurasi berhasil diperbarui.")
    except Exception as e:
        print(f"Gagal memperbarui konfigurasi: {e}")


def main():
    """Fungsi utama untuk menjalankan proses export dan update konfigurasi."""
    print("Memulai proses export ke PDF...")
    export()
    update_config()


if __name__ == "__main__":
    main()
