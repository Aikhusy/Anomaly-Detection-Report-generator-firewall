import json
from datetime import datetime
from Process import ExportToPDF as export
import tkinter as tk


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
    root = tk.Tk()
    root.title("Aplikasi Tkinter Sederhana")
    root.geometry("300x100")

    tombol = tk.Button(root, text="Klik Saya", command=export)
    tombol.pack(pady=20)

    root.mainloop()
    update_config()


if __name__ == "__main__":
    main()
