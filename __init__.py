from Process import ExportToPDF as export


export()
import json
from datetime import datetime


with open("config.json", "r") as file:
    data = json.load(file)

data["LastExport"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
data["LogLevel"] = "INFO"  
data["Message"] = "Konfigurasi database diperbarui"


with open("config.json", "w") as file:
    json.dump(data, file, indent=4)