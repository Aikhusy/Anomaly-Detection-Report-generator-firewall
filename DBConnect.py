import pyodbc
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import Image
from JsonFileImporter import GlobalHandler as JsonFileImporter

def GetDBData():
    data = JsonFileImporter("HashedDBInfo.json")
    return data if isinstance(data, (dict, list)) else 0

def GetDBType():
    try:
        data = GetDBData()
        return data
    except KeyError:
        return "Error: 'dbName' not found in the data."
    except Exception as e:
        return f"Unexpected Error: {str(e)}"

def Connect():
    connection_data = GetDBType()
    if isinstance(connection_data, dict):
        try:
            connection_string = (
                f"DRIVER={connection_data['Driver']};"
                f"SERVER={connection_data['Server']};"
                f"DATABASE={connection_data['Database']};"
                f"UID={connection_data['UID']};"
                f"PWD={connection_data['PWD']};"
                f"Encrypt={connection_data['Encrypt']};"
                f"TrustServerCertificate={connection_data['TrustServerCertificate']};"
            )
            return pyodbc.connect(connection_string)
        except Exception as e:
            return f"Unexpected Error: {str(e)}"
    else:
        return "Invalid database configuration format."



def GenerateGraph(last_100_data, graph_path="cpu_usage_graph.png"):
    """Membuat grafik penggunaan CPU dari 100 data terakhir."""
    ids = [row[0] for row in last_100_data]  # ID data
    cpu_usage = [row[1] for row in last_100_data]  # Penggunaan CPU

    plt.figure(figsize=(10, 5))
    plt.plot(ids[::-1], cpu_usage[::-1], marker='o', linestyle='-', color='b', label='CPU Usage')
    plt.xlabel("Data ID")
    plt.ylabel("CPU Usage (%)")
    plt.title("Grafik Penggunaan CPU (100 Data Terakhir)")
    plt.legend()
    plt.grid()
    
    plt.savefig(graph_path)
    plt.close()
    return graph_path

def ExportToPDF(filename="firewall_report.pdf"):
    """Mengekspor 5 data terakhir dan grafik ke dalam PDF."""
    last_5_data, last_100_data = FetchData()
    
    if isinstance(last_5_data, str):  # Jika terjadi error dalam pengambilan data
        print(last_5_data)
        return last_5_data

    graph_path = GenerateGraph(last_100_data)

    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 14)
    c.drawString(30, height - 30, "Laporan Penggunaan CPU Firewall")

    # Menampilkan 5 Data Terakhir
    c.setFont("Helvetica", 12)
    y_position = height - 60
    c.drawString(30, y_position, "5 Data Terbaru:")
    y_position -= 20

    for row in last_5_data:
        c.drawString(30, y_position, str(row))
        y_position -= 20

    # Menambahkan Grafik ke PDF
    c.drawString(30, y_position - 20, "Grafik Penggunaan CPU:")
    y_position -= 250  # Beri ruang untuk gambar

    img = Image(graph_path, width=400, height=200)
    img.drawOn(c, 100, y_position)

    c.save()
    print(f"PDF berhasil dibuat: {filename}")
    return filename

