from DBConnect import Connect as Connect
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import datetime

def FetchData():
    """Mengambil 5 data terakhir dan 100 data terakhir untuk grafik."""
    conn = Connect()
    if isinstance(conn, str):
        return conn
    cursor = conn.cursor()
    
    cursor.execute("SELECT TOP 5 * FROM tbl_t_firewall_cpu ORDER BY id DESC")
    last_5_data = cursor.fetchall()

    cursor.execute("SELECT TOP 10 id, fw_cpu_usage_percentage FROM tbl_t_firewall_cpu ORDER BY id DESC")
    last_100_data = cursor.fetchall()

    conn.close()
    return last_5_data, last_100_data

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

def ExportToPDF(filename="firewall_report.pdf",time=datetime.datetime.now()):
    """Mengekspor 5 data terakhir dan grafik ke dalam PDF dalam bentuk tabel."""
    last_5_data, last_100_data = FetchData()
    
    if isinstance(last_5_data, str):
        print(last_5_data)
        return last_5_data

    graph_path = GenerateGraph(last_100_data)

    if time is None:
        time = datetime.datetime.now()
    
    # Format waktu sebagai string jika perlu
    time_str = time.strftime("%d-%m-%Y %H:%M:%S")

    # Setup dokumen
    doc = SimpleDocTemplate(filename, pagesize=letter)
    elements = []

    styles = getSampleStyleSheet()
    title = Paragraph("Laporan Penggunaan CPU Firewall", styles["Heading1"])
    elements.append(title)
    elements.append(Spacer(1, 12))

    

    subtitle = Paragraph(time_str, styles["Heading2"])
    elements.append(subtitle)
    elements.append(Spacer(1, 6))

    # Tambahkan tabel untuk 5 data terakhir
    subtitle = Paragraph("5 Data Terbaru:", styles["Heading2"])
    elements.append(subtitle)
    elements.append(Spacer(1, 6))

    
    # Menambahkan header kolom (opsional, tergantung isi tabelmu)
    header = ["ID", "fk_m_firewall", "cpu_usage_percentage", "waktu"]  # Sesuaikan nama kolom dengan yang kamu miliki
    data = [header] + [list(row) for row in last_5_data]

    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 24))

    # Tambahkan grafik
    graph = Image(graph_path, width=400, height=200)
    elements.append(Paragraph("Grafik Penggunaan CPU:", styles["Heading2"]))
    elements.append(Spacer(1, 6))
    elements.append(graph)

    # Simpan dokumen
    doc.build(elements)
    print(f"PDF berhasil dibuat: {filename}")
    return filename