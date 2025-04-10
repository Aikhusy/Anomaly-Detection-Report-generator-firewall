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

    cursor.execute("select f.fw_name, cs.uptime, cs.fwtmp, cs.varloglog, cs.ram, cs.swap, cs.memory_error, cs.cpu, cs.rx_error_total, cs.tx_error_total, cs.sync_mode, cs.sync_state, cs.license_expiration_status, cs.raid_state, cs.hotfix_module  from tbl_t_firewall_current_status as cs inner join tbl_m_firewall as f on cs.fk_m_firewall = f.id ")
    current_status= cursor.fetchall()

    conn.close()
    return last_5_data, last_100_data, current_status

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

def ExportToPDF(filename="firewall_report.pdf", time=datetime.datetime.now()):
    """Mengekspor 5 data terakhir dan grafik ke dalam PDF dalam bentuk tabel."""
    last_5_data, last_100_data, current_status = FetchData()

    if isinstance(last_5_data, str):
        print(last_5_data)
        return last_5_data

    graph_path = GenerateGraph(last_100_data)

    if time is None:
        time = datetime.datetime.now()

    time_str = time.strftime("%d-%m-%Y %H:%M:%S")
    doc = SimpleDocTemplate(filename, pagesize=letter)
    elements = []

    styles = getSampleStyleSheet()
    styleN = styles["BodyText"]
    styleH1 = styles["Heading1"]
    styleH2 = styles["Heading2"]

    # Judul dan waktu
    elements.append(Paragraph("Laporan Penggunaan CPU Firewall", styleH1))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(time_str, styleH2))
    elements.append(Spacer(1, 6))

    # Header untuk tabel
    header1 = ["Firewall Name", "Uptime", "FW Temp", "/var/log", "RAM", "Swap", "Memory Error", "CPU"]
    header2 = ["RX Error", "TX Error", "Sync Mode", "Sync State", "License Status", "RAID State", "Hotfix Module"]

    # Siapkan data tabel bagian 1 dan 2
    data1 = [header1]
    data2 = [header2]

    for row in current_status:
        data1.append([Paragraph(str(cell), styleN) for cell in row[:8]])
        data2.append([Paragraph(str(cell), styleN) for cell in row[8:]])

    # Buat tabel tanpa colWidths agar fleksibel
    table1 = Table(data1)
    table2 = Table(data2)

    # Gaya tabel
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ])

    table1.setStyle(style)
    table2.setStyle(style)

    elements.append(Paragraph("Status Saat Ini (Bagian 1):", styleH2))
    elements.append(Spacer(1, 6))
    elements.append(table1)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Status Saat Ini (Bagian 2):", styleH2))
    elements.append(Spacer(1, 6))
    elements.append(table2)
    elements.append(Spacer(1, 24))

    # Tabel 5 data terbaru
    last_5_header = ["ID", "fk_m_firewall", "CPU Usage (%)", "Timestamp"]
    data_last_5 = [last_5_header] + [[Paragraph(str(cell), styleN) for cell in row] for row in last_5_data]
    table_last_5 = Table(data_last_5)
    table_last_5.setStyle(style)

    elements.append(Paragraph("5 Data Terbaru:", styleH2))
    elements.append(Spacer(1, 6))
    elements.append(table_last_5)
    elements.append(Spacer(1, 24))

    # Grafik
    graph = Image(graph_path, width=400, height=200)
    elements.append(Paragraph("Grafik Penggunaan CPU:", styleH2))
    elements.append(Spacer(1, 6))
    elements.append(graph)

    # Simpan PDF
    doc.build(elements)
    print(f"PDF berhasil dibuat: {filename}")
    return filename
