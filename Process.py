from DBConnect import Connect as Connect
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
import datetime
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Image,
    Paragraph, Spacer
)
from reportlab.lib import colors
from DocumentHeader import GlobalHandler as DocumentHeader
from DocumentGeneral import GlobalHandler as DocumentGeneral

def FetchData():
    """Mengambil 5 data terakhir dan 100 data terakhir untuk grafik."""
    conn = Connect()
    if isinstance(conn, str):
        return conn
    cursor = conn.cursor()
    
    # cursor.execute("SELECT TOP 5 * FROM tbl_t_firewall_cpu ORDER BY id DESC")
    # last_5_data = cursor.fetchall()

    # cursor.execute("SELECT TOP 10 id, fw_cpu_usage_percentage FROM tbl_t_firewall_cpu ORDER BY id DESC")
    # last_100_data = cursor.fetchall()

    query = """
        SELECT f.fw_name, counts.total_row
        FROM (
            SELECT fk_m_firewall, COUNT(*) AS total_row
            FROM tbl_t_firewall_uptime
            GROUP BY fk_m_firewall
        ) AS counts
        INNER JOIN tbl_m_firewall AS f ON counts.fk_m_firewall = f.id
    """
    cursor.execute(query)
    counted_rows = cursor.fetchall()

    query = """
        SELECT 
            f.fw_name,
            cs.uptime,
            cs.fwtmp,
            cs.varloglog,
            cs.ram,
            cs.swap,
            cs.memory_error,
            cs.cpu,
            cs.rx_error_total,
            cs.tx_error_total,
            cs.sync_mode,
            cs.sync_state,
            cs.license_expiration_status,
            cs.raid_state,
            cs.hotfix_module
        FROM 
            tbl_t_firewall_current_status AS cs
        INNER JOIN 
            tbl_m_firewall AS f 
            ON cs.fk_m_firewall = f.id;

    """
    cursor.execute(query)
    current_status = cursor.fetchall()

    conn.close()
    return counted_rows,current_status

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
    counted_rows,current_status = FetchData()

    datass={
        "counted_rows" : counted_rows,
        "current_status": current_status
    }

    if time is None:
        time = datetime.datetime.now()

    # time_str = time.strftime("%d-%m-%Y %H:%M:%S")
    doc = SimpleDocTemplate(filename, pagesize=letter)
    elements = []
    
    styles = getSampleStyleSheet()
    styleN = styles["BodyText"]
    styleH1 = styles["Heading1"]
    styleH2 = styles["Heading2"]

    inputs = {
        "sitename": "BRCG01",
        "startdate": "2025-01-01",
        "enddate": "2025-04-01",
        "exportdate": "2025-04-10",
        "totalfw": 5,
        "month": "April",
        "year": "2025",
        "image_path": "logo.png"  
    }

    print(datass["counted_rows"])
    elements=DocumentHeader(elements,inputs)
    elements=DocumentGeneral(elements,datass)

    doc.build(elements)
    print(f"PDF berhasil dibuat: {filename}")
    return filename
