from DBConnect import Connect as Connect
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
import datetime
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Image,
    Paragraph, Spacer
)
import pandas as pd
from reportlab.lib import colors
from DocumentHeader import GlobalHandler as DocumentHeader
from DocumentGeneral import GlobalHandler as DocumentGeneral
from UptimeAnomalyDetect import GlobalHandler as UptimeAnomaly

def FetchData():

    conn = Connect()
    if isinstance(conn, str):
        return conn
    cursor = conn.cursor()

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
        SELECT TOP 20 
            fw_days_uptime, 
            fw_number_of_users, 
            fw_load_avg_1_min, 
            fw_load_avg_5_min, 
            fw_load_avg_15_min, 
            created_at
        FROM tbl_t_firewall_uptime
        WHERE fk_m_firewall = 1
        ORDER BY created_at DESC
    
    """
    cursor.execute(query)
    uptime = cursor.fetchall()

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
    return counted_rows,current_status, uptime

def ExportToPDF(filename="firewall_report.pdf", time=datetime.datetime.now()):
    counted_rows, current_status, uptime = FetchData()

    # uptime_columns = [
    #     'fw_days_uptime', 
    #     'fw_number_of_users', 
    #     'fw_load_avg_1_min', 
    #     'fw_load_avg_5_min', 
    #     'fw_load_avg_15_min', 
    #     'created_at'
    # ]

    # df_uptime = pd.DataFrame(uptime, columns=uptime_columns)

    datass = {
        "counted_rows": counted_rows,
        "current_status": current_status
    }

    if time is None:
        time = datetime.datetime.now()

    dataAnomaly = UptimeAnomaly(uptime)

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

    # print(datass["counted_rows"])

    elements = DocumentHeader(elements, inputs)
    elements = DocumentGeneral(elements, datass)

    doc.build(elements)
    print(f"PDF berhasil dibuat: {filename}")
    return filename