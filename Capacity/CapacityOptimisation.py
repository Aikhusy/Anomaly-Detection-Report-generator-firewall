import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from reportlab.platypus import Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def create_capacity_optimisation_chart(df):
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(df))
    ax.plot(x, df['fw_vals'], label='Koneksi Aktif', color='blue', marker='o')
    ax.plot(x, df['fw_peaks'], label='Puncak Koneksi', color='red', linestyle='--', marker='x')
    ax.plot(x, df['fw_slinks'], label='Session Links', color='green', linestyle='-.', marker='s')

    ax.set_title('Distribusi Koneksi Firewall')
    ax.set_xlabel('Periode')
    ax.set_ylabel('Jumlah Koneksi')
    ax.legend()
    ax.grid(True)

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='PNG', bbox_inches='tight')
    img_buffer.seek(0)
    plt.close(fig)

    return img_buffer

def capacity_optimization_conclusions(df):
    limit = df['fw_limit'].iloc[0]
    avg_val = df['fw_vals'].mean()
    max_peak = df['fw_peaks'].max()
    slinks_ratio = df['fw_slinks'].mean() / avg_val if avg_val != 0 else 0

    growth_rate = df['fw_vals'].diff().mean()
    growth_percent = (growth_rate / avg_val) * 100 if avg_val != 0 else 0

    forecast = "Perlu upgrade dalam waktu dekat" if growth_percent > 5 else "Kapasitas masih cukup untuk tren saat ini"

    return {
        'connection_limit': limit,
        'current_avg': avg_val,
        'max_peak': max_peak,
        'slinks_ratio': slinks_ratio,
        'forecast': forecast,
        'growth_rate': growth_rate,
        'growth_percent': growth_percent
    }

def process_capacity_data(data):
    column_names = ['fw_hostname', 'fw_names', 'fw_id', 'fw_vals', 
                   'fw_peaks', 'fw_slinks', 'fw_limit']
    data_list = [tuple(row) for row in data]
    df = pd.DataFrame(data_list, columns=column_names)
    conclusions = capacity_optimization_conclusions(df)
    return conclusions, df

def Title(elements, word):
    styles = getSampleStyleSheet()
    if 'HeaderStyle' not in styles:
        styles.add(ParagraphStyle(
            name='HeaderStyle',
            fontName='Helvetica-Bold',
            fontSize=14,
            leading=16,
            alignment=0,
        ))
    elements.append(Paragraph(word, styles['HeaderStyle']))
    elements.append(Spacer(1, 6))
    return elements

def CapacityDistributionPlot(elements, img_data, title):
    page_width, _ = A4
    elements = Title(elements, title)
    elements.append(Spacer(1, 12))
    img = Image(img_data, width=page_width*0.8, height=page_width*0.5)
    outer_table = Table([[img]])
    outer_table.setStyle(TableStyle([
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    elements.append(outer_table)
    elements.append(Spacer(1, 20))
    return elements

def CapacityConclusionTable(elements, conclusions, df):
    page_width, _ = A4
    title_capacity = ["Kesimpulan Analisis Kapasitas Koneksi Firewall"]
    data_capacity = [title_capacity]
    data_capacity.append(["Parameter", "Nilai"])
    connection_limit = conclusions.get('connection_limit')
    if isinstance(connection_limit, (int, float)):
        formatted_limit = f"{connection_limit:,}"
    else:
        formatted_limit = str(connection_limit)

    data_capacity.append(["Total Limit Koneksi", formatted_limit])
    data_capacity.append(["Trend Pertumbuhan", f"{conclusions['growth_rate']:.2f} koneksi/periode ({conclusions['growth_percent']:.2f}%)"])
    data_capacity.append(["Rata-rata Session Links", f"{df['fw_slinks'].mean():.2f} (Ratio: {conclusions['slinks_ratio']:.2f})"])

    title_recommendation = ["Rekomendasi Optimisasi Kapasitas"]
    data_recommendation = [title_recommendation]
    data_recommendation.append(["Aspek", "Rekomendasi"])
    data_recommendation.append(["Forecast", conclusions['forecast']])
    
    def create_table(data):
        col_widths = [page_width * 0.2, page_width * 0.6]
        table = Table(data, colWidths=col_widths, hAlign='LEFT')
        style = [
            ('SPAN', (0, 0), (-1, 0)),
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, 1), colors.lightgrey),
            ('ALIGN', (0, 1), (-1, 1), 'CENTER'),
            ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 1), (-1, 1), 9),
            ('FONTNAME', (0, 2), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 2), (-1, -1), 9),
            ('ALIGN', (0, 2), (0, -1), 'LEFT'),
            ('ALIGN', (1, 2), (1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 1), (-1, -1), 1, colors.black),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]
        table.setStyle(TableStyle(style))
        return table

    elements.append(create_table(data_capacity))
    elements.append(Spacer(1, 15))
    elements.append(create_table(data_recommendation))
    elements.append(Spacer(1, 15))
    return elements

def GlobalHandler(elements, capacity_data):
    try:
        import logging
        import time
        import traceback

        logger = logging.getLogger(__name__)
        start_time = time.time()

        logger.info("Memproses data capacity optimisation firewall...")

        conclusions, df = process_capacity_data(capacity_data)

        if conclusions and df is not None:
            elements.append(PageBreak())
            elements = Title(elements, "Kesimpulan dan Rekomendasi Kapasitas")
            elements.append(Spacer(1, 10))
            elements = CapacityConclusionTable(elements, conclusions, df)

            img_data = create_capacity_optimisation_chart(df)
            elements = CapacityDistributionPlot(elements, img_data, "Distribusi Kapasitas Koneksi")

        else:
            elements.append(Paragraph("Error: Tidak dapat membuat analisis kapasitas", getSampleStyleSheet()['Normal']))
            elements.append(Spacer(1, 20))

        logger.info(f"Analisis capacity selesai dalam {time.time() - start_time:.2f} detik")
        return elements

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"Error dalam integrasi ke PDF: {str(e)}\n{tb}")
        elements.append(Paragraph(f"Error: {str(e)}", getSampleStyleSheet()['Normal']))
        elements.append(Spacer(1, 20))
        return elements