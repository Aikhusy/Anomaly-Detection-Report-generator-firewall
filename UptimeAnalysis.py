import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
import traceback
import io
from reportlab.platypus import Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_distribution(df):
    """
    Membuat plot distribusi KDE untuk setiap kolom numerik dalam DataFrame
    dan mengembalikan file-like object dari plot
    """
    fig, axes = plt.subplots(nrows=len(df.columns)-1, ncols=1, figsize=(10, 4 * (len(df.columns)-1)))
    if len(df.columns)-1 == 1:
        axes = [axes]
        
    for i, col in enumerate(df.columns):
        if col == 'created_at':
            continue
        ax = axes[i if i < len(axes) else 0]
        
        # Plot KDE
        df[col].plot(kind='kde', ax=ax, color='skyblue', linewidth=2, title=f'Distribusi KDE: {col}')
        
        # Tambahkan garis mean, median, dan modus
        mean_val = df[col].mean()
        median_val = df[col].median()
        mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else None
        
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
        if mode_val is not None:
            ax.axvline(mode_val, color='orange', linestyle='--', label=f'Modus: {mode_val:.2f}')
        
        ax.set_xlabel(col)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    
    # Alih-alih menyimpan ke file, simpan ke dalam memory buffer
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png', dpi=150)
    img_data.seek(0)  # Kembali ke awal buffer
    plt.close()
    
    return img_data

def DistributionPlot(elements, img_data):
    page_width, _ = A4
    
    # Tambahkan judul untuk bagian plot
    elements = Title(elements, "Distribusi Data Uptime")
    
    # Buat objek Image dari data gambar
    img = Image(img_data, width=page_width*0.5, height=page_width*1)
    
    # Bungkus gambar dengan Table untuk mengatur posisi
    outer_table = Table([[img]])
    outer_table.setStyle(TableStyle([
        ('LEFTPADDING', (0, 0), (-1, -1), 60),  # Margin kiri
        ('RIGHTPADDING', (0, 0), (-1, -1), -30),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    
    elements.append(outer_table)
    elements.append(Spacer(1, 20))
    return elements

def Title(elements, word):
    styless = getSampleStyleSheet()
    if 'HeaderStyle' not in styless:
        styless.add(ParagraphStyle(
            name='HeaderStyle',
            fontName='TimesNewRoman',
            fontSize=14,
            leading=16,
            alignment=0,  # Center
            leftIndent=-10,
            rightIndent=10,
        ))
    elements.append(Paragraph(word, styless['HeaderStyle']))
    elements.append(Spacer(1, 6))
    return elements

def ConclusionParagraph(elements, df):
    """
    Menambahkan kesimpulan berdasarkan analisis data uptime
    """
    styless = getSampleStyleSheet()
    
    # Tambahkan style untuk kesimpulan jika belum ada
    if 'ConclusionStyle' not in styless:
        styless.add(ParagraphStyle(
            name='ConclusionStyle',
            fontName='TimesNewRoman',
            fontSize=12,
            leading=14,
            alignment=0,  # Left
            leftIndent=10,
            rightIndent=10,
        ))
    
    # Tambahkan style untuk bullet points
    if 'BulletStyle' not in styless:
        styless.add(ParagraphStyle(
            name='BulletStyle',
            fontName='TimesNewRoman',
            fontSize=12,
            leading=14,
            alignment=0,  # Left
            leftIndent=30,
            rightIndent=10,
            firstLineIndent=-15,
            bulletIndent=15,
        ))
    
    # Hitung statistik dari data
    max_uptime = df['fw_days_uptime'].max()
    min_uptime = df['fw_days_uptime'].min()
    ever_down = "Ya" if min_uptime == 0 else "Tidak"
    
    avg_users = int(round(df['fw_number_of_users'].mean()))
    max_users = df['fw_number_of_users'].max()
    
    avg_load_1 = round(df['fw_load_avg_1_min'].mean(), 2)
    avg_load_5 = round(df['fw_load_avg_5_min'].mean(), 2)
    avg_load_15 = round(df['fw_load_avg_15_min'].mean(), 2)
    
    # Analisis tambahan
    load_trend = ""
    if avg_load_1 > avg_load_5 > avg_load_15:
        load_trend = "Beban CPU cenderung meningkat dalam waktu terakhir, yang mungkin mengindikasikan adanya peningkatan aktivitas atau potensi masalah."
    elif avg_load_1 < avg_load_5 < avg_load_15:
        load_trend = "Beban CPU cenderung menurun, yang menunjukkan sistem semakin stabil."
    else:
        load_trend = "Beban CPU berfluktuasi, yang menunjukkan aktivitas sistem tidak konsisten."
    
    user_to_load_ratio = "tinggi" if (avg_users / max(avg_load_1, 0.01)) < 10 else "rendah"
    resource_conclusion = f"Rasio jumlah user terhadap beban sistem {user_to_load_ratio}, " + \
                        ("mengindikasikan penggunaan resource yang intensif per user." if user_to_load_ratio == "tinggi" else "mengindikasikan penggunaan resource yang efisien.")
    
    # Tambahkan judul kesimpulan
    elements = Title(elements, "Kesimpulan Analisis Data Uptime")
    
    # Tambahkan paragraf kesimpulan
    uptime_summary = f"Waktu nyala perangkat tertinggi adalah <b>{max_uptime}</b> hari. " + \
                    f"Perangkat pernah mati: <b>{ever_down}</b>."
    elements.append(Paragraph(uptime_summary, styless['ConclusionStyle']))
    elements.append(Spacer(1, 6))
    
    user_summary = f"Rata-rata jumlah user yang aktif adalah <b>{avg_users}</b> user. " + \
                  f"User yang pernah login terbanyak adalah <b>{max_users}</b> user."
    elements.append(Paragraph(user_summary, styless['ConclusionStyle']))
    elements.append(Spacer(1, 10))
    
    # Bullet points untuk load average
    elements.append(Paragraph("<b>Analisis Beban CPU:</b>", styless['ConclusionStyle']))
    elements.append(Spacer(1, 4))
    
    elements.append(Paragraph("• Dalam <b>1 menit terakhir</b>, rata-rata ada <b>{}</b> proses yang membutuhkan CPU.".format(avg_load_1),
                             styless['BulletStyle']))
    elements.append(Paragraph("• Dalam <b>5 menit terakhir</b>, rata-rata ada <b>{}</b> proses yang membutuhkan CPU.".format(avg_load_5),
                             styless['BulletStyle']))
    elements.append(Paragraph("• Dalam <b>15 menit terakhir</b>, rata-rata ada <b>{}</b> proses yang membutuhkan CPU.".format(avg_load_15),
                             styless['BulletStyle']))
    elements.append(Spacer(1, 10))
    
    # Kesimpulan tambahan
    elements.append(Paragraph("<b>Wawasan Tambahan:</b>", styless['ConclusionStyle']))
    elements.append(Spacer(1, 4))
    elements.append(Paragraph("• " + load_trend, styless['BulletStyle']))
    elements.append(Paragraph("• " + resource_conclusion, styless['BulletStyle']))
    
    # Rekomendasi (opsional)
    if avg_load_1 > 5 or avg_load_5 > 5:  # Ambang batas load average tinggi
        elements.append(Spacer(1, 10))
        elements.append(Paragraph("<b>Rekomendasi:</b> Pertimbangkan untuk memonitor penggunaan sistem lebih lanjut karena beban CPU rata-rata cukup tinggi.",
                                styless['ConclusionStyle']))
    
    elements.append(Spacer(1, 20))
    return elements

def ProcessUptimeData(uptime_data):
    try:
        start_time = time.time()
        
        logger.info("Memproses data uptime...")
        column_names = ['fw_days_uptime', 'fw_number_of_users', 
                   'fw_load_avg_1_min', 'fw_load_avg_5_min', 'fw_load_avg_15_min', 'created_at']

        # Convert pyodbc.Row objects to regular Python lists or tuples
        data_list = [tuple(row) for row in uptime_data]
        
        # Create DataFrame from the converted data
        df = pd.DataFrame(data_list, columns=column_names)
        
        # Generate plot and get image data
        img_data = plot_distribution(df)
        
        return img_data, df
        
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error dalam pembuatan distribusi: {str(e)}\n{tb}")
        return None, None

def GlobalHandler(elements, uptime_data):
    try:
        # Dapatkan data gambar dan DataFrame dari ProcessUptimeData
        img_data, df = ProcessUptimeData(uptime_data)
        
        if img_data and df is not None:

            elements.append(PageBreak())
            # Tambahkan plot ke elements PDF
            elements = DistributionPlot(elements, img_data)
            
            elements.append(PageBreak())
            # Tambahkan kesimpulan analisis
            elements = ConclusionParagraph(elements, df)
        else:
            # Jika ada error, tambahkan pesan error ke PDF
            elements.append(Paragraph("Error: Tidak dapat membuat plot distribusi", getSampleStyleSheet()['Normal']))
            elements.append(Spacer(1, 20))
            
        return elements
        
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error dalam integrasi ke PDF: {str(e)}\n{tb}")
        elements.append(Paragraph(f"Error: {str(e)}", getSampleStyleSheet()['Normal']))
        elements.append(Spacer(1, 20))
        return elements