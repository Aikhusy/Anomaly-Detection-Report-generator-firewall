import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
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
    Membuat plot distribusi untuk setiap kolom dalam dataframe
    Menampilkan statistik deskriptif termasuk mean, median, mode, quartiles dan Z-score
    """    
    fig, axes = plt.subplots(nrows=len(df.columns) - 1, ncols=1, figsize=(10, 5 * (len(df.columns) - 1)))
    if len(df.columns) - 1 == 1:
        axes = [axes]

    plot_idx = 0
    for col in df.columns:
        if col == 'created_at':
            continue

        ax = axes[plot_idx]

        if col == 'fw_days_uptime':
            ax.plot(df['created_at'], df[col], marker='o', linestyle='-', color='skyblue')
            ax.set_title(f'Days Uptime')
            ax.set_xlabel('Created At')
            ax.set_ylabel(col)
            
            # Tambahkan rata-rata sebagai garis horizontal
            mean_val = df[col].mean()
            ax.axhline(mean_val, color='red', linestyle='--', 
                      label=f'Mean: {mean_val:.2f}')
            
            # Tambahkan Q1, Q2, Q3
            q1 = df[col].quantile(0.25)
            q2 = df[col].median()
            q3 = df[col].quantile(0.75)
            
            ax.axhline(q1, color='green', linestyle='-.', 
                      label=f'Q1: {q1:.2f}')
            ax.axhline(q2, color='blue', linestyle='-.', 
                      label=f'Q2 (Median): {q2:.2f}')
            ax.axhline(q3, color='purple', linestyle='-.', 
                      label=f'Q3: {q3:.2f}')
            
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        else:
            # Plot KDE
            df[col].plot(kind='kde', ax=ax, color='skyblue', linewidth=2, 
                        title=f'Distribusi KDE: {col}')

            # Hitung statistik deskriptif
            mean_val = df[col].mean()
            median_val = df[col].median()
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else None
            
            # Kuartil
            q1 = df[col].quantile(0.25)
            q2 = median_val  # Q2 adalah median
            q3 = df[col].quantile(0.75)
            
            # Hitung Z-score
            z_scores = stats.zscore(df[col])
            # Temukan nilai dengan Z-score tertinggi (outlier potensial)
            max_z_idx = np.abs(z_scores).argmax()
            max_z_val = df[col].iloc[max_z_idx]
            max_z_score = z_scores[max_z_idx]
            
            # Tambahkan garis vertikal untuk statistik
            ax.axvline(mean_val, color='red', linestyle='--', 
                      label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='blue', linestyle='--', 
                      label=f'Median (Q2): {median_val:.2f}')
            if mode_val is not None:
                ax.axvline(mode_val, color='orange', linestyle='--', 
                          label=f'Mode: {mode_val:.2f}')
            
            # Tambahkan Q1 dan Q3
            ax.axvline(q1, color='green', linestyle='-.', 
                      label=f'Q1: {q1:.2f}')
            ax.axvline(q3, color='purple', linestyle='-.', 
                      label=f'Q3: {q3:.2f}')
            
            # Tambahkan nilai Z-score tertinggi
            ax.axvline(max_z_val, color='magenta', linestyle=':', 
                      label=f'Max |Z-score|: {max_z_score:.2f} at {max_z_val:.2f}')
            
            # Tambahkan box di pojok kanan atas untuk informasi statistik
            stats_text = (
                f"Statistik Deskriptif:\n"
                f"Mean: {mean_val:.2f}\n"
                f"Q1 (25%): {q1:.2f}\n"
                f"Median (Q2): {q2:.2f}\n"
                f"Q3 (75%): {q3:.2f}\n"
                f"IQR: {(q3-q1):.2f}\n"
                f"Max |Z-score|: {abs(max_z_score):.2f}"
            )
            
            # Tambahkan teks statistik sebagai annotasi
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=props)
            
            ax.set_xlabel(col)
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        ax.grid(True, alpha=0.3)
        plot_idx += 1

    plt.tight_layout()
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png', dpi=150, bbox_inches='tight')
    img_data.seek(0)
    plt.close()

    return img_data


def DistributionPlot(elements, img_data,index):
    page_width, _ = A4
    if (index !=2):
        img = Image(img_data, width=page_width*0.8, height=page_width*0.8)
        
        outer_table = Table([[img]])
        outer_table.setStyle(TableStyle([
            ('LEFTPADDING', (0, 0), (-1, -1), 0),  
            ('RIGHTPADDING', (0, 0), (-1, -1), -30),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
    else :
        img = Image(img_data, width=page_width*0.8, height=page_width*0.4)
        
        # Bungkus gambar dengan Table untuk mengatur posisi
        outer_table = Table([[img]])
        outer_table.setStyle(TableStyle([
            ('LEFTPADDING', (0, 0), (-1, -1), 0),  # Margin kiri
            ('RIGHTPADDING', (0, 0), (-1, -1), -30),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
    
    elements.append(outer_table)
    elements.append(Spacer(1, 20))
    return elements

def plot_distribution(df, columns):
    fig, axes = plt.subplots(nrows=len(columns), ncols=1, figsize=(10, 5 * len(columns)))
    if len(columns) == 1:
        axes = [axes]

    for idx, col in enumerate(columns):
        ax = axes[idx]
        if col == 'fw_days_uptime':
            ax.plot(df['created_at'], df[col], marker='o', linestyle='-', color='skyblue')
            ax.set_title(f'Days Uptime')
            ax.set_xlabel('Created At')
            ax.set_ylabel(col)

            mean_val = df[col].mean()
            q1 = df[col].quantile(0.25)
            q2 = df[col].median()
            q3 = df[col].quantile(0.75)

            ax.axhline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            ax.axhline(q1, color='green', linestyle='-.', label=f'Q1: {q1:.2f}')
            ax.axhline(q2, color='blue', linestyle='-.', label=f'Q2 (Median): {q2:.2f}')
            ax.axhline(q3, color='purple', linestyle='-.', label=f'Q3: {q3:.2f}')
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        else:
            df[col].plot(kind='kde', ax=ax, color='skyblue', linewidth=2, title=f'Distribusi KDE: {col}')
            mean_val = df[col].mean()
            median_val = df[col].median()
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else None
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            z_scores = stats.zscore(df[col])
            max_z_idx = np.abs(z_scores).argmax()
            max_z_val = df[col].iloc[max_z_idx]
            max_z_score = z_scores[max_z_idx]

            ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='blue', linestyle='--', label=f'Median (Q2): {median_val:.2f}')
            if mode_val is not None:
                ax.axvline(mode_val, color='orange', linestyle='--', label=f'Mode: {mode_val:.2f}')
            ax.axvline(q1, color='green', linestyle='-.', label=f'Q1: {q1:.2f}')
            ax.axvline(q3, color='purple', linestyle='-.', label=f'Q3: {q3:.2f}')
            ax.axvline(max_z_val, color='magenta', linestyle=':', label=f'Max |Z-score|: {max_z_score:.2f} at {max_z_val:.2f}')

            stats_text = (
                f"Statistik Deskriptif:\n"
                f"Mean: {mean_val:.2f}\n"
                f"Q1 (25%): {q1:.2f}\n"
                f"Median (Q2): {median_val:.2f}\n"
                f"Q3 (75%): {q3:.2f}\n"
                f"IQR: {(q3-q1):.2f}\n"
                f"Max |Z-score|: {abs(max_z_score):.2f}"
            )
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=props)
            ax.set_xlabel(col)
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png', dpi=150, bbox_inches='tight')
    img_data.seek(0)
    plt.close()
    return img_data


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

def ConclusionTable(elements, df):
    """
    Menambahkan kesimpulan berdasarkan analisis data uptime dalam bentuk tabel
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
    
    page_width, _ = A4
    
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
    if avg_load_1 > avg_load_5 > avg_load_15:
        load_trend = "Beban CPU cenderung meningkat dalam waktu terakhir"
    elif avg_load_1 < avg_load_5 < avg_load_15:
        load_trend = "Beban CPU cenderung menurun, sistem semakin stabil"
    else:
        load_trend = "Beban CPU berfluktuasi, aktivitas sistem tidak konsisten"
    
    user_to_load_ratio = "tinggi" if (avg_users / max(avg_load_1, 0.01)) < 10 else "rendah"
    resource_conclusion = f"Rasio user terhadap beban sistem {user_to_load_ratio}"
    
    # Judul untuk setiap tabel
    title_1 = ["Kesimpulan Analisis Data Uptime"]
    title_2 = ["Analisis Beban CPU"]
    
    
    # Data untuk tabel utama
    data_1 = [title_1]
    data_1.append(["Parameter", "Nilai"])
    data_1.append(["Uptime Tertinggi", f"{max_uptime} hari"])
    data_1.append(["Perangkat Pernah Mati", ever_down])
    data_1.append(["Rata-rata Jumlah User", f"{avg_users} user"])
    data_1.append(["Jumlah User Terbanyak", f"{max_users} user"])
    
    # Data untuk tabel beban CPU
    data_2 = [title_2]
    data_2.append(["Periode", "Rata-rata Beban CPU"])
    data_2.append(["1 menit terakhir", str(avg_load_1)])
    data_2.append(["5 menit terakhir", str(avg_load_5)])
    data_2.append(["15 menit terakhir", str(avg_load_15)])
    
    

    def create_table(data):
        if len(data[0]) == 1:
            col_widths = [page_width * 0.2, page_width * 0.4]
        else:
            col_widths = [page_width * 0.2, page_width * 0.4]

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
    
    # Bungkus dan beri margin kiri
    
    elements.append(create_table(data_1))
    elements.append(Spacer(1, 15))
    elements.append(create_table(data_2))
    elements.append(Spacer(1, 15))
    
    return elements

def ProcessUptimeData(uptime_data):
    try:
        logger.info("Memproses data uptime...")
        column_names = ['fw_days_uptime', 'fw_number_of_users', 
                        'fw_load_avg_1_min', 'fw_load_avg_5_min', 
                        'fw_load_avg_15_min', 'created_at']
        data_list = [tuple(row) for row in uptime_data]
        df = pd.DataFrame(data_list, columns=column_names)
        return df
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error dalam pemrosesan data: {str(e)}\n{tb}")
        return None

def GlobalHandler(elements, uptime_data):
    try:
        df = ProcessUptimeData(uptime_data)
        if df is None:
            elements.append(Paragraph("Error: Tidak dapat memproses data", getSampleStyleSheet()['Normal']))
            return elements

        page_groups = [
            ['fw_days_uptime', 'fw_number_of_users'],
            ['fw_load_avg_1_min', 'fw_load_avg_5_min'],
            ['fw_load_avg_15_min']
        ]

        for i, group in enumerate(page_groups):
            elements.append(PageBreak())
            img_data = plot_distribution(df, group)
            elements = DistributionPlot(elements, img_data, index=i)

        elements = ConclusionTable(elements, df)
        
        return elements

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error dalam integrasi ke PDF: {str(e)}\n{tb}")
        elements.append(Paragraph(f"Error: {str(e)}", getSampleStyleSheet()['Normal']))
        return elements
