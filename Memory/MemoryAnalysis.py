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

def plot_memory_usage(df_mem, df_swap):
    """
    Membuat plot penggunaan memori (RAM dan Swap) dari dataframe
    Menampilkan statistik deskriptif untuk kedua jenis memori
    """
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 12))
    
    # Plot untuk RAM Memory
    ax1 = axes[0]
    
    # Hitung persentase penggunaan memori
    df_mem['mem_usage_percent'] = (df_mem['mem_used'] / df_mem['mem_total']) * 100
    
    # Plot penggunaan memori RAM vs waktu
    ax1.plot(df_mem['created_at'], df_mem['mem_usage_percent'], marker='o', linestyle='-', color='blue', 
            label='RAM Usage %')
    ax1.set_title('Penggunaan RAM Memory')
    ax1.set_xlabel('Waktu')
    ax1.set_ylabel('Penggunaan (%)')
    ax1.grid(True, alpha=0.3)
    
    # Tambahkan rata-rata sebagai garis horizontal
    mean_val = df_mem['mem_usage_percent'].mean()
    ax1.axhline(mean_val, color='red', linestyle='--', 
              label=f'Mean: {mean_val:.2f}%')
    
    # Tambahkan Q1, Q2, Q3
    q1 = df_mem['mem_usage_percent'].quantile(0.25)
    q2 = df_mem['mem_usage_percent'].median()
    q3 = df_mem['mem_usage_percent'].quantile(0.75)
    
    ax1.axhline(q1, color='green', linestyle='-.', 
              label=f'Q1: {q1:.2f}%')
    ax1.axhline(q2, color='blue', linestyle='-.', 
              label=f'Q2 (Median): {q2:.2f}%')
    ax1.axhline(q3, color='purple', linestyle='-.', 
              label=f'Q3: {q3:.2f}%')
    
    # Tambahkan teks statistik sebagai annotasi untuk RAM
    mem_stats_text = (
        f"Statistik RAM:\n"
        f"Mean Usage: {mean_val:.2f}%\n"
        f"Q1 (25%): {q1:.2f}%\n"
        f"Median (Q2): {q2:.2f}%\n"
        f"Q3 (75%): {q3:.2f}%\n"
        f"IQR: {(q3-q1):.2f}%\n"
        f"Total RAM: {df_mem['mem_total'].iloc[0]/1048576:.2f} GB\n"
        f"Peak Usage: {df_mem['mem_used'].max()/1048576:.2f} GB"
    )
    
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    ax1.text(0.02, 0.95, mem_stats_text, transform=ax1.transAxes, fontsize=9,
           verticalalignment='top', bbox=props)
    
    ax1.legend(loc='upper right')
    
    # Plot untuk Swap Memory
    ax2 = axes[1]
    
    # Hitung persentase penggunaan swap
    if not df_swap.empty:
        df_swap['swap_usage_percent'] = (df_swap['mem_used'] / df_swap['mem_total']) * 100 if df_swap['mem_total'].max() > 0 else 0
        
        # Plot penggunaan swap vs waktu
        ax2.plot(df_swap['created_at'], df_swap['swap_usage_percent'], marker='o', linestyle='-', color='orange', 
                label='Swap Usage %')
        ax2.set_title('Penggunaan Swap Memory')
        ax2.set_xlabel('Waktu')
        ax2.set_ylabel('Penggunaan (%)')
        ax2.grid(True, alpha=0.3)
        
        # Tambahkan rata-rata sebagai garis horizontal untuk swap
        swap_mean_val = df_swap['swap_usage_percent'].mean()
        ax2.axhline(swap_mean_val, color='red', linestyle='--', 
                  label=f'Mean: {swap_mean_val:.2f}%')
        
        # Tambahkan Q1, Q2, Q3 untuk swap
        swap_q1 = df_swap['swap_usage_percent'].quantile(0.25)
        swap_q2 = df_swap['swap_usage_percent'].median()
        swap_q3 = df_swap['swap_usage_percent'].quantile(0.75)
        
        ax2.axhline(swap_q1, color='green', linestyle='-.', 
                  label=f'Q1: {swap_q1:.2f}%')
        ax2.axhline(swap_q2, color='blue', linestyle='-.', 
                  label=f'Q2 (Median): {swap_q2:.2f}%')
        ax2.axhline(swap_q3, color='purple', linestyle='-.', 
                  label=f'Q3: {swap_q3:.2f}%')
        
        # Tambahkan teks statistik sebagai annotasi untuk swap
        swap_stats_text = (
            f"Statistik Swap:\n"
            f"Mean Usage: {swap_mean_val:.2f}%\n"
            f"Q1 (25%): {swap_q1:.2f}%\n"
            f"Median (Q2): {swap_q2:.2f}%\n"
            f"Q3 (75%): {swap_q3:.2f}%\n"
            f"IQR: {(swap_q3-swap_q1):.2f}%\n"
            f"Total Swap: {df_swap['mem_total'].iloc[0]/1048576:.2f} GB\n"
            f"Peak Usage: {df_swap['mem_used'].max()/1048576:.2f} GB"
        )
        
        props = dict(boxstyle='round', facecolor='moccasin', alpha=0.5)
        ax2.text(0.02, 0.95, swap_stats_text, transform=ax2.transAxes, fontsize=9,
               verticalalignment='top', bbox=props)
        
        ax2.legend(loc='upper right')
    else:
        ax2.text(0.5, 0.5, 'No Swap Memory Data Available', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax2.transAxes, fontsize=12)
    
    plt.tight_layout()
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png', dpi=150, bbox_inches='tight')
    img_data.seek(0)
    plt.close()

    return img_data

def plot_memory_distribution(df_mem, df_swap):
    """
    Membuat plot distribusi untuk parameter memori penting
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    
    # Plot distribusi untuk RAM
    # RAM Used
    ax1 = axes[0, 0]
    df_mem['mem_used_gb'] = df_mem['mem_used'] / 1048576  # Convert to GB
    df_mem['mem_used_gb'].plot(kind='kde', ax=ax1, color='blue', linewidth=2,
                             title='Distribusi RAM Used (GB)')
    
    # Hitung statistik
    mean_val = df_mem['mem_used_gb'].mean()
    median_val = df_mem['mem_used_gb'].median()
    q1 = df_mem['mem_used_gb'].quantile(0.25)
    q3 = df_mem['mem_used_gb'].quantile(0.75)
    
    # Tambahkan garis vertikal untuk statistik
    ax1.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f} GB')
    ax1.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f} GB')
    ax1.axvline(q1, color='orange', linestyle='-.', label=f'Q1: {q1:.2f} GB')
    ax1.axvline(q3, color='purple', linestyle='-.', label=f'Q3: {q3:.2f} GB')
    
    ax1.set_xlabel('RAM Used (GB)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # RAM Available
    ax2 = axes[0, 1]
    df_mem['mem_available_gb'] = df_mem['mem_available'] / 1048576  # Convert to GB
    df_mem['mem_available_gb'].plot(kind='kde', ax=ax2, color='green', linewidth=2,
                                  title='Distribusi RAM Available (GB)')
    
    # Hitung statistik
    mean_val = df_mem['mem_available_gb'].mean()
    median_val = df_mem['mem_available_gb'].median()
    q1 = df_mem['mem_available_gb'].quantile(0.25)
    q3 = df_mem['mem_available_gb'].quantile(0.75)
    
    # Tambahkan garis vertikal untuk statistik
    ax2.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f} GB')
    ax2.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f} GB')
    ax2.axvline(q1, color='orange', linestyle='-.', label=f'Q1: {q1:.2f} GB')
    ax2.axvline(q3, color='purple', linestyle='-.', label=f'Q3: {q3:.2f} GB')
    
    ax2.set_xlabel('RAM Available (GB)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Plot untuk Cache dan Swap jika data tersedia
    if not df_mem.empty:
        # RAM Cache
        ax3 = axes[1, 0]
        df_mem['mem_cache_gb'] = df_mem['mem_cache'] / 1048576  # Convert to GB
        df_mem['mem_cache_gb'].plot(kind='kde', ax=ax3, color='purple', linewidth=2,
                                title='Distribusi RAM Cache (GB)')
        
        # Hitung statistik
        mean_val = df_mem['mem_cache_gb'].mean()
        median_val = df_mem['mem_cache_gb'].median()
        q1 = df_mem['mem_cache_gb'].quantile(0.25)
        q3 = df_mem['mem_cache_gb'].quantile(0.75)
        
        # Tambahkan garis vertikal untuk statistik
        ax3.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f} GB')
        ax3.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f} GB')
        ax3.axvline(q1, color='orange', linestyle='-.', label=f'Q1: {q1:.2f} GB')
        ax3.axvline(q3, color='purple', linestyle='-.', label=f'Q3: {q3:.2f} GB')
        
        ax3.set_xlabel('Cache (GB)')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
    else:
        ax3 = axes[1, 0]
        ax3.text(0.5, 0.5, 'No Cache Data Available', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax3.transAxes, fontsize=12)
    
    # Swap Used (if available)
    ax4 = axes[1, 1]
    if not df_swap.empty and df_swap['mem_total'].max() > 0:
        df_swap['swap_used_gb'] = df_swap['mem_used'] / 1048576
        if df_swap['swap_used_gb'].nunique() > 1:
            df_swap['swap_used_gb'].plot(kind='kde', ax=ax4, color='orange', linewidth=2,
                                        title='Distribusi Swap Used (GB)')
            mean_val = df_swap['swap_used_gb'].mean()
            median_val = df_swap['swap_used_gb'].median()
            q1 = df_swap['swap_used_gb'].quantile(0.25)
            q3 = df_swap['swap_used_gb'].quantile(0.75)
            
            ax4.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f} GB')
            ax4.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f} GB')
            ax4.axvline(q1, color='orange', linestyle='-.', label=f'Q1: {q1:.2f} GB')
            ax4.axvline(q3, color='purple', linestyle='-.', label=f'Q3: {q3:.2f} GB')
            
            ax4.set_xlabel('Swap Used (GB)')
            ax4.legend(loc='upper right')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Swap Data Tidak Bervariasi',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax4.transAxes, fontsize=12)
    else:
        ax4.text(0.5, 0.5, 'No Swap Data Available',
                horizontalalignment='center', verticalalignment='center',
                transform=ax4.transAxes, fontsize=12)

    plt.tight_layout()
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png', dpi=150, bbox_inches='tight')
    img_data.seek(0)
    plt.close()

    return img_data

def MemoryDistributionPlot(elements, img_data, title):
    page_width, _ = A4
    
    # Tambahkan judul untuk bagian plot jika diperlukan
    elements = Title(elements, title)
    elements.append(Spacer(1, 12))
    
    # Buat objek Image dari data gambar
    img = Image(img_data, width=page_width*0.8, height=page_width*0.8)
    
    # Bungkus gambar dengan Table untuk mengatur posisi
    outer_table = Table([[img]])
    outer_table.setStyle(TableStyle([
        ('LEFTPADDING', (0, 0), (-1, -1), 0),  # Margin kiri
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    
    elements.append(outer_table)
    elements.append(Spacer(1, 20))
    return elements

def Title(elements, word):
    styles = getSampleStyleSheet()
    if 'HeaderStyle' not in styles:
        styles.add(ParagraphStyle(
            name='HeaderStyle',
            fontName='Helvetica-Bold',
            fontSize=14,
            leading=16,
            alignment=0,  # 0=left, 1=center, 2=right
            leftIndent=0,
            rightIndent=0,
        ))
    elements.append(Paragraph(word, styles['HeaderStyle']))
    elements.append(Spacer(1, 6))
    return elements

def MemoryConclusionTable(elements, df_mem, df_swap):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

    page_width, _ = A4

    total_ram_gb = df_mem['mem_total'].iloc[0] / 1048576
    avg_used_ram_gb = df_mem['mem_used'].mean() / 1048576
    avg_used_ram_percent = (df_mem['mem_used'].mean() / df_mem['mem_total'].mean()) * 100
    peak_used_ram_gb = df_mem['mem_used'].max() / 1048576
    peak_used_ram_percent = (df_mem['mem_used'].max() / df_mem['mem_total'].max()) * 100
    avg_available_ram_gb = df_mem['mem_available'].mean() / 1048576
    avg_cache_ram_gb = df_mem['mem_cache'].mean() / 1048576

    title_ram = ["Kesimpulan Analisis RAM Memory"]
    data_ram = [title_ram]
    data_ram.append(["Parameter", "Nilai"])
    data_ram.append(["Total RAM", f"{total_ram_gb:.2f} GB"])
    data_ram.append(["Rata-rata Penggunaan RAM", f"{avg_used_ram_gb:.2f} GB ({avg_used_ram_percent:.2f}%)"])
    data_ram.append(["Penggunaan RAM Tertinggi", f"{peak_used_ram_gb:.2f} GB ({peak_used_ram_percent:.2f}%)"])
    data_ram.append(["Rata-rata RAM Tersedia", f"{avg_available_ram_gb:.2f} GB"])
    data_ram.append(["Rata-rata Cache", f"{avg_cache_ram_gb:.2f} GB"])

    if avg_used_ram_percent > 80:
        ram_status = "KRITIS: Penggunaan RAM sangat tinggi, risiko sistem tidak stabil"
    elif avg_used_ram_percent > 60:
        ram_status = "PERHATIAN: Penggunaan RAM tinggi, pertimbangkan upgrade atau optimasi"
    else:
        ram_status = "NORMAL: Penggunaan RAM dalam batas aman"

    data_ram.append(["Status RAM", ram_status])

    if not df_swap.empty and df_swap['mem_total'].max() > 0:
        total_swap_gb = df_swap['mem_total'].iloc[0] / 1048576
        avg_used_swap_gb = df_swap['mem_used'].mean() / 1048576
        avg_used_swap_percent = (df_swap['mem_used'].mean() / df_swap['mem_total'].mean()) * 100
        peak_used_swap_gb = df_swap['mem_used'].max() / 1048576
        peak_used_swap_percent = (df_swap['mem_used'].max() / df_swap['mem_total'].max()) * 100

        title_swap = ["Kesimpulan Analisis Swap Memory"]
        data_swap = [title_swap]
        data_swap.append(["Parameter", "Nilai"])
        data_swap.append(["Total Swap", f"{total_swap_gb:.2f} GB"])
        data_swap.append(["Rata-rata Penggunaan Swap", f"{avg_used_swap_gb:.2f} GB ({avg_used_swap_percent:.2f}%)"])
        data_swap.append(["Penggunaan Swap Tertinggi", f"{peak_used_swap_gb:.2f} GB ({peak_used_swap_percent:.2f}%)"])

        if avg_used_swap_percent > 50:
            swap_status = "KRITIS: Penggunaan Swap tinggi, kinerja sistem menurun signifikan"
        elif avg_used_swap_percent > 20:
            swap_status = "PERHATIAN: Penggunaan Swap moderat, indikasi RAM tidak cukup"
        elif avg_used_swap_percent > 0:
            swap_status = "PERHATIAN RINGAN: Swap digunakan, monitor penggunaan RAM"
        else:
            swap_status = "NORMAL: Swap tidak digunakan"

        data_swap.append(["Status Swap", swap_status])

    title_rekomendasi = ["Rekomendasi"]
    data_rekomendasi = [title_rekomendasi]
    data_rekomendasi.append(["Aspek", "Rekomendasi"])

    if avg_used_ram_percent > 80:
        ram_rekomendasi = "Segera tambah RAM atau optimasi penggunaan dengan mengurangi jumlah layanan atau aplikasi yang berjalan"
    elif avg_used_ram_percent > 60:
        ram_rekomendasi = "Monitor penggunaan RAM secara berkala, pertimbangkan upgrade jika tren penggunaan meningkat"
    else:
        ram_rekomendasi = "Penggunaan RAM optimal, tidak perlu tindakan khusus"

    data_rekomendasi.append(["RAM", ram_rekomendasi])

    if not df_swap.empty and df_swap['mem_total'].max() > 0:
        if avg_used_swap_percent > 20:
            swap_rekomendasi = "Tambah RAM fisik untuk mengurangi ketergantungan pada Swap"
        elif avg_used_swap_percent > 0:
            swap_rekomendasi = "Monitor penggunaan RAM dan Swap, identifikasi aplikasi yang banyak menggunakan memori"
        else:
            swap_rekomendasi = "Penggunaan Swap optimal, tidak perlu tindakan khusus"

        data_rekomendasi.append(["Swap", swap_rekomendasi])

    cache_to_ram_ratio = (avg_cache_ram_gb / total_ram_gb) * 100
    if cache_to_ram_ratio > 40:
        cache_rekomendasi = "Cache menggunakan porsi RAM yang besar, pertimbangkan untuk menyesuaikan parameter cache sistem"
    else:
        cache_rekomendasi = "Penggunaan cache sesuai ekspektasi"

    data_rekomendasi.append(["Cache", cache_rekomendasi])

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


    elements.append(create_table(data_ram))
    elements.append(Spacer(1, 15))

    if not df_swap.empty and df_swap['mem_total'].max() > 0:
        elements.append(create_table(data_swap))
        elements.append(Spacer(1, 15))

    elements.append(create_table(data_rekomendasi))
    elements.append(Spacer(1, 15))


    return elements


def ProcessMemoryData(memory_data):
    try:
        start_time = time.time()
        
        logger.info("Memproses data memori firewall...")
        column_names = ['mem_type', 'mem_total', 
                       'mem_used', 'mem_free', 'mem_shared', 'mem_cache', 
                       'mem_available', 'created_at']

        # Convert data to regular Python lists or tuples if needed
        data_list = [tuple(row) for row in memory_data]
        
        # Create DataFrame from the data
        df = pd.DataFrame(data_list, columns=column_names)
        
        # Pisahkan data berdasarkan mem_type
        df_mem = df[df['mem_type'] == 'Mem'].copy().reset_index(drop=True)
        df_swap = df[df['mem_type'] == 'Swap'].copy().reset_index(drop=True)
        
        # Generate plots and get image data
        usage_img_data = plot_memory_usage(df_mem, df_swap)
        distribution_img_data = plot_memory_distribution(df_mem, df_swap)
        
        return usage_img_data, distribution_img_data, df_mem, df_swap
        
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error dalam pembuatan analisis memori: {str(e)}\n{tb}")
        return None, None, None, None

def GlobalHandler(elements, memory_data):
    try:
        # Dapatkan data gambar dan DataFrame dari ProcessMemoryData
        usage_img_data, distribution_img_data, df_mem, df_swap = ProcessMemoryData(memory_data)
        
        if usage_img_data and distribution_img_data and df_mem is not None:
            elements.append(PageBreak())
            # Tambahkan plot penggunaan memori ke elements PDF
            elements = MemoryDistributionPlot(elements, usage_img_data, "Analisis Penggunaan Memori")
            
            # Tambahkan plot distribusi memori ke elements PDF
            elements.append(PageBreak())
            elements = MemoryDistributionPlot(elements, distribution_img_data, "Distribusi Parameter Memori")
            
            # Tambahkan kesimpulan analisis
            elements.append(PageBreak())
            elements = Title(elements, "Kesimpulan dan Rekomendasi")
            elements.append(Spacer(1, 10))
            elements = MemoryConclusionTable(elements, df_mem, df_swap)
        else:
            # Jika ada error, tambahkan pesan error ke PDF
            elements.append(Paragraph("Error: Tidak dapat membuat analisis memori", getSampleStyleSheet()['Normal']))
            elements.append(Spacer(1, 20))
            
        return elements
        
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error dalam integrasi ke PDF: {str(e)}\n{tb}")
        
        elements.append(Paragraph(f"Error: {str(e)}", getSampleStyleSheet()['Normal']))
        elements.append(Spacer(1, 20))
        return elements

# Contoh penggunaan di script utama:
