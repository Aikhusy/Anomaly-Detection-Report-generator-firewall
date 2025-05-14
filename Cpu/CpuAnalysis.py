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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_cpu_usage(df_cpu):
    """
    Plot CPU usage metrics over time with statistical annotations
    """
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 12))
    
    # Plot for CPU Usage Percentage
    ax1 = axes[0]
    
    # Plot CPU usage vs time
    ax1.plot(df_cpu['created_at'], df_cpu['fw_cpu_usage_percentage'], marker='o', linestyle='-', color='blue', 
            label='CPU Usage %')
    ax1.plot(df_cpu['created_at'], df_cpu['fw_cpu_user_time_percentage'], marker='x', linestyle='-', color='green', 
            label='User Time %')
    ax1.plot(df_cpu['created_at'], df_cpu['fw_cpu_system_time_percentage'], marker='^', linestyle='-', color='orange', 
            label='System Time %')
    
    ax1.set_title('CPU Usage Over Time')
    ax1.set_xlabel('Waktu')
    ax1.set_ylabel('Persentase (%)')
    ax1.grid(True, alpha=0.3)
    
    # Add average as horizontal line
    mean_usage = df_cpu['fw_cpu_usage_percentage'].mean()
    ax1.axhline(mean_usage, color='red', linestyle='--', 
              label=f'Mean Usage: {mean_usage:.2f}%')
    
    # Add Q1, Q2, Q3
    q1 = df_cpu['fw_cpu_usage_percentage'].quantile(0.25)
    q2 = df_cpu['fw_cpu_usage_percentage'].median()
    q3 = df_cpu['fw_cpu_usage_percentage'].quantile(0.75)
    
    ax1.axhline(q1, color='green', linestyle='-.', 
              label=f'Q1: {q1:.2f}%')
    ax1.axhline(q2, color='blue', linestyle='-.', 
              label=f'Q2 (Median): {q2:.2f}%')
    ax1.axhline(q3, color='purple', linestyle='-.', 
              label=f'Q3: {q3:.2f}%')
    
    # Add statistics text annotation
    cpu_stats_text = (
        f"Statistik CPU Usage:\n"
        f"Mean: {mean_usage:.2f}%\n"
        f"Q1 (25%): {q1:.2f}%\n"
        f"Median (Q2): {q2:.2f}%\n"
        f"Q3 (75%): {q3:.2f}%\n"
        f"IQR: {(q3-q1):.2f}%\n"
        f"Min: {df_cpu['fw_cpu_usage_percentage'].min():.2f}%\n"
        f"Max: {df_cpu['fw_cpu_usage_percentage'].max():.2f}%\n"
        f"CPU Cores: {int(df_cpu['fw_cpu_number'].iloc[0])}"
    )
    
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    ax1.text(0.02, 0.95, cpu_stats_text, transform=ax1.transAxes, fontsize=9,
           verticalalignment='top', bbox=props)
    
    ax1.legend(loc='upper right')
    
    # Plot for CPU Queue Length and Interrupts
    ax2 = axes[1]
    
    # Create twin axis for different scales
    ax2_twin = ax2.twinx()
    
    # Plot queue length
    line1 = ax2.plot(df_cpu['created_at'], df_cpu['fw_cpu_queue_length'], marker='o', linestyle='-', color='blue', 
            label='Queue Length')
    ax2.set_ylabel('Queue Length', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    # Plot interrupts per sec on twin axis
    line2 = ax2_twin.plot(df_cpu['created_at'], df_cpu['fw_cpu_interrupt_per_sec'], marker='s', linestyle='-', color='red', 
                         label='Interrupts/sec')
    ax2_twin.set_ylabel('Interrupts per Second', color='red')
    ax2_twin.tick_params(axis='y', labelcolor='red')
    
    ax2.set_title('CPU Queue Length and Interrupts')
    ax2.set_xlabel('Waktu')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics annotations
    queue_mean = df_cpu['fw_cpu_queue_length'].mean()
    queue_max = df_cpu['fw_cpu_queue_length'].max()
    int_mean = df_cpu['fw_cpu_interrupt_per_sec'].mean()
    int_max = df_cpu['fw_cpu_interrupt_per_sec'].max()
    
    queue_stats_text = (
        f"Queue Length Stats:\n"
        f"Mean: {queue_mean:.2f}\n"
        f"Max: {queue_max:.2f}\n\n"
        f"Interrupts Stats:\n"
        f"Mean: {int_mean:.2f}/sec\n"
        f"Max: {int_max:.2f}/sec"
    )
    
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.5)
    ax2.text(0.02, 0.95, queue_stats_text, transform=ax2.transAxes, fontsize=9,
           verticalalignment='top', bbox=props)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right')
    
    plt.tight_layout()
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png', dpi=150, bbox_inches='tight')
    img_data.seek(0)
    plt.close()

    return img_data

def plot_cpu_distribution(df_cpu):
    """
    Plot distribution of CPU metrics
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    
    # Plot distribution for CPU Usage Percentage
    ax1 = axes[0, 0]
    df_cpu['fw_cpu_usage_percentage'].plot(kind='kde', ax=ax1, color='blue', linewidth=2,
                             title='Distribusi CPU Usage (%)')
    
    # Calculate statistics
    mean_val = df_cpu['fw_cpu_usage_percentage'].mean()
    median_val = df_cpu['fw_cpu_usage_percentage'].median()
    q1 = df_cpu['fw_cpu_usage_percentage'].quantile(0.25)
    q3 = df_cpu['fw_cpu_usage_percentage'].quantile(0.75)
    
    # Add vertical lines for statistics
    ax1.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}%')
    ax1.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}%')
    ax1.axvline(q1, color='orange', linestyle='-.', label=f'Q1: {q1:.2f}%')
    ax1.axvline(q3, color='purple', linestyle='-.', label=f'Q3: {q3:.2f}%')
    
    ax1.set_xlabel('CPU Usage (%)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot distribution for CPU User Time
    ax2 = axes[0, 1]
    df_cpu['fw_cpu_user_time_percentage'].plot(kind='kde', ax=ax2, color='green', linewidth=2,
                                  title='Distribusi CPU User Time (%)')
    
    # Calculate statistics
    mean_val = df_cpu['fw_cpu_user_time_percentage'].mean()
    median_val = df_cpu['fw_cpu_user_time_percentage'].median()
    q1 = df_cpu['fw_cpu_user_time_percentage'].quantile(0.25)
    q3 = df_cpu['fw_cpu_user_time_percentage'].quantile(0.75)
    
    # Add vertical lines for statistics
    ax2.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}%')
    ax2.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}%')
    ax2.axvline(q1, color='orange', linestyle='-.', label=f'Q1: {q1:.2f}%')
    ax2.axvline(q3, color='purple', linestyle='-.', label=f'Q3: {q3:.2f}%')
    
    ax2.set_xlabel('CPU User Time (%)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Plot distribution for CPU System Time
    ax3 = axes[1, 0]
    df_cpu['fw_cpu_system_time_percentage'].plot(kind='kde', ax=ax3, color='orange', linewidth=2,
                                title='Distribusi CPU System Time (%)')
    
    # Calculate statistics
    mean_val = df_cpu['fw_cpu_system_time_percentage'].mean()
    median_val = df_cpu['fw_cpu_system_time_percentage'].median()
    q1 = df_cpu['fw_cpu_system_time_percentage'].quantile(0.25)
    q3 = df_cpu['fw_cpu_system_time_percentage'].quantile(0.75)
    
    # Add vertical lines for statistics
    ax3.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}%')
    ax3.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}%')
    ax3.axvline(q1, color='orange', linestyle='-.', label=f'Q1: {q1:.2f}%')
    ax3.axvline(q3, color='purple', linestyle='-.', label=f'Q3: {q3:.2f}%')
    
    ax3.set_xlabel('CPU System Time (%)')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Plot distribution for CPU Idle Time
    ax4 = axes[1, 1]
    df_cpu['fw_cpu_idle_time_percentage'].plot(kind='kde', ax=ax4, color='purple', linewidth=2,
                                title='Distribusi CPU Idle Time (%)')
    
    # Calculate statistics
    mean_val = df_cpu['fw_cpu_idle_time_percentage'].mean()
    median_val = df_cpu['fw_cpu_idle_time_percentage'].median()
    q1 = df_cpu['fw_cpu_idle_time_percentage'].quantile(0.25)
    q3 = df_cpu['fw_cpu_idle_time_percentage'].quantile(0.75)
    
    # Add vertical lines for statistics
    ax4.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}%')
    ax4.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}%')
    ax4.axvline(q1, color='orange', linestyle='-.', label=f'Q1: {q1:.2f}%')
    ax4.axvline(q3, color='purple', linestyle='-.', label=f'Q3: {q3:.2f}%')
    
    ax4.set_xlabel('CPU Idle Time (%)')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png', dpi=150, bbox_inches='tight')
    img_data.seek(0)
    plt.close()

    return img_data

def Title(elements, word):
    """
    Add a title to PDF elements
    """
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

def CpuDistributionPlot(elements, img_data, title):
    """
    Add a CPU distribution plot to PDF elements
    """
    page_width, _ = A4
    
    # Add title for plot section if needed
    elements = Title(elements, title)
    elements.append(Spacer(1, 12))
    
    # Create Image object from image data
    img = Image(img_data, width=page_width*0.8, height=page_width*0.8)
    
    # Wrap image with Table to position it
    outer_table = Table([[img]])
    outer_table.setStyle(TableStyle([
        ('LEFTPADDING', (0, 0), (-1, -1), 0),  # Left margin
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    
    elements.append(outer_table)
    elements.append(Spacer(1, 20))
    return elements

def generate_cpu_recommendations(df_cpu):
    """
    Generate CPU usage recommendations based on analysis
    """
    recommendations = {}
    
    # Get key metrics for analysis
    avg_usage = df_cpu['fw_cpu_usage_percentage'].mean()
    max_usage = df_cpu['fw_cpu_usage_percentage'].max()
    percent_high_usage = len(df_cpu[df_cpu['fw_cpu_usage_percentage'] > 80]) / len(df_cpu) * 100
    avg_queue = df_cpu['fw_cpu_queue_length'].mean()
    max_queue = df_cpu['fw_cpu_queue_length'].max()
    cpu_cores = int(df_cpu['fw_cpu_number'].iloc[0])
    
    # Usage recommendation
    if avg_usage < 30:
        recommendations['usage'] = "Penggunaan CPU sangat rendah. Anda dapat mempertimbangkan untuk mengurangi jumlah CPU atau mengkonsolidasikan beban kerja untuk menghemat sumber daya."
    elif avg_usage < 60:
        recommendations['usage'] = "Penggunaan CPU dalam batas normal. Sistem beroperasi dalam kisaran yang efisien."
    elif avg_usage < 80:
        recommendations['usage'] = "Penggunaan CPU cukup tinggi. Pantau kinerja sistem untuk memastikan tidak ada masalah yang berkembang."
    else:
        recommendations['usage'] = "Penggunaan CPU sangat tinggi. Disarankan untuk meningkatkan kapasitas CPU atau mengoptimalkan proses untuk mengurangi beban."
    
    # Queue length recommendation
    if avg_queue < 1:
        recommendations['queue'] = "Panjang antrian CPU sangat baik. Tidak ada tanda bottleneck pemrosesan."
    elif avg_queue < 2:
        recommendations['queue'] = "Panjang antrian CPU normal. Sistem menangani beban dengan baik."
    elif avg_queue < 5:
        recommendations['queue'] = "Panjang antrian CPU moderat. Pantau aplikasi yang berjalan untuk kemungkinan optimasi."
    else:
        recommendations['queue'] = "Panjang antrian CPU tinggi. Ini menunjukkan adanya bottleneck CPU. Pertimbangkan untuk menambah kapasitas CPU atau mengurangi beban kerja."
    
    # System/User time ratio recommendation
    avg_system = df_cpu['fw_cpu_system_time_percentage'].mean()
    avg_user = df_cpu['fw_cpu_user_time_percentage'].mean()
    
    if avg_system / (avg_system + avg_user) > 0.4:
        recommendations['system_time'] = "Rasio waktu sistem terhadap total waktu CPU cukup tinggi. Ini bisa menunjukkan banyak operasi I/O, sistem panggilan, atau kesalahan page fault. Periksa driver dan optimasi kernel."
    else:
        recommendations['system_time'] = "Rasio waktu sistem terhadap total waktu CPU normal. Menunjukkan keseimbangan yang baik antara kode user space dan operasi sistem."
    
    # Interrupt rate recommendation
    avg_interrupt = df_cpu['fw_cpu_interrupt_per_sec'].mean()
    if avg_interrupt > 10000:
        recommendations['interrupts'] = "Tingkat interrupt sangat tinggi. Periksa perangkat yang mungkin menghasilkan interrupt berlebihan dan pertimbangkan penggunaan mekanisme seperti interrupt coalescing."
    elif avg_interrupt > 5000:
        recommendations['interrupts'] = "Tingkat interrupt moderat hingga tinggi. Pantau kinerja perangkat jaringan dan disk untuk kemungkinan optimasi."
    else:
        recommendations['interrupts'] = "Tingkat interrupt dalam batas normal."
    
    # Overall system status
    if max_usage > 95 and max_queue > 5:
        recommendations['overall'] = "Sistem menunjukkan gejala kekurangan sumber daya CPU. Disarankan untuk meng-upgrade hardware atau mengoptimalkan beban kerja."
    elif percent_high_usage > 20:
        recommendations['overall'] = "Sistem mengalami periode penggunaan CPU tinggi. Pertimbangkan untuk menambah kapasitas jika tren ini berlanjut."
    elif avg_usage < 20:
        recommendations['overall'] = "Sistem tampaknya memiliki sumber daya CPU berlebih. Pertimbangkan untuk melakukan konsolidasi beban kerja."
    else:
        recommendations['overall'] = "Sistem menunjukkan penggunaan CPU yang seimbang dan sehat."
    
    return recommendations

def CpuConclusionTable(elements, df_cpu):
    """
    Create a table with CPU analysis conclusions and recommendations
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

    page_width, _ = A4

    # Calculate key metrics
    total_cpu_cores = int(df_cpu['fw_cpu_number'].iloc[0])
    avg_usage = df_cpu['fw_cpu_usage_percentage'].mean()
    max_usage = df_cpu['fw_cpu_usage_percentage'].max()
    avg_user = df_cpu['fw_cpu_user_time_percentage'].mean()
    avg_system = df_cpu['fw_cpu_system_time_percentage'].mean()
    avg_idle = df_cpu['fw_cpu_idle_time_percentage'].mean()
    avg_queue = df_cpu['fw_cpu_queue_length'].mean()
    max_queue = df_cpu['fw_cpu_queue_length'].max()
    avg_interrupts = df_cpu['fw_cpu_interrupt_per_sec'].mean()
    max_interrupts = df_cpu['fw_cpu_interrupt_per_sec'].max()

    # Generate CPU data table
    title_cpu = ["Kesimpulan Analisis CPU"]
    data_cpu = [title_cpu]
    data_cpu.append(["Parameter", "Nilai"])
    data_cpu.append(["Total CPU Cores", f"{total_cpu_cores}"])
    data_cpu.append(["Rata-rata Penggunaan CPU", f"{avg_usage:.2f}%"])
    data_cpu.append(["Penggunaan CPU Tertinggi", f"{max_usage:.2f}%"])
    data_cpu.append(["Rata-rata User Time", f"{avg_user:.2f}%"])
    data_cpu.append(["Rata-rata System Time", f"{avg_system:.2f}%"])
    data_cpu.append(["Rata-rata Idle Time", f"{avg_idle:.2f}%"])
    data_cpu.append(["Rata-rata Queue Length", f"{avg_queue:.2f}"])
    data_cpu.append(["Queue Length Tertinggi", f"{max_queue:.2f}"])
    data_cpu.append(["Rata-rata Interrupts per second", f"{avg_interrupts:.2f}"])
    data_cpu.append(["Interrupts per second Tertinggi", f"{max_interrupts:.2f}"])

    # Generate recommendations
    recommendations = generate_cpu_recommendations(df_cpu)
    
    # Create recommendations table
    title_rekomendasi = ["Rekomendasi CPU"]
    data_rekomendasi = [title_rekomendasi]
    data_rekomendasi.append(["Aspek", "Rekomendasi"])
    data_rekomendasi.append(["Penggunaan CPU", recommendations['usage']])
    data_rekomendasi.append(["Queue Length", recommendations['queue']])
    data_rekomendasi.append(["System Time", recommendations['system_time']])
    data_rekomendasi.append(["Interrupts", recommendations['interrupts']])
    data_rekomendasi.append(["Status Keseluruhan", recommendations['overall']])

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

    # Add tables to elements
    elements.append(create_table(data_cpu))
    elements.append(Spacer(1, 15))
    elements.append(create_table(data_rekomendasi))
    elements.append(Spacer(1, 15))

    return elements

def ProcessCpuData(cpu_data):
    """
    Process CPU data and generate analysis visualizations
    """
    try:
        start_time = time.time()
        
        logger.info("Memproses data CPU firewall...")
        column_names = ['fw_cpu_user_time_percentage',
                       'fw_cpu_system_time_percentage',
                       'fw_cpu_idle_time_percentage',
                       'fw_cpu_usage_percentage',
                       'fw_cpu_queue_length',
                       'fw_cpu_interrupt_per_sec',
                       'fw_cpu_number',
                       'created_at']

        # Convert data to regular Python lists or tuples if needed
        data_list = [tuple(row) for row in cpu_data]
        
        # Create DataFrame from the data
        df = pd.DataFrame(data_list, columns=column_names)
        
        # Make sure numeric columns are float type
        numeric_cols = [col for col in df.columns if col != 'created_at']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Generate plots and get image data
        usage_img_data = plot_cpu_usage(df)
        distribution_img_data = plot_cpu_distribution(df)
        
        logger.info(f"Processing CPU data completed in {time.time() - start_time:.2f} seconds")
        return usage_img_data, distribution_img_data, df
        
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error dalam pembuatan analisis CPU: {str(e)}\n{tb}")
        return None, None, None

def GlobalHandler(elements, cpu_data):
    """
    Main handler for CPU analysis integration
    """
    try:
        usage_img_data, distribution_img_data, df_cpu = ProcessCpuData(cpu_data)
        
        if usage_img_data and distribution_img_data and df_cpu is not None:
            elements.append(PageBreak())
            # Add CPU usage plot to PDF elements
            elements = CpuDistributionPlot(elements, usage_img_data, "Analisis Penggunaan CPU")
            
            # Add CPU distribution plot to PDF elements
            elements.append(PageBreak())
            elements = CpuDistributionPlot(elements, distribution_img_data, "Distribusi Parameter CPU")
            
            # Add analysis conclusions
            elements.append(PageBreak())
            elements = Title(elements, "Kesimpulan dan Rekomendasi CPU")
            elements.append(Spacer(1, 10))
            elements = CpuConclusionTable(elements, df_cpu)
        else:
            # If there's an error, add error message to PDF
            elements.append(Paragraph("Error: Tidak dapat membuat analisis CPU", getSampleStyleSheet()['Normal']))
            elements.append(Spacer(1, 20))
            
        return elements
        
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error dalam integrasi ke PDF: {str(e)}\n{tb}")
        
        elements.append(Paragraph(f"Error: {str(e)}", getSampleStyleSheet()['Normal']))
        elements.append(Spacer(1, 20))
        return elements