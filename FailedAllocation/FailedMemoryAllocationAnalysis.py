def plot_allocation_stats(df_alloc):
    """
    Create plots for memory allocation statistics.
    
    Args:
        df_alloc: DataFrame containing memory allocation data
        
    Returns:
        BytesIO object containing the plot image
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import io
    from matplotlib.ticker import FuncFormatter
    
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 15))
    
    # 1. Plot Total Memory and Peak Memory
    ax1 = axes[0]
    ax1.plot(df_alloc['created_at'], df_alloc['fw_total_memory']/1024/1024, 
             marker='o', linestyle='-', color='blue', label='Total Memory')
    ax1.plot(df_alloc['created_at'], df_alloc['fw_peak_memory']/1024/1024, 
             marker='s', linestyle='-', color='red', label='Peak Memory')
    
    # Format y-axis to show values in MB
    def mb_formatter(x, pos):
        return f'{x:.1f} MB'
    
    ax1.yaxis.set_major_formatter(FuncFormatter(mb_formatter))
    
    ax1.set_title('Firewall Memory Usage (Total vs Peak)')
    ax1.set_xlabel('Waktu')
    ax1.set_ylabel('Memory (MB)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Add statistics box
    total_mem_mean = df_alloc['fw_total_memory'].mean()/1024/1024
    peak_mem_mean = df_alloc['fw_peak_memory'].mean()/1024/1024
    mem_diff = peak_mem_mean - total_mem_mean
    mem_util_percent = (total_mem_mean / peak_mem_mean) * 100 if peak_mem_mean > 0 else 0
    
    stats_text = (
        f"Statistik Memory Usage:\n"
        f"Avg Total Memory: {total_mem_mean:.2f} MB\n"
        f"Avg Peak Memory: {peak_mem_mean:.2f} MB\n"
        f"Memory Utilization: {mem_util_percent:.2f}%\n"
        f"Available Headroom: {mem_diff:.2f} MB"
    )
    
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=props)
    
    # 2. Plot Allocation and Free Counts
    ax2 = axes[1]
    ax2.plot(df_alloc['created_at'], df_alloc['fw_total_alloc'], 
             marker='o', linestyle='-', color='green', label='Total Allocations')
    ax2.plot(df_alloc['created_at'], df_alloc['fw_total_free'], 
             marker='s', linestyle='-', color='purple', label='Total Frees')
    
    # Calculate allocation delta (allocations - frees)
    df_alloc['alloc_delta'] = df_alloc['fw_total_alloc'] - df_alloc['fw_total_free']
    ax2.plot(df_alloc['created_at'], df_alloc['alloc_delta'], 
             marker='x', linestyle='--', color='orange', label='Allocation Delta')
    
    ax2.set_title('Memory Allocation and Free Operations')
    ax2.set_xlabel('Waktu')
    ax2.set_ylabel('Count')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # Add statistics box for allocations
    alloc_mean = df_alloc['fw_total_alloc'].mean()
    free_mean = df_alloc['fw_total_free'].mean()
    delta_mean = df_alloc['alloc_delta'].mean()
    alloc_free_ratio = alloc_mean / free_mean if free_mean > 0 else 0
    
    alloc_stats_text = (
        f"Statistik Alokasi:\n"
        f"Avg Allocations: {alloc_mean:.2f}\n"
        f"Avg Frees: {free_mean:.2f}\n"
        f"Avg Delta: {delta_mean:.2f}\n"
        f"Alloc/Free Ratio: {alloc_free_ratio:.4f}"
    )
    
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
    ax2.text(0.02, 0.95, alloc_stats_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=props)
    
    # 3. Plot Failed Allocations and Frees
    ax3 = axes[2]
    
    # Check if there are any failed allocations or frees
    has_failures = (df_alloc['fw_failed_alloc'].sum() > 0) or (df_alloc['fw_failed_free'].sum() > 0)
    
    if has_failures:
        ax3.plot(df_alloc['created_at'], df_alloc['fw_failed_alloc'], 
                 marker='o', linestyle='-', color='red', label='Failed Allocations')
        ax3.plot(df_alloc['created_at'], df_alloc['fw_failed_free'], 
                 marker='s', linestyle='-', color='orange', label='Failed Frees')
        
        ax3.set_title('Failed Memory Operations')
        ax3.set_xlabel('Waktu')
        ax3.set_ylabel('Count')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right')
        
        # Add statistics for failures
        failed_alloc_sum = df_alloc['fw_failed_alloc'].sum()
        failed_free_sum = df_alloc['fw_failed_free'].sum()
        failed_alloc_rate = (failed_alloc_sum / df_alloc['fw_total_alloc'].sum()) * 100 if df_alloc['fw_total_alloc'].sum() > 0 else 0
        failed_free_rate = (failed_free_sum / df_alloc['fw_total_free'].sum()) * 100 if df_alloc['fw_total_free'].sum() > 0 else 0
        
        failure_stats_text = (
            f"Statistik Kegagalan:\n"
            f"Total Failed Allocations: {failed_alloc_sum}\n"
            f"Total Failed Frees: {failed_free_sum}\n"
            f"Failed Allocation Rate: {failed_alloc_rate:.4f}%\n"
            f"Failed Free Rate: {failed_free_rate:.4f}%"
        )
        
        props = dict(boxstyle='round', facecolor='mistyrose', alpha=0.5)
        ax3.text(0.02, 0.95, failure_stats_text, transform=ax3.transAxes, fontsize=9,
                 verticalalignment='top', bbox=props)
    else:
        ax3.text(0.5, 0.5, 'No Failed Memory Operations Detected', 
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax3.transAxes, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png', dpi=150, bbox_inches='tight')
    img_data.seek(0)
    plt.close()
    
    return img_data

def plot_allocation_distribution(df_alloc):
    """
    Create distribution plots for memory allocation metrics.
    
    Args:
        df_alloc: DataFrame containing memory allocation data
        
    Returns:
        BytesIO object containing the plot image
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import io
    from matplotlib.ticker import FuncFormatter
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 12))
    
    # 1. Distribution of Total Memory
    ax1 = axes[0, 0]
    df_alloc['fw_total_memory_mb'] = df_alloc['fw_total_memory'] / 1024 / 1024  # Convert to MB
    
    # Check if there's enough variation for a meaningful distribution
    if df_alloc['fw_total_memory_mb'].nunique() > 1:
        df_alloc['fw_total_memory_mb'].plot(kind='kde', ax=ax1, color='blue', linewidth=2,
                                         title='Distribusi Total Memory')
        ax1.hist(df_alloc['fw_total_memory_mb'], bins=20, alpha=0.3, color='blue', density=True)
        
        # Add statistics
        mean_val = df_alloc['fw_total_memory_mb'].mean()
        median_val = df_alloc['fw_total_memory_mb'].median()
        min_val = df_alloc['fw_total_memory_mb'].min()
        max_val = df_alloc['fw_total_memory_mb'].max()
        
        ax1.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f} MB')
        ax1.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f} MB')
        ax1.axvline(min_val, color='orange', linestyle='-.', label=f'Min: {min_val:.2f} MB')
        ax1.axvline(max_val, color='purple', linestyle='-.', label=f'Max: {max_val:.2f} MB')
        
        ax1.set_xlabel('Total Memory (MB)')
        ax1.legend(loc='upper right')
    else:
        ax1.text(0.5, 0.5, 'Insufficient variation for distribution analysis', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax1.transAxes, fontsize=10)
    
    ax1.grid(True, alpha=0.3)
    
    # 2. Distribution of Peak Memory
    ax2 = axes[0, 1]
    df_alloc['fw_peak_memory_mb'] = df_alloc['fw_peak_memory'] / 1024 / 1024  # Convert to MB
    
    if df_alloc['fw_peak_memory_mb'].nunique() > 1:
        df_alloc['fw_peak_memory_mb'].plot(kind='kde', ax=ax2, color='red', linewidth=2,
                                      title='Distribusi Peak Memory')
        ax2.hist(df_alloc['fw_peak_memory_mb'], bins=20, alpha=0.3, color='red', density=True)
        
        # Add statistics
        mean_val = df_alloc['fw_peak_memory_mb'].mean()
        median_val = df_alloc['fw_peak_memory_mb'].median()
        min_val = df_alloc['fw_peak_memory_mb'].min()
        max_val = df_alloc['fw_peak_memory_mb'].max()
        
        ax2.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f} MB')
        ax2.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f} MB')
        ax2.axvline(min_val, color='orange', linestyle='-.', label=f'Min: {min_val:.2f} MB')
        ax2.axvline(max_val, color='purple', linestyle='-.', label=f'Max: {max_val:.2f} MB')
        
        ax2.set_xlabel('Peak Memory (MB)')
        ax2.legend(loc='upper right')
    else:
        ax2.text(0.5, 0.5, 'Insufficient variation for distribution analysis', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax2.transAxes, fontsize=10)
    
    ax2.grid(True, alpha=0.3)
    
    # 3. Distribution of Allocation Delta (total_alloc - total_free)
    ax3 = axes[1, 0]
    df_alloc['alloc_delta'] = df_alloc['fw_total_alloc'] - df_alloc['fw_total_free']
    
    if df_alloc['alloc_delta'].nunique() > 1:
        df_alloc['alloc_delta'].plot(kind='kde', ax=ax3, color='green', linewidth=2,
                                 title='Distribusi Allocation Delta')
        ax3.hist(df_alloc['alloc_delta'], bins=20, alpha=0.3, color='green', density=True)
        
        # Add statistics
        mean_val = df_alloc['alloc_delta'].mean()
        median_val = df_alloc['alloc_delta'].median()
        min_val = df_alloc['alloc_delta'].min()
        max_val = df_alloc['alloc_delta'].max()
        
        ax3.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
        ax3.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
        ax3.axvline(min_val, color='orange', linestyle='-.', label=f'Min: {min_val:.2f}')
        ax3.axvline(max_val, color='purple', linestyle='-.', label=f'Max: {max_val:.2f}')
        
        ax3.set_xlabel('Allocation Delta')
        ax3.legend(loc='upper right')
    else:
        ax3.text(0.5, 0.5, 'Insufficient variation for distribution analysis', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax3.transAxes, fontsize=10)
    
    ax3.grid(True, alpha=0.3)
    
    # 4. Scatter plot: Total Allocations vs Total Frees
    ax4 = axes[1, 1]
    ax4.scatter(df_alloc['fw_total_alloc'], df_alloc['fw_total_free'], 
               alpha=0.6, c='purple', edgecolors='k')
    
    # Add perfect balance line (y=x)
    min_val = min(df_alloc['fw_total_alloc'].min(), df_alloc['fw_total_free'].min())
    max_val = max(df_alloc['fw_total_alloc'].max(), df_alloc['fw_total_free'].max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Balance (y=x)')
    
    # Calculate correlation
    corr = df_alloc['fw_total_alloc'].corr(df_alloc['fw_total_free'])
    
    # Add trend line
    if len(df_alloc) > 1:  # Ensure there are enough points for regression
        z = np.polyfit(df_alloc['fw_total_alloc'], df_alloc['fw_total_free'], 1)
        p = np.poly1d(z)
        ax4.plot(df_alloc['fw_total_alloc'], p(df_alloc['fw_total_alloc']), 
                "g--", label=f'Trend: y={z[0]:.4f}x+{z[1]:.2f}')
    
    ax4.set_title('Total Allocations vs Total Frees')
    ax4.set_xlabel('Total Allocations')
    ax4.set_ylabel('Total Frees')
    
    # Add correlation text
    corr_text = f'Correlation: {corr:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax4.text(0.05, 0.95, corr_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', bbox=props)
    
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='lower right')
    
    plt.tight_layout()
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png', dpi=150, bbox_inches='tight')
    img_data.seek(0)
    plt.close()
    
    return img_data

def process_memory_allocation_data(alloc_data):
    """
    Process memory allocation data from the input dataset.
    
    Args:
        alloc_data: List of memory allocation data rows
        
    Returns:
        tuple: (stats_image, distribution_image, DataFrame)
    """
    import pandas as pd
    import numpy as np
    import time
    import logging
    import traceback
    
    logger = logging.getLogger(__name__)
    
    try:
        start_time = time.time()
        
        logger.info("Memproses data alokasi memori firewall...")
        column_names = ['fw_total_memory', 'fw_peak_memory', 
                        'fw_total_alloc', 'fw_failed_alloc', 
                        'fw_total_free', 'fw_failed_free', 'created_at']

        # Convert data to regular Python lists or tuples if needed
        data_list = [tuple(row) for row in alloc_data]
        
        # Create DataFrame from the data
        df_alloc = pd.DataFrame(data_list, columns=column_names)
        
        # Generate plots and get image data
        stats_img_data = plot_allocation_stats(df_alloc)
        distribution_img_data = plot_allocation_distribution(df_alloc)
        
        logger.info(f"Pemrosesan data alokasi memori selesai dalam {time.time() - start_time:.2f} detik")
        return stats_img_data, distribution_img_data, df_alloc
        
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error dalam pembuatan analisis alokasi memori: {str(e)}\n{tb}")
        return None, None, None

def allocation_conclusion_table(elements, df_alloc):
    """
    Create a conclusion table for memory allocation analysis.
    
    Args:
        elements: The PDF elements list to append to
        df_alloc: DataFrame containing memory allocation data
        
    Returns:
        Updated elements list
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    
    page_width, _ = A4
    
    # Calculate key metrics
    total_memory_mb = df_alloc['fw_total_memory'].mean() / 1024 / 1024
    peak_memory_mb = df_alloc['fw_peak_memory'].mean() / 1024 / 1024
    memory_utilization = (total_memory_mb / peak_memory_mb) * 100 if peak_memory_mb > 0 else 0
    
    total_allocs = df_alloc['fw_total_alloc'].max()  # Take max as these are cumulative counters
    total_frees = df_alloc['fw_total_free'].max()
    allocation_delta = total_allocs - total_frees
    
    failed_allocs = df_alloc['fw_failed_alloc'].sum()
    failed_frees = df_alloc['fw_failed_free'].sum()
    
    failed_alloc_rate = (failed_allocs / total_allocs) * 100 if total_allocs > 0 else 0
    failed_free_rate = (failed_frees / total_frees) * 100 if total_frees > 0 else 0
    
    # Determine memory leak risk level
    if allocation_delta == 0:
        leak_risk = "Rendah - Semua alokasi telah dibebaskan dengan sempurna"
    elif allocation_delta < 10:
        leak_risk = "Rendah - Delta alokasi kecil dan normal"
    elif allocation_delta < 100:
        leak_risk = "Rendah-Sedang - Delta alokasi terdeteksi namun masih dalam batas normal"
    elif allocation_delta < 1000:
        leak_risk = "Sedang - Perhatikan adanya potensi memory leak kecil"
    else:
        leak_risk = "Tinggi - Indikasi kuat adanya memory leak"
    
    # Determine health status
    if failed_allocs > 0:
        health_status = "Bermasalah - Terdapat kegagalan alokasi memori"
        recommendation = "Penambahan kapasitas memori/investigasi lebih lanjut sangat diperlukan"
    elif allocation_delta > 1000:
        health_status = "Perlu Perhatian - Indikasi memory leak"
        recommendation = "Lakukan investigasi lebih lanjut pada aplikasi yang memory hungry"
    elif memory_utilization > 90:
        health_status = "Perlu Perhatian - Penggunaan memori tinggi"
        recommendation = "Pertimbangkan peningkatan kapasitas atau optimasi aplikasi"
    else:
        health_status = "Sehat - Tidak ada indikasi masalah alokasi memori"
        recommendation = "Tidak diperlukan tindakan khusus"
    
    # Create table data
    title_alloc = ["Kesimpulan Analisis Alokasi Memori Firewall"]
    data_alloc = [title_alloc]
    data_alloc.append(["Parameter", "Nilai"])
    data_alloc.append(["Rata-rata Total Memory", f"{total_memory_mb:.2f} MB"])
    data_alloc.append(["Rata-rata Peak Memory", f"{peak_memory_mb:.2f} MB"])
    data_alloc.append(["Utilisasi Memori", f"{memory_utilization:.2f}%"])
    data_alloc.append(["Total Alokasi", f"{total_allocs:,}"])
    data_alloc.append(["Total Free", f"{total_frees:,}"])
    data_alloc.append(["Delta Alokasi", f"{allocation_delta:,}"])
    data_alloc.append(["Kegagalan Alokasi", f"{failed_allocs:,} ({failed_alloc_rate:.6f}%)"])
    data_alloc.append(["Kegagalan Free", f"{failed_frees:,} ({failed_free_rate:.6f}%)"])
    
    # Create recommendation table
    title_recom = ["Rekomendasi Memori Alokasi"]
    data_recom = [title_recom]
    data_recom.append(["Parameter", "Keterangan"])
    data_recom.append(["Status Kesehatan", health_status])
    data_recom.append(["Risiko Memory Leak", leak_risk])
    data_recom.append(["Rekomendasi", recommendation])
    
    def create_table(data):
        if len(data[0]) == 1:
            col_widths = [page_width * 0.8]
        else:
            col_widths = [page_width * 0.3, page_width * 0.5]

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

    # Add title and spacing
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
    
    elements.append(Paragraph("Analisis Alokasi Memori Firewall", styles['HeaderStyle']))
    elements.append(Spacer(1, 10))
    
    # Add tables
    elements.append(create_table(data_alloc))
    elements.append(Spacer(1, 15))
    elements.append(create_table(data_recom))
    elements.append(Spacer(1, 15))

    return elements

def memory_allocation_plot(elements, img_data, title):
    """
    Add memory allocation plot to the PDF elements.
    
    Args:
        elements: The PDF elements list to append to
        img_data: Image data as BytesIO
        title: Title for the plot section
        
    Returns:
        Updated elements list
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import Table, TableStyle, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    
    page_width, _ = A4
    
    # Add title
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
    
    elements.append(Paragraph(title, styles['HeaderStyle']))
    elements.append(Spacer(1, 12))
    
    # Create Image object from image data
    img = Image(img_data, width=page_width*0.8, height=page_width*0.8)
    
    # Wrap image with Table for positioning
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

def GlobalHandler(elements, alloc_data):
    import traceback
    import logging
    from reportlab.platypus import PageBreak, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    
    logger = logging.getLogger(__name__)
    
    try:
        # Process data and generate plots
        stats_img_data, distribution_img_data, df_alloc = process_memory_allocation_data(alloc_data)
        
        if stats_img_data and distribution_img_data and df_alloc is not None:
            # Add page break before starting new section
            elements.append(PageBreak())
            
            # Add usage statistics plot
            elements = memory_allocation_plot(elements, stats_img_data, 
                                           "Analisis Alokasi Memori Firewall")
            
            # Add distribution plots
            elements.append(PageBreak())
            elements = memory_allocation_plot(elements, distribution_img_data, 
                                           "Distribusi Parameter Alokasi Memori")
            
            # Add conclusion and recommendations
            elements.append(PageBreak())
            elements = allocation_conclusion_table(elements, df_alloc)
        else:
            # Add error message if processing failed
            elements.append(Paragraph("Error: Tidak dapat membuat analisis alokasi memori", 
                                     getSampleStyleSheet()['Normal']))
            elements.append(Spacer(1, 20))
        
        return elements
        
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error dalam integrasi analisis alokasi memori ke PDF: {str(e)}\n{tb}")
        
        # Add error message to PDF
        elements.append(Paragraph(f"Error: {str(e)}", getSampleStyleSheet()['Normal']))
        elements.append(Spacer(1, 20))
        
        return elements