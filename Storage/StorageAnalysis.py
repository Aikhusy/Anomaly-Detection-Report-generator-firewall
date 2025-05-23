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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_storage_usage(df_storage):
    """
    Plot storage usage over time for each mount point
    """
    # Get unique mount points
    mount_points = df_storage['fw_mounted_on'].unique()
    
    # Create a figure with enough subplots for each mount point
    n_plots = len(mount_points)
    fig, axes = plt.subplots(nrows=n_plots, ncols=1, figsize=(12, 5*n_plots))
    
    # Handle the case where there's only one mount point
    if n_plots == 1:
        axes = [axes]
    
    # Plot each mount point's usage
    for i, mount_point in enumerate(mount_points):
        ax = axes[i]
        
        # Get data for this mount point
        df_mount = df_storage[df_storage['fw_mounted_on'] == mount_point].copy()
        
        # Plot the usage percentage over time
        ax.plot(df_mount['created_at'], df_mount['fw_used_percentage'], 
                marker='o', linestyle='-', color='blue', 
                label=f'Usage %')
        
        # Calculate stats
        mean_val = df_mount['fw_used_percentage'].mean()
        q1 = df_mount['fw_used_percentage'].quantile(0.25)
        q2 = df_mount['fw_used_percentage'].median()
        q3 = df_mount['fw_used_percentage'].quantile(0.75)
        
        # Add horizontal lines for statistics
        ax.axhline(mean_val, color='red', linestyle='--', 
                   label=f'Mean: {mean_val:.2f}%')
        ax.axhline(q1, color='green', linestyle='-.', 
                   label=f'Q1: {q1:.2f}%')
        ax.axhline(q2, color='blue', linestyle='-.', 
                   label=f'Q2 (Median): {q2:.2f}%')
        ax.axhline(q3, color='purple', linestyle='-.', 
                   label=f'Q3: {q3:.2f}%')
        
        # Add stats text annotation
        stats_text = (
            f"Statistik Storage {mount_point}:\n"
            f"Mean Usage: {mean_val:.2f}%\n"
            f"Q1 (25%): {q1:.2f}%\n"
            f"Median (Q2): {q2:.2f}%\n"
            f"Q3 (75%): {q3:.2f}%\n"
            f"IQR: {(q3-q1):.2f}%\n"
            f"Total Size: {df_mount['fw_total'].iloc[0]/1048576:.2f} GB\n"
            f"Peak Usage: {df_mount['fw_used'].max()/1048576:.2f} GB"
        )
        
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
        
        # Set titles and labels
        ax.set_title(f'Penggunaan Storage: {mount_point}')
        ax.set_xlabel('Waktu')
        ax.set_ylabel('Penggunaan (%)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png', dpi=150, bbox_inches='tight')
    img_data.seek(0)
    plt.close()
    
    return img_data

def plot_storage_distribution(df_storage):
    """
    Plot storage distribution for each mount point
    """
    # Get unique mount points
    mount_points = df_storage['fw_mounted_on'].unique()
    n_mounts = len(mount_points)
    
    # Create a figure with subplots: 2 columns, enough rows for all mount points
    n_rows = (n_mounts + 1) // 2  # Round up division
    fig, axes = plt.subplots(nrows=n_rows, ncols=2, figsize=(14, 5*n_rows))
    
    # Flatten the axes array for easier indexing
    if n_rows == 1 and n_mounts == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot distribution for each mount point
    for i, mount_point in enumerate(mount_points):
        if i < len(axes):
            ax = axes[i]
            
            # Get data for this mount point
            df_mount = df_storage[df_storage['fw_mounted_on'] == mount_point].copy()
            
            # Convert to GB for better readability
            df_mount['used_gb'] = df_mount['fw_used'] / 1048576
            df_mount['available_gb'] = df_mount['fw_available'] / 1048576
            
            # Plot KDE for used storage
            if df_mount['used_gb'].nunique() > 1:  # Only plot if we have varied data
                df_mount['used_gb'].plot(kind='kde', ax=ax, color='blue', linewidth=2,
                                        label='Used Storage')
                
                # Calculate statistics
                mean_val = df_mount['used_gb'].mean()
                median_val = df_mount['used_gb'].median()
                q1 = df_mount['used_gb'].quantile(0.25)
                q3 = df_mount['used_gb'].quantile(0.75)
                
                # Add vertical lines for statistics
                ax.axvline(mean_val, color='red', linestyle='--', 
                           label=f'Mean: {mean_val:.2f} GB')
                ax.axvline(median_val, color='green', linestyle='--', 
                           label=f'Median: {median_val:.2f} GB')
                ax.axvline(q1, color='orange', linestyle='-.', 
                           label=f'Q1: {q1:.2f} GB')
                ax.axvline(q3, color='purple', linestyle='-.', 
                           label=f'Q3: {q3:.2f} GB')
                
                # Add available storage distribution
                if df_mount['available_gb'].nunique() > 1:
                    df_mount['available_gb'].plot(kind='kde', ax=ax, 
                                                 color='green', linewidth=2,
                                                 label='Available Storage')
                
                ax.set_title(f'Distribusi Storage: {mount_point}')
                ax.set_xlabel('Storage (GB)')
                ax.legend(loc='upper right')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'Storage Data untuk {mount_point} tidak bervariasi',
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=12)
    
    # Hide empty subplots if any
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png', dpi=150, bbox_inches='tight')
    img_data.seek(0)
    plt.close()
    
    return img_data

def Title(elements, word):
    """
    Add a title to the PDF
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

def StorageDistributionPlot(elements, img_data, title):
    """
    Add a plot to the PDF
    """
    page_width, _ = A4
    
    # Add title if provided
    elements = Title(elements, title)
    elements.append(Spacer(1, 12))
    
    # Create Image object from image data
    img = Image(img_data, width=page_width*0.8, height=page_width*0.8)
    
    # Wrap image in a Table for positioning
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

def StorageConclusionTable(elements, df_storage):
    """
    Create a conclusion table for each mount point
    """
    page_width, _ = A4
    styles = getSampleStyleSheet()
    
    # Group by mount point for analysis
    mount_points = df_storage['fw_mounted_on'].unique()
    
    for mount_point in mount_points:
        df_mount = df_storage[df_storage['fw_mounted_on'] == mount_point].copy()
        
        # Calculate statistics
        total_storage_gb = df_mount['fw_total'].iloc[0] / 1048576
        avg_used_gb = df_mount['fw_used'].mean() / 1048576
        avg_used_percent = df_mount['fw_used_percentage'].mean()
        peak_used_gb = df_mount['fw_used'].max() / 1048576
        peak_used_percent = df_mount['fw_used_percentage'].max()
        avg_available_gb = df_mount['fw_available'].mean() / 1048576
        filesystem = df_mount['fw_filesystem'].iloc[0]
        
        # Create summary data
        title_storage = [f"Kesimpulan Analisis Storage: {mount_point}"]
        data_storage = [title_storage]
        data_storage.append(["Parameter", "Nilai"])
        data_storage.append(["Filesystem", f"{filesystem}"])
        data_storage.append(["Total Storage", f"{total_storage_gb:.2f} GB"])
        data_storage.append(["Rata-rata Penggunaan", f"{avg_used_gb:.2f} GB ({avg_used_percent:.2f}%)"])
        data_storage.append(["Penggunaan Tertinggi", f"{peak_used_gb:.2f} GB ({peak_used_percent:.2f}%)"])
        data_storage.append(["Rata-rata Tersedia", f"{avg_available_gb:.2f} GB"])
        
        # Create recommendations based on usage patterns
        title_rekomendasi = ["Rekomendasi"]
        data_rekomendasi = [title_rekomendasi]
        data_rekomendasi.append(["Kategori", "Rekomendasi"])
        
        # Add capacity planning recommendation
        if peak_used_percent > 85:
            capacity_rekomendasi = ("Penggunaan storage mencapai lebih dari 85% pada beberapa periode. "
                                   "Disarankan untuk merencanakan penambahan kapasitas atau pembersihan data.")
        elif peak_used_percent > 70:
            capacity_rekomendasi = ("Penggunaan storage mencapai lebih dari 70% pada beberapa periode. "
                                   "Perlu pemantauan ketat terhadap pertumbuhan penggunaan.")
        else:
            capacity_rekomendasi = "Kapasitas storage masih mencukupi berdasarkan pola penggunaan saat ini."
        
        data_rekomendasi.append(["Kapasitas", capacity_rekomendasi])
        
        # Add growth trend recommendation
        # Calculate daily growth rate based on linear regression
        if len(df_mount) > 1:
            # Convert timestamps to ordinal values for regression
            x = np.array(range(len(df_mount)))
            y = df_mount['fw_used_percentage'].values
            
            slope, _, _, _, _ = stats.linregress(x, y)
            daily_samples = 12 * 24  # Assuming 5-minute intervals
            daily_growth_percent = slope * daily_samples
            
            if daily_growth_percent > 1:
                growth_rekomendasi = (f"Storage mengalami peningkatan sekitar {daily_growth_percent:.2f}% per hari. "
                                     f"Pada tingkat pertumbuhan ini, kapasitas penuh akan tercapai dalam "
                                     f"sekitar {((100 - avg_used_percent) / daily_growth_percent):.1f} hari.")
            elif daily_growth_percent > 0.1:
                growth_rekomendasi = (f"Storage mengalami peningkatan ringan sekitar {daily_growth_percent:.2f}% per hari. "
                                     f"Disarankan untuk memantau pertumbuhan secara berkala.")
            else:
                growth_rekomendasi = "Tidak terdeteksi pertumbuhan penggunaan storage yang signifikan."
        else:
            growth_rekomendasi = "Belum cukup data untuk menganalisis tren pertumbuhan."
        
        data_rekomendasi.append(["Pertumbuhan", growth_rekomendasi])
        
        # Create table style
        def create_table(data):
            if len(data[0]) == 1:
                col_widths = [page_width * 0.2, page_width * 0.5]
            else:
                col_widths = [page_width * 0.2, page_width * 0.5]
            
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
        elements.append(create_table(data_storage))
        elements.append(Spacer(1, 15))
        elements.append(create_table(data_rekomendasi))
        elements.append(Spacer(1, 25))
    
    return elements

def ProcessStorageData(storage_data):
    """
    Process storage data and generate analysis
    """
    try:
        start_time = time.time()
        
        logger.info("Memproses data storage...")
        column_names = [
            'fw_filesystem', 'fw_mounted_on', 'fw_total',  
            'fw_available', 'fw_used', 'fw_used_percentage',
            'created_at'
        ]
        
        # Convert data to regular Python lists or tuples if needed
        data_list = [tuple(row) for row in storage_data]
        
        # Create DataFrame from the data
        df = pd.DataFrame(data_list, columns=column_names)
        
        # Ensure numeric columns are properly typed
        numeric_cols = ['fw_total', 'fw_available', 'fw_used', 'fw_used_percentage']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure timestamp column is properly formatted
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        
        # Sort by timestamp
        df = df.sort_values('created_at')
        
        # Generate plots and get image data
        usage_img_data = plot_storage_usage(df)
        distribution_img_data = plot_storage_distribution(df)
        
        return usage_img_data, distribution_img_data, df
        
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error dalam pembuatan analisis storage: {str(e)}\n{tb}")
        return None, None, None

def GlobalHandler(elements, storage_data):
    """
    Handle the storage analysis and add it to PDF elements
    """
    try:
        usage_img_data, distribution_img_data, df_storage = ProcessStorageData(storage_data)
        
        if usage_img_data and distribution_img_data and df_storage is not None:
            elements.append(PageBreak())
            # Add storage usage plot to PDF elements
            elements = StorageDistributionPlot(elements, usage_img_data, "Analisis Penggunaan Storage")
            
            # Add storage distribution plot to PDF elements
            elements.append(PageBreak())
            elements = StorageDistributionPlot(elements, distribution_img_data, "Distribusi Storage")
            
            # Add conclusion analysis
            elements.append(PageBreak())
            elements = Title(elements, "Kesimpulan dan Rekomendasi Storage")
            elements.append(Spacer(1, 10))
            elements = StorageConclusionTable(elements, df_storage)
        else:
            # If there's an error, add error message to PDF
            elements.append(Paragraph("Error: Tidak dapat membuat analisis storage", getSampleStyleSheet()['Normal']))
            elements.append(Spacer(1, 20))
            
        return elements
        
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error dalam integrasi ke PDF: {str(e)}\n{tb}")
        
        elements.append(Paragraph(f"Error: {str(e)}", getSampleStyleSheet()['Normal']))
        elements.append(Spacer(1, 20))
        return elements

# Example usage
# This would be called from your main PDF generation function
# storage_data = [...] # Your actual storage data
# elements = []
# elements = GlobalStorageHandler(elements, storage_data)