
import pandas as pd
import io
from reportlab.platypus import Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
import logging
import time
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def Title(elements, word):
    """Add a title to the PDF elements"""
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

def RAIDAnalysisTable(elements, df):
    """Create a summary table for RAID analysis"""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
    
    page_width, _ = A4
    styles = getSampleStyleSheet()
    
    # Generate summary statistics
    raid_level = df['raid_level'].iloc[0]
    disk_count = df['raid_number_of_disks'].iloc[0]
    raid_size = df['raid_size'].iloc[0]
    raid_state = df['raid_state'].iloc[0]
    raid_flag = df['raid_flag'].iloc[0]
    consistency = "Konsisten" if (df['raid_state'] == raid_state).all() and (df['raid_flag'] == raid_flag).all() else "Tidak Konsisten"
    
    # Create summary table
    title_raid = ["Ringkasan Analisis RAID"]
    data_raid = [title_raid]
    data_raid.append(["Parameter", "Nilai"])
    data_raid.append(["RAID Level", raid_level])
    data_raid.append(["Jumlah Disk", f"{disk_count}"])
    data_raid.append(["Kapasitas RAID", raid_size])
    data_raid.append(["Status RAID", raid_state])
    data_raid.append(["Flag RAID", raid_flag])
    data_raid.append(["Konsistensi Status", consistency])
    
    # Determine risk level and recommendation based on RAID state
    risk_level = "TINGGI" if raid_state == "DEGRADED" else "RENDAH"
    
    # Create recommendation based on RAID state and flag
    recommendation = ""
    if raid_state == "DEGRADED":
        recommendation = "RAID dalam kondisi DEGRADED, yang berarti salah satu disk telah gagal. Tindakan segera diperlukan untuk mencegah kehilangan data total. Segera ganti disk yang rusak dan rebuild RAID."
        if raid_flag == "VOLUME_INACTIVE":
            recommendation += " Volume dalam status tidak aktif, yang semakin meningkatkan risiko. Aktifkan kembali volume setelah perbaikan."
    else:
        recommendation = "RAID berfungsi normal. Lakukan pemantauan rutin untuk memastikan kesehatan sistem tetap terjaga."
    
    # Add recommendation table
    title_recommendation = ["Rekomendasi"]
    data_recommendation = [title_recommendation]
    data_recommendation.append(["Tingkat Risiko", risk_level])
    data_recommendation.append(["Rekomendasi Tindakan", recommendation])
    
    # Create and style tables
    def create_table(data):
        if len(data[0]) == 1:
            col_widths = [page_width * 0.8]
        else:
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

    elements.append(create_table(data_raid))
    elements.append(Spacer(1, 15))
    elements.append(create_table(data_recommendation))
    elements.append(Spacer(1, 20))
    
    # Add detailed analysis explanation
    elements.append(Paragraph("Analisis Detail:", styles['Heading2']))
    elements.append(Spacer(1, 10))
    
    # Explanation text based on the data
    if raid_level == "RAID-1":
        raid_explanation = (
            "RAID-1 menggunakan teknologi mirroring di mana data disalin secara identik ke dua atau lebih disk. "
            "Ini memberikan redundansi data yang baik tetapi mengurangi kapasitas penyimpanan yang tersedia secara efektif menjadi "
            "kapasitas satu disk. Dalam konfigurasi normal, RAID-1 dapat bertahan dari kegagalan satu disk."
        )
    else:
        raid_explanation = f"Sistem menggunakan konfigurasi {raid_level}. Analisis spesifik untuk level RAID ini tidak tersedia."
    
    elements.append(Paragraph(raid_explanation, styles['Normal']))
    elements.append(Spacer(1, 10))
    
    if raid_state == "DEGRADED":
        state_explanation = (
            "Status DEGRADED mengindikasikan bahwa array RAID sedang beroperasi dengan redundansi berkurang atau tanpa redundansi sama sekali. "
            "Hal ini biasanya terjadi ketika salah satu disk dalam array mengalami kegagalan. Dalam RAID-1 dengan 2 disk, "
            "ini berarti array sedang beroperasi dengan hanya satu disk yang berfungsi. Kondisi ini sangat berisiko karena "
            "kegagalan disk kedua akan menyebabkan kehilangan data total."
        )
        elements.append(Paragraph(state_explanation, styles['Normal']))
        elements.append(Spacer(1, 10))
    
    if raid_flag == "VOLUME_INACTIVE":
        flag_explanation = (
            "Flag VOLUME_INACTIVE menunjukkan bahwa volume RAID tidak aktif. Ini adalah kondisi kritis yang perlu "
            "ditangani segera. Volume yang tidak aktif tidak dapat diakses oleh sistem dan dapat mengindikasikan "
            "masalah serius pada array RAID. Kemungkinan penyebabnya termasuk kegagalan disk yang parah, "
            "masalah controller RAID, atau issue hardware lainnya."
        )
        elements.append(Paragraph(flag_explanation, styles['Normal']))
    
    return elements

def RAIDHistoryTable(elements, df):
    """Create a table showing the history of RAID statuses"""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    
    page_width, _ = A4
    styles = getSampleStyleSheet()
    
    # Generate history data for table
    elements.append(Paragraph("Histori Status RAID:", styles['Heading2']))
    elements.append(Spacer(1, 10))
    
    # Create list for history table
    data = [["No", "RAID Level", "Jumlah Disk", "Kapasitas", "Status", "Flag"]]
    
    # Add data rows
    for i, row in df.iterrows():
        data.append([
            str(i+1),
            row['raid_level'],
            str(row['raid_number_of_disks']),
            row['raid_size'],
            row['raid_state'],
            row['raid_flag']
        ])
    
    # Create and style table
    col_widths = [
        page_width * 0.05,  # No
        page_width * 0.15,  # RAID Level
        page_width * 0.1,   # Jumlah Disk
        page_width * 0.15,  # Kapasitas
        page_width * 0.15,  # Status
        page_width * 0.3    # Flag
    ]
    
    table = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ALIGN', (0, 0), (0, -1), 'CENTER'),  # No column
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),  # Run Token column
        ('ALIGN', (2, 0), (2, -1), 'CENTER'),  # RAID Level column
        ('ALIGN', (3, 0), (3, -1), 'CENTER'),  # Disk count column
        ('ALIGN', (4, 0), (4, -1), 'CENTER'),  # Capacity column
        ('ALIGN', (6, 0), (6, -1), 'LEFT'),  # Status column  
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]
    
    for i in range(1, len(data)):
        if data[i][5] == "DEGRADED":
            style.append(('BACKGROUND', (5, i), (5, i), colors.pink))
    
    table.setStyle(TableStyle(style))
    elements.append(table)
    elements.append(Spacer(1, 20))
    
    return elements

def ProcessRAIDData(raid_data):
    try:
        start_time = time.time()
        
        logger.info("Memproses data RAID firewall...")
        column_names = [
         'raid_volume_id',
            'raid_level', 'raid_number_of_disks', 'raid_size',
            'raid_state', 'raid_flag'
        ]

        data_list = [tuple(row) for row in raid_data]
        
        df = pd.DataFrame(data_list, columns=column_names)
        
        processing_time = time.time() - start_time
        logger.info(f"Waktu pemrosesan data RAID: {processing_time:.2f} detik")
        
        return df
        
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error dalam pembuatan analisis RAID: {str(e)}\n{tb}")
        return None

def GlobalRAIDHandler(elements, raid_data):
    try:
        df = ProcessRAIDData(raid_data)
        
        if df is not None:
            elements = Title(elements, "Analisis Sistem RAID")
            elements.append(Spacer(1, 10))
            
            # Add RAID analysis table
            elements = RAIDAnalysisTable(elements, df)
            
            # Add history table on a new page
            elements.append(PageBreak())
            elements = Title(elements, "Histori Status RAID")
            elements.append(Spacer(1, 10))
            elements = RAIDHistoryTable(elements, df)
        else:
            # If there's an error, add error message to PDF
            elements.append(Paragraph("Error: Tidak dapat membuat analisis RAID", getSampleStyleSheet()['Normal']))
            elements.append(Spacer(1, 20))
            
        return elements
        
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error dalam integrasi analisis RAID ke PDF: {str(e)}\n{tb}")
        
        elements.append(Paragraph(f"Error: {str(e)}", getSampleStyleSheet()['Normal']))
        elements.append(Spacer(1, 20))
        return elements

# Example usage
def create_raid_analysis_report(raid_data):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    import io
    
    # Create PDF buffer
    buffer = io.BytesIO()
    
    # Create PDF document
    doc = SimpleDocTemplate(buffer, pagesize=A4, 
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=72)
    
    # Create elements list
    elements = []
    
    # Add title
    styles = getSampleStyleSheet()
    elements.append(Paragraph("Laporan Analisis RAID", styles['Title']))
    elements.append(Spacer(1, 20))
    
    # Process RAID data and add to elements
    elements = GlobalRAIDHandler(elements, raid_data)
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    
    return buffer