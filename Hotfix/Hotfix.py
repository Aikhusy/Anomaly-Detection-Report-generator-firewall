import pandas as pd
import matplotlib.pyplot as plt
import io
import logging
from reportlab.platypus import Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_hotfix_data(hotfix_data):
    logger.info("Memproses data Hotfix...")
    data_list = [tuple(row) for row in hotfix_data]
    df_hotfix = pd.DataFrame(data_list, columns=['fw_kernel', 'fw_build_number'])
    
    df_count = df_hotfix.groupby(['fw_kernel', 'fw_build_number']).size().reset_index(name='count')
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    labels = [f"{kernel} {build} ({count})" for kernel, build, count in 
              zip(df_count['fw_kernel'], df_count['fw_build_number'], df_count['count'])]
    
    wedges, texts, autotexts = ax.pie(df_count['count'], 
                                     autopct='%1.1f%%',
                                     textprops={'fontsize': 10},
                                     startangle=90)

    ax.set_title('Distribusi Versi Firewall', fontsize=14)
    
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    plt.axis('equal')
    
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png', dpi=150, bbox_inches='tight')
    img_data.seek(0)
    plt.close()
    
    return img_data, df_hotfix

def create_hotfix_table(elements, df_hotfix):

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        name='TitleStyle',
        parent=styles['Heading1'],
        fontSize=14,
        alignment=1,  
    )
    elements.append(Paragraph("Analisis Data Hotfix Firewall", title_style))
    elements.append(Spacer(1, 12))
    
    summary_style = ParagraphStyle(
        name='SummaryStyle',
        parent=styles['Normal'],
        fontSize=10,
    )
    
    unique_kernels = df_hotfix['fw_kernel'].nunique()
    unique_builds = df_hotfix['fw_build_number'].nunique()
    total_entries = len(df_hotfix)
    
    summary_text = f"""
    <b>Ringkasan Data:</b>
    <br/>• Total entri: {total_entries}
    <br/>• Jumlah versi kernel unik: {unique_kernels}
    <br/>• Jumlah build number unik: {unique_builds}
    """
    
    elements.append(Paragraph(summary_text, summary_style))
    elements.append(Spacer(1, 12))
    
    table_data = [["No.", "Firewall Kernel", "Build Number"]]
    
    for i, (_, row) in enumerate(df_hotfix.iterrows(), 1):
        table_data.append([i, row['fw_kernel'], row['fw_build_number']])

    page_width, _ = A4
    col_widths = [page_width * 0.1, page_width * 0.45, page_width * 0.25]
    
    table = Table(table_data, colWidths=col_widths)
    
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (0, 1), (0, -1), 'CENTER'),
        ('ALIGN', (1, 1), (2, -1), 'LEFT'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ])
    
    # Tambahkan zebra striping untuk kemudahan membaca
    for row_num in range(1, len(table_data)):
        if row_num % 2 == 0:
            table_style.add('BACKGROUND', (0, row_num), (-1, row_num), colors.whitesmoke)
    
    table.setStyle(table_style)
    elements.append(table)
    elements.append(Spacer(1, 12))
    
    # Tambahkan statistik distribusi
    dist_style = ParagraphStyle(
        name='DistributionStyle',
        parent=styles['Normal'],
        fontSize=10,
    )
    
    # Hitung distribusi versi firewall
    version_distribution = df_hotfix.groupby(['fw_kernel', 'fw_build_number']).size().reset_index(name='count')
    
    dist_text = "<b>Distribusi Versi Firewall:</b><br/>"
    for _, row in version_distribution.iterrows():
        percentage = (row['count'] / total_entries) * 100
        dist_text += f"• {row['fw_kernel']} {row['fw_build_number']}: {row['count']} ({percentage:.1f}%)<br/>"
    
    elements.append(Paragraph(dist_text, dist_style))
    elements.append(Spacer(1, 20))
    
    # Tambahkan kesimpulan
    conclusion_style = ParagraphStyle(
        name='ConclusionStyle',
        parent=styles['Normal'],
        fontSize=10,
    )
    
    # Analisis kesimpulan berdasarkan data
    if unique_kernels == 1 and unique_builds == 1:
        conclusion = """
        <b>Kesimpulan:</b>
        <br/>Semua firewall menggunakan versi kernel dan build number yang sama. Hal ini menunjukkan konsistensi dalam 
        penerapan versi firewall di seluruh sistem. Dari perspektif manajemen dan keamanan, ini merupakan praktik yang baik 
        karena menyederhanakan proses pemeliharaan dan memastikan semua sistem mendapatkan patch keamanan yang sama.
        """
    elif unique_kernels == 1 and unique_builds > 1:
        conclusion = """
        <b>Kesimpulan:</b>
        <br/>Semua firewall menggunakan versi kernel yang sama tetapi dengan build number yang berbeda. 
        Ini menunjukkan bahwa meskipun platform dasar firewall seragam, terdapat variasi dalam level patch atau hotfix yang diterapkan.
        Disarankan untuk mengevaluasi perbedaan build number dan mempertimbangkan standarisasi ke build number terbaru untuk keamanan optimal.
        """
    else:
        conclusion = """
        <b>Kesimpulan:</b>
        <br/>Terdapat variasi dalam versi kernel dan build number yang digunakan. Hal ini mungkin menunjukkan pendekatan 
        penerapan yang tidak seragam atau proses update yang bertahap. Untuk manajemen yang lebih baik, disarankan untuk 
        membuat rencana standarisasi versi firewall atau memiliki justifikasi yang jelas untuk penggunaan versi yang berbeda.
        """
    
    elements.append(Paragraph(conclusion, conclusion_style))
    
    return elements

def process_hotfix_recommendations(elements, df_hotfix):

    styles = getSampleStyleSheet()
    rec_style = ParagraphStyle(
        name='RecommendationStyle',
        parent=styles['Normal'],
        fontSize=10,
    )
    
    # Tambahkan judul untuk rekomendasi
    rec_title = ParagraphStyle(
        name='RecTitle',
        parent=styles['Heading2'],
        fontSize=12,
    )
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("Rekomendasi", rec_title))
    elements.append(Spacer(1, 8))
    
    # Dapatkan informasi versi yang digunakan
    current_version = df_hotfix['fw_kernel'].iloc[0]
    current_build = df_hotfix['fw_build_number'].iloc[0]
    
    # Berikan rekomendasi berdasarkan versi yang teridentifikasi
    if current_version == 'R81.20' and current_build == '.20':
        recommendation = """
        <b>Evaluasi Versi Firewall:</b>
        <br/>• Sistem menggunakan Check Point R81.20 dengan build number .20
        <br/>• Ini adalah versi yang masih aktif didukung oleh Check Point
        <br/>• Pastikan Take update terbaru telah diterapkan dari Check Point User Center
        <br/>• Direkomendasikan untuk melakukan manajemen patch secara berkala setiap bulan
        <br/>• Pertimbangkan untuk menyusun rencana upgrade ke versi R81.20 build terbaru jika tersedia
        
        <br/><b>Tindakan yang Direkomendasikan:</b>
        <br/>1. Verifikasi update terbaru dari Check Point telah diterapkan
        <br/>2. Dokumentasikan baseline konfigurasi untuk versi yang digunakan
        <br/>3. Tetapkan jadwal regular untuk evaluasi dan penerapan patch
        <br/>4. Pastikan lingkungan pengujian tersedia sebelum menerapkan patch ke lingkungan produksi
        """
    else:
        recommendation = """
        <b>Evaluasi Versi Firewall:</b>
        <br/>• Sistem menggunakan Check Point {current_version} dengan build number {current_build}
        <br/>• Verifikasi status dukungan untuk versi ini dari Check Point
        <br/>• Pastikan Take update terbaru telah diterapkan dari Check Point User Center
        <br/>• Direkomendasikan untuk melakukan manajemen patch secara berkala
        
        <br/><b>Tindakan yang Direkomendasikan:</b>
        <br/>1. Verifikasi update terbaru dari Check Point telah diterapkan
        <br/>2. Evaluasi apakah upgrade ke versi yang lebih baru diperlukan
        <br/>3. Dokumentasikan baseline konfigurasi untuk versi yang digunakan
        <br/>4. Tetapkan jadwal regular untuk evaluasi dan penerapan patch
        <br/>5. Pastikan lingkungan pengujian tersedia sebelum menerapkan patch ke lingkungan produksi
        """.format(current_version=current_version, current_build=current_build)
    
    elements.append(Paragraph(recommendation, rec_style))
    
    return elements

def HotfixAnalysisHandler(elements,hotfix_data):

    from reportlab.lib.styles import getSampleStyleSheet
        
    try:
        # Proses data hotfix
        img_data, df_hotfix = analyze_hotfix_data(hotfix_data)

        elements.append(PageBreak())
        # Buat tabel analisis hotfix
        elements = create_hotfix_table(elements, df_hotfix)
        
        # Tambahkan page break
        elements.append(PageBreak())
                
        # Tambahkan rekomendasi
        elements = process_hotfix_recommendations(elements, df_hotfix)
        
    except Exception as e:
        # Tangani error
        error_style = ParagraphStyle(
            name='ErrorStyle',
            parent=getSampleStyleSheet()['Normal'],
            textColor=colors.red,
            fontSize=10,
        )
        error_msg = f"Error saat memproses data hotfix: {str(e)}"
        elements.append(Paragraph(error_msg, error_style))
    
    return elements