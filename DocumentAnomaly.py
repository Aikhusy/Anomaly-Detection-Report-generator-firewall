from reportlab.platypus import Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

pdfmetrics.registerFont(TTFont('TimesNewRoman', 'Times.ttf'))

# Ambil style bawaan
styless = getSampleStyleSheet()

# Tambahkan style baru hanya jika belum ada
if 'HeaderStyle' not in styless:
    styless.add(ParagraphStyle(
        name='HeaderStyle',
        fontName='TimesNewRoman',
        fontSize=14,
        leading=16,
        alignment=0,  # Center
        leftIndent=-10,    # jarak dari kiri (dalam poin)
        rightIndent=10,
    ))

def Title(elements,word):
    elements.append(Paragraph(word, styless['HeaderStyle']))
    elements.append(Spacer(1, 6))
    return elements

def GlobalHandler(elements, inputs):
    data = inputs

    table_data = [['fw_days_uptime', 'fw_number_of_users', 'fw_load_avg_1_min', 'fw_load_avg_5_min', 'fw_load_avg_15_min', 'created_at']]
    for row in data:
        table_data.append([
            row.get('fw_days_uptime', ''),
            row.get('fw_number_of_users', ''),
            row.get('fw_load_avg_1_min', ''),
            row.get('fw_load_avg_5_min', ''),
            row.get('fw_load_avg_15_min', ''),
            str(row.get('created_at', ''))
        ])

    page_width, _ = A4
    col_widths = [page_width * 0.2] * 6

    inner_table = Table(table_data, hAlign='LEFT')
    inner_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, -1), 'TimesNewRoman'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))

    outer_table = Table([[inner_table]], colWidths=[page_width])
    outer_table.setStyle(TableStyle([
        ('LEFTPADDING', (0, 0), (-1, -1), 60),
        ('RIGHTPADDING', (0, 0), (-1, -1), -30),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))

    elements.append(outer_table)
    elements.append(Spacer(1, 20))
    return elements