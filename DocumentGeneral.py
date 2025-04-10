from reportlab.platypus import Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Registrasi font Times New Roman
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

def Title(elements):
    elements.append(Paragraph("Firewall Generals", styless['HeaderStyle']))
    elements.append(Spacer(1, 6))
    return elements

def CountTable(elements, inputs):
    data = inputs  # [('FW-JIEP03', 8928), ('CILEFWNS622', 8947)]

    table_data = [["Firewall Name", "Total Rows"]]
    table_data += [[row[0], str(row[1])] for row in data]

    page_width, _ = A4
    col_widths = [page_width * 0.2, page_width * 0.2]

    inner_table = Table(table_data, colWidths=col_widths, hAlign='LEFT')
    inner_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, -1), 'TimesNewRoman'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))

    # Bungkus dengan Table satu kolom untuk mengatur offset ke kiri
    outer_table = Table([[inner_table]], colWidths=[page_width])
    outer_table.setStyle(TableStyle([
        ('LEFTPADDING', (0, 0), (-1, -1), 60),  # Ini dia: margin kiri negatif
        ('RIGHTPADDING', (0, 0), (-1, -1), -30),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))

    elements.append(outer_table)
    elements.append(Spacer(1, 20))
    return elements 

def GlobalHandler(elements, inputs):
    elements = Title(elements)
    elements = CountTable(elements,inputs)
    return elements
