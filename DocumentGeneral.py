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

def Title(elements,word):
    elements.append(Paragraph(word, styless['HeaderStyle']))
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

def CurrentStatus(elements, inputs):

    page_width, _ = A4

    # Pisahkan menjadi 2 bagian
    header_1 = ["Firewall Name", "Uptime", "FWTmp", "VarLog", "RAM", "Swap", "Memory Error", "CPU"]
    header_2 = ["RX Error", "TX Error", "Sync Mode", "Sync State", "License", "RAID State", "Hotfix"]

    table_data_1 = [header_1]
    table_data_2 = [header_2]

    for row in inputs:
        # Ambil bagian-bagian kolom
        row_1 = row[:7] + (row[7],)  # CPU di indeks ke-7
        row_2 = row[8:]

        table_data_1.append(row_1)
        table_data_2.append(row_2)

    # Lebar kolom rata saja biar konsisten
    col_widths_1 = [page_width / len(header_1)] * len(header_1)
    col_widths_2 = [page_width / len(header_2)] * len(header_2)

    def create_table(data, col_widths, cpu_col_index=None):
        table = Table(data, hAlign='LEFT')
        style = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, -1), 'TimesNewRoman'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]

        # Tambahkan warna hijau untuk kolom CPU jika < 90
        if cpu_col_index is not None:
            for i, row in enumerate(data[1:], start=1):  # Lewati header
                try:
                    cpu_value = float(row[cpu_col_index])
                    if cpu_value < 90:
                        style.append(('TEXTCOLOR', (cpu_col_index, i), (cpu_col_index, i), colors.green))
                except ValueError:
                    continue

        table.setStyle(TableStyle(style))
        return table

    table1 = create_table(table_data_1, col_widths_1, cpu_col_index=7)
    table2 = create_table(table_data_2, col_widths_2)

    # Bungkus dan beri margin kiri
    def wrap_table(table):
        outer_table = Table([[table]], colWidths=[page_width])
        outer_table.setStyle(TableStyle([
            ('LEFTPADDING', (0, 0), (-1, -1), 60),  # Ini dia: margin kiri negatif
            ('RIGHTPADDING', (0, 0), (-1, -1), -30),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ]))
        return outer_table

    elements.append(wrap_table(table1))
    elements.append(Spacer(1, 12))
    elements.append(wrap_table(table2))
    elements.append(Spacer(1, 20))
    return elements

def GlobalHandler(elements, inputs):
    elements = Title(elements,"firewall General")
    elements = CountTable(elements,inputs["counted_rows"])
    elements = Title(elements,"Firewall Current Status")
    elements = CurrentStatus(elements,inputs["current_status"])
    return elements
