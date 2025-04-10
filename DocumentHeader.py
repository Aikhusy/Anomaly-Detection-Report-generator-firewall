from reportlab.platypus import Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Pastikan kamu punya 'times.ttf' (font Times New Roman)
pdfmetrics.registerFont(TTFont('TimesNewRoman', 'Times.ttf'))

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name='HeaderStyle',
    fontName='TimesNewRoman',
    fontSize=14,
    leading=16,
    alignment=1,  # Center
))

def Header(inputs, elements):
    page_width, _ = A4

    # Logo
    try:
        logo = Image(inputs.get("image_path", ""), width=80, height=80)
    except:
        logo = Paragraph("", styles["HeaderStyle"])

    # Teks Judul
    title1 = Paragraph("CHECKPOINT FIREWALL STATUS REPORT", styles["HeaderStyle"])
    title2 = Paragraph(f"{inputs['month']} {inputs['year']}", styles["HeaderStyle"])
    title3 = Paragraph("PT PAMAPERSADA", styles["HeaderStyle"])

    # Tabel Vertikal Judul
    text_cell = Table([[title1], [title2], [title3]], colWidths=[page_width * 0.6])
    text_cell.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, -1), 'TimesNewRoman'),
        ('FONTSIZE', (0, 0), (-1, -1), 14),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))

    # Gabung Logo dan Teks
    header_table = Table([[logo, text_cell]], colWidths=[page_width * 0.2, page_width * 0.6])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (0, 0), 'CENTER'),
        ('ALIGN', (1, 0), (1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), 'TimesNewRoman'),
        ('FONTSIZE', (0, 0), (-1, -1), 14),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))

    elements.append(header_table)
    elements.append(Spacer(1, 20))
    return elements


def TabelGeneral(inputs, elements):
    table_data = [
        ["SITE NAME", inputs["sitename"]],
        ["Start Date", inputs["startdate"]],
        ["End Date", inputs["enddate"]],
        ["Export Date", inputs["exportdate"]],
        ["Total Firewall", str(inputs["totalfw"])]
    ]

    page_width, _ = A4
    col_widths = [page_width * 0.2, page_width * 0.6]  # 30% dan 70%

    table = Table(table_data, colWidths=col_widths)
    table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, -1), 'TimesNewRoman'),
        ('FONTSIZE', (0, 0), (-1, -1), 14),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 30))
    return elements

def GlobalHandler(elements, inputs):
    elements = Header(inputs, elements)
    elements = TabelGeneral(inputs, elements)
    print("PDF berhasil dibuat.")
    return elements
