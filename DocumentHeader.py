from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

styles = getSampleStyleSheet()

def Header(inputs, elements):
    header_text = Paragraph(
        f"CHECKPOINT FIREWALL STATUS REPORT<br/>{inputs['month']} {inputs['year']}<br/>PT PAMAPERSADA",
        styles["Title"]
    )
    elements.append(header_text)
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
    table = Table(table_data, colWidths=[120, 200])
    table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 20))
    return elements

def GlobalHandler(elements,inputs):
    elements = Header(inputs, elements)
    elements = TabelGeneral(inputs, elements)
    
    
    print("PDF berhasil dibuat.")
    return elements