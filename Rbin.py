def handler():
    # Judul dan waktu
    elements.append(Paragraph("Laporan Penggunaan CPU Firewall", styleH1))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(time_str, styleH2))
    elements.append(Spacer(1, 6))

    # Header untuk tabel
    header1 = ["Firewall Name", "Uptime", "FW Temp", "/var/log", "RAM", "Swap", "Memory Error", "CPU"]
    header2 = ["RX Error", "TX Error", "Sync Mode", "Sync State", "License Status", "RAID State", "Hotfix Module"]

    # Siapkan data tabel bagian 1 dan 2
    data1 = [header1]
    data2 = [header2]

    for row in current_status:
        data1.append([Paragraph(str(cell), styleN) for cell in row[:8]])
        data2.append([Paragraph(str(cell), styleN) for cell in row[8:]])

    # Buat tabel tanpa colWidths agar fleksibel
    table1 = Table(data1)
    table2 = Table(data2)

    # Gaya tabel
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ])

    table1.setStyle(style)
    table2.setStyle(style)

    elements.append(Paragraph("Status Saat Ini (Bagian 1):", styleH2))
    elements.append(Spacer(1, 6))
    elements.append(table1)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Status Saat Ini (Bagian 2):", styleH2))
    elements.append(Spacer(1, 6))
    elements.append(table2)
    elements.append(Spacer(1, 24))

    # Tabel 5 data terbaru
    last_5_header = ["ID", "fk_m_firewall", "CPU Usage (%)", "Timestamp"]
    data_last_5 = [last_5_header] + [[Paragraph(str(cell), styleN) for cell in row] for row in last_5_data]
    table_last_5 = Table(data_last_5)
    table_last_5.setStyle(style)

    elements.append(Paragraph("5 Data Terbaru:", styleH2))
    elements.append(Spacer(1, 6))
    elements.append(table_last_5)
    elements.append(Spacer(1, 24))

    # Grafik
    graph = Image(graph_path, width=400, height=200)
    elements.append(Paragraph("Grafik Penggunaan CPU:", styleH2))
    elements.append(Spacer(1, 6))
    elements.append(graph)
