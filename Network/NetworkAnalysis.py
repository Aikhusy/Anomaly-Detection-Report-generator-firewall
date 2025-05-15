import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
import time
import traceback
import io
from reportlab.platypus import Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_network_traffic(df):
    """
    Plot network traffic (RX/TX) for each interface over time
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import io
    
    # Get unique interfaces
    interfaces = df['interface'].unique()
    
    # Create multiple plots - one for each interface
    fig, axes = plt.subplots(nrows=len(interfaces), ncols=1, figsize=(14, 4*len(interfaces)))
    
    # If only one interface, axes is not an array, so make it one
    if len(interfaces) == 1:
        axes = [axes]
    
    # Iterate through each interface and create a plot
    for i, interface in enumerate(interfaces):
        # Create an explicit copy of the filtered DataFrame
        interface_df = df[df['interface'] == interface].copy()
        
        # Convert bytes to MB for better readability
        interface_df['rx_bytes_mb'] = interface_df['rx_bytes'] / (1024 * 1024)
        interface_df['tx_bytes_mb'] = interface_df['tx_bytes'] / (1024 * 1024)
        
        # Calculate traffic rates (derivative of bytes)
        interface_df['rx_rate'] = interface_df['rx_bytes_mb'].diff() / interface_df['created_at'].diff().dt.total_seconds()
        interface_df['tx_rate'] = interface_df['tx_bytes_mb'].diff() / interface_df['created_at'].diff().dt.total_seconds()
        
        # Replace negative values and NaN with 0 (can happen if counters reset) - avoid inplace operations
        interface_df['rx_rate'] = interface_df['rx_rate'].fillna(0)
        interface_df['tx_rate'] = interface_df['tx_rate'].fillna(0)
        interface_df['rx_rate'] = interface_df['rx_rate'].clip(lower=0)
        interface_df['tx_rate'] = interface_df['tx_rate'].clip(lower=0)
        
        # Plot RX and TX rates
        ax = axes[i]
        ax.plot(interface_df['created_at'], interface_df['rx_rate'], 'b-', label='RX Rate (MB/s)')
        ax.plot(interface_df['created_at'], interface_df['tx_rate'], 'r-', label='TX Rate (MB/s)')
        
        # Compute and plot average rates
        rx_mean = interface_df['rx_rate'].mean()
        tx_mean = interface_df['tx_rate'].mean()
        ax.axhline(rx_mean, color='blue', alpha=0.3, linestyle='--', label=f'Avg RX: {rx_mean:.3f} MB/s')
        ax.axhline(tx_mean, color='red', alpha=0.3, linestyle='--', label=f'Avg TX: {tx_mean:.3f} MB/s')
        
        # Format the plot
        ax.set_title(f'Network Traffic - {interface}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Data Rate (MB/s)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        
        # Format x-axis for better date display
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        fig.autofmt_xdate()
        
        # Add annotation with interface stats
        stats_text = (
            f"Interface: {interface}\n"
            f"MTU: {interface_df['mtu'].iloc[-1]}\n"
            f"Total RX: {interface_df['rx_bytes'].iloc[-1]/(1024*1024*1024):.2f} GB\n"
            f"Total TX: {interface_df['tx_bytes'].iloc[-1]/(1024*1024*1024):.2f} GB\n"
            f"Peak RX Rate: {interface_df['rx_rate'].max():.3f} MB/s\n"
            f"Peak TX Rate: {interface_df['tx_rate'].max():.3f} MB/s"
        )
        
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png', dpi=150, bbox_inches='tight')
    img_data.seek(0)
    plt.close()
    
    return img_data

def plot_packet_errors(df):
    """
    Plot packet errors and drops for each interface
    """
    # Get unique interfaces
    interfaces = df['interface'].unique()
    
    # Create figure with 2 rows per interface (errors and drops)
    fig, axes = plt.subplots(nrows=len(interfaces), ncols=2, figsize=(14, 4*len(interfaces)))
    
    # If only one interface, reshape axes
    if len(interfaces) == 1:
        axes = axes.reshape(1, 2)
    
    # Iterate through each interface
    for i, interface in enumerate(interfaces):
        interface_df = df[df['interface'] == interface].copy()
        
        # Calculate error and drop rates
        for direction in ['rx', 'tx']:
            interface_df[f'{direction}_error_rate'] = interface_df[f'{direction}_errors'] / interface_df[f'{direction}_packets']
            interface_df[f'{direction}_drop_rate'] = interface_df[f'{direction}_dropped'] / interface_df[f'{direction}_packets']
            
            # Replace NaN with 0 and ensure values are in range [0, 1]
            interface_df[f'{direction}_error_rate'].fillna(0, inplace=True)
            interface_df[f'{direction}_drop_rate'].fillna(0, inplace=True)
            interface_df[f'{direction}_error_rate'] = interface_df[f'{direction}_error_rate'].clip(0, 1)
            interface_df[f'{direction}_drop_rate'] = interface_df[f'{direction}_drop_rate'].clip(0, 1)
        
        # Plot error rates
        ax1 = axes[i, 0]
        ax1.plot(interface_df['created_at'], interface_df['rx_error_rate'] * 100, 'b-', 
                label='RX Error Rate %')
        ax1.plot(interface_df['created_at'], interface_df['tx_error_rate'] * 100, 'r-', 
                label='TX Error Rate %')
        
        ax1.set_title(f'Error Rates - {interface}')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Error Rate (%)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=8)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        fig.autofmt_xdate()
        
        # Plot drop rates
        ax2 = axes[i, 1]
        ax2.plot(interface_df['created_at'], interface_df['rx_drop_rate'] * 100, 'b-', 
                label='RX Drop Rate %')
        ax2.plot(interface_df['created_at'], interface_df['tx_drop_rate'] * 100, 'r-', 
                label='TX Drop Rate %')
        
        ax2.set_title(f'Drop Rates - {interface}')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Drop Rate (%)')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=8)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        
        # Add error stats annotation
        error_stats = (
            f"Interface: {interface}\n"
            f"Max RX Error Rate: {interface_df['rx_error_rate'].max()*100:.2f}%\n"
            f"Max TX Error Rate: {interface_df['tx_error_rate'].max()*100:.2f}%\n"
            f"Total RX Errors: {interface_df['rx_errors'].iloc[-1]}\n"
            f"Total TX Errors: {interface_df['tx_errors'].iloc[-1]}"
        )
        
        props = dict(boxstyle='round', facecolor='lightcoral', alpha=0.5)
        ax1.text(0.02, 0.95, error_stats, transform=ax1.transAxes, fontsize=9,
               verticalalignment='top', bbox=props)
        
        # Add drop stats annotation
        drop_stats = (
            f"Interface: {interface}\n"
            f"Max RX Drop Rate: {interface_df['rx_drop_rate'].max()*100:.2f}%\n"
            f"Max TX Drop Rate: {interface_df['tx_drop_rate'].max()*100:.2f}%\n"
            f"Total RX Drops: {interface_df['rx_dropped'].iloc[-1]}\n"
            f"Total TX Drops: {interface_df['tx_dropped'].iloc[-1]}"
        )
        
        props = dict(boxstyle='round', facecolor='lightcoral', alpha=0.5)
        ax2.text(0.02, 0.95, drop_stats, transform=ax2.transAxes, fontsize=9,
               verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png', dpi=150, bbox_inches='tight')
    img_data.seek(0)
    plt.close()
    
    return img_data

def detect_network_changes(df):
    """
    Detect changes in critical network parameters (hwaddr, inet_addr, bcast, mask)
    """
    changes = defaultdict(list)
    
    # For each interface, check for changes in critical parameters
    for interface in df['interface'].unique():
        interface_df = df[df['interface'] == interface].copy()
        
        # Set initial values
        current_hwaddr = interface_df['hwaddr'].iloc[0] if not pd.isna(interface_df['hwaddr'].iloc[0]) else None
        current_inet = interface_df['inet_addr'].iloc[0] if not pd.isna(interface_df['inet_addr'].iloc[0]) else None
        current_bcast = interface_df['bcast'].iloc[0] if not pd.isna(interface_df['bcast'].iloc[0]) else None
        current_mask = interface_df['mask'].iloc[0] if not pd.isna(interface_df['mask'].iloc[0]) else None
        
        # Check each row for changes
        for idx, row in interface_df.iterrows():
            # Skip first row
            if idx == interface_df.index[0]:
                continue
                
            # Check for MAC address change
            if not pd.isna(row['hwaddr']) and current_hwaddr != row['hwaddr'] and row['hwaddr'] is not None:
                changes[interface].append({
                    'parameter': 'hwaddr',
                    'old_value': current_hwaddr,
                    'new_value': row['hwaddr'],
                    'timestamp': row['created_at']
                })
                current_hwaddr = row['hwaddr']
            
            # Check for IP address change
            if not pd.isna(row['inet_addr']) and current_inet != row['inet_addr'] and row['inet_addr'] is not None:
                changes[interface].append({
                    'parameter': 'inet_addr',
                    'old_value': current_inet,
                    'new_value': row['inet_addr'],
                    'timestamp': row['created_at']
                })
                current_inet = row['inet_addr']
            
            # Check for broadcast address change
            if not pd.isna(row['bcast']) and current_bcast != row['bcast'] and row['bcast'] is not None:
                changes[interface].append({
                    'parameter': 'bcast',
                    'old_value': current_bcast,
                    'new_value': row['bcast'],
                    'timestamp': row['created_at']
                })
                current_bcast = row['bcast']
            
            # Check for subnet mask change
            if not pd.isna(row['mask']) and current_mask != row['mask'] and row['mask'] is not None:
                changes[interface].append({
                    'parameter': 'mask',
                    'old_value': current_mask,
                    'new_value': row['mask'],
                    'timestamp': row['created_at']
                })
                current_mask = row['mask']
    
    return changes

def NetworkDistributionPlot(elements, img_data, title):
    page_width, _ = A4
    
    # Tambahkan judul untuk bagian plot jika diperlukan
    elements = Title(elements, title)
    elements.append(Spacer(1, 12))
    
    # Buat objek Image dari data gambar
    img = Image(img_data, width=page_width*0.8, height=page_width*0.8)
    
    # Bungkus gambar dengan Table untuk mengatur posisi
    outer_table = Table([[img]])
    outer_table.setStyle(TableStyle([
        ('LEFTPADDING', (0, 0), (-1, -1), 0),  # Margin kiri
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    
    elements.append(outer_table)
    elements.append(Spacer(1, 20))
    return elements

def Title(elements, word):
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

def NetworkChangesTable(elements, changes):
    """
    Create a table showing all detected network changes
    """
    styles = getSampleStyleSheet()
    page_width, _ = A4
    
    # Add title
    elements = Title(elements, "Deteksi Perubahan Konfigurasi Network")
    elements.append(Spacer(1, 12))
    
    # If no changes detected
    if not changes:
        elements.append(Paragraph("Tidak ada perubahan konfigurasi network yang terdeteksi.", styles['Normal']))
        elements.append(Spacer(1, 12))
        return elements
    
    # For each interface with changes
    for interface, interface_changes in changes.items():
        if not interface_changes:
            continue
            
        # Create table data
        title_row = [f"Perubahan Konfigurasi Interface: {interface}"]
        data = [title_row]
        data.append(["Parameter", "Nilai Lama", "Nilai Baru", "Waktu Perubahan"])
        
        # Add all changes to table
        for change in interface_changes:
            data.append([
                change['parameter'],
                str(change['old_value']),
                str(change['new_value']),
                change['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            ])
        
        # Create the table
        col_widths = [page_width * 0.15, page_width * 0.25, page_width * 0.25, page_width * 0.2]
        table = Table(data, colWidths=col_widths, hAlign='LEFT')
        
        # Style the table
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
            ('ALIGN', (1, 2), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 1), (-1, -1), 1, colors.black),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]
        table.setStyle(TableStyle(style))
        
        elements.append(table)
        elements.append(Spacer(1, 15))
    
    return elements

def NetworkInterfaceSummaryTable(elements, df):
    """
    Create a summary table of all network interfaces
    """
    styles = getSampleStyleSheet()
    page_width, _ = A4
    
    # Add title
    elements = Title(elements, "Ringkasan Interface Network")
    elements.append(Spacer(1, 12))
    
    # Get the latest data for each interface
    interface_summary = []
    for interface in df['interface'].unique():
        interface_df = df[df['interface'] == interface]
        latest = interface_df.iloc[-1]
        
        # Calculate total traffic in GB
        rx_gb = latest['rx_bytes'] / (1024**3)
        tx_gb = latest['tx_bytes'] / (1024**3)
        
        # Calculate error rates
        rx_error_rate = latest['rx_errors'] / latest['rx_packets'] * 100 if latest['rx_packets'] > 0 else 0
        tx_error_rate = latest['tx_errors'] / latest['tx_packets'] * 100 if latest['tx_packets'] > 0 else 0
        
        # Calculate drop rates
        rx_drop_rate = latest['rx_dropped'] / latest['rx_packets'] * 100 if latest['rx_packets'] > 0 else 0
        tx_drop_rate = latest['tx_dropped'] / latest['tx_packets'] * 100 if latest['tx_packets'] > 0 else 0
        
        interface_summary.append({
            'interface': interface,
            'inet_addr': latest['inet_addr'],
            'hwaddr': latest['hwaddr'],
            'mask': latest['mask'],
            'mtu': latest['mtu'],
            'rx_gb': rx_gb,
            'tx_gb': tx_gb,
            'rx_error_rate': rx_error_rate,
            'tx_error_rate': tx_error_rate,
            'rx_drop_rate': rx_drop_rate,
            'tx_drop_rate': tx_drop_rate
        })
    
    # Create table data
    title_row = ["Ringkasan Interface Network"]
    data = [title_row]
    data.append(["Interface", "IP Address", "MAC Address", "Total RX (GB)", "Total TX (GB)", "Error Rate", "Status"])
    
    # Add data rows
    for item in interface_summary:
        # Determine status based on error and drop rates
        status = "OK"
        if item['rx_error_rate'] > 1.0 or item['tx_error_rate'] > 1.0:
            status = "PERHATIAN - Error Rate Tinggi"
        elif item['rx_drop_rate'] > 1.0 or item['tx_drop_rate'] > 1.0:
            status = "PERHATIAN - Drop Rate Tinggi"
        
        # If IP is empty, mark as down
        if pd.isna(item['inet_addr']) or item['inet_addr'] == '':
            status = "DOWN / TIDAK AKTIF"
        
        data.append([
            item['interface'],
            str(item['inet_addr']) if not pd.isna(item['inet_addr']) else "N/A",
            str(item['hwaddr']) if not pd.isna(item['hwaddr']) else "N/A",
            f"{item['rx_gb']:.2f}",
            f"{item['tx_gb']:.2f}",
            f"{max(item['rx_error_rate'], item['tx_error_rate']):.2f}%",
            status
        ])
    
    # Create the table
    col_widths = [
        page_width * 0.12,  # Interface
        page_width * 0.15,  # IP
        page_width * 0.18,  # MAC
        page_width * 0.12,  # RX
        page_width * 0.12,  # TX
        page_width * 0.12,  # Error Rate
        page_width * 0.15   # Status
    ]
    
    table = Table(data, colWidths=col_widths, hAlign='LEFT')
    
    # Style the table
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
        ('ALIGN', (1, 2), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 1), (-1, -1), 1, colors.black),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]
    
    # Add conditional formatting - highlight problem rows
    for i in range(2, len(data)):
        if "PERHATIAN" in data[i][-1]:
            style.append(('BACKGROUND', (0, i), (-1, i), colors.lightpink))
        elif "DOWN" in data[i][-1]:
            style.append(('BACKGROUND', (0, i), (-1, i), colors.lightgrey))
    
    table.setStyle(TableStyle(style))
    elements.append(table)
    elements.append(Spacer(1, 15))
    
    return elements

def NetworkConclusionTable(elements, df, changes):
    """
    Create a conclusion and recommendations table
    """
    styles = getSampleStyleSheet()
    page_width, _ = A4
    
    # Add title
    elements = Title(elements, "Kesimpulan dan Rekomendasi Network")
    elements.append(Spacer(1, 12))
    
    # Analyze data for recommendations
    interface_count = len(df['interface'].unique())
    active_interfaces = len(df.loc[df['inet_addr'].notnull() & (df['inet_addr'] != ''), 'interface'].unique())
    
    # Check for interfaces with high error rates
    high_error_interfaces = []
    for interface in df['interface'].unique():
        interface_df = df[df['interface'] == interface]
        if interface_df.empty:
            continue
            
        # Calculate average error rates
        rx_errors = interface_df['rx_errors'].iloc[-1]
        tx_errors = interface_df['tx_errors'].iloc[-1]
        rx_packets = interface_df['rx_packets'].iloc[-1]
        tx_packets = interface_df['tx_packets'].iloc[-1]
        
        rx_error_rate = rx_errors / rx_packets * 100 if rx_packets > 0 else 0
        tx_error_rate = tx_errors / tx_packets * 100 if tx_packets > 0 else 0
        
        if rx_error_rate > 1.0 or tx_error_rate > 1.0:
            high_error_interfaces.append(interface)
    
    # Create data for conclusion table
    title_row = ["Kesimpulan Analisis Network"]
    data = [title_row]
    data.append(["Parameter", "Nilai"])
    data.append(["Total Interface", str(interface_count)])
    data.append(["Interface Aktif", f"{active_interfaces} dari {interface_count}"])
    data.append(["Interface dengan Perubahan Konfigurasi", str(len(changes))])
    data.append(["Interface dengan Error Rate Tinggi", str(len(high_error_interfaces))])
    
    # Create the table
    col_widths = [page_width * 0.3, page_width * 0.5]
    table = Table(data, colWidths=col_widths, hAlign='LEFT')
    
    # Style the table
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
    elements.append(table)
    elements.append(Spacer(1, 15))
    
    # Create recommendations table
    title_row = ["Rekomendasi Network"]
    data = [title_row]
    data.append(["Aspek", "Rekomendasi"])
    
    # Add recommendations based on analysis
    if len(high_error_interfaces) > 0:
        high_error_text = f"Interface berikut memiliki error rate tinggi: {', '.join(high_error_interfaces)}. "
        high_error_text += "Lakukan pengecekan kabel fisik, konfigurasi driver, dan pastikan tidak ada interferensi elektromagnetik."
        data.append(["Error Rate Tinggi", high_error_text])
    
    if len(changes) > 0:
        changes_text = f"Terdeteksi {len(changes)} interface dengan perubahan konfigurasi. "
        changes_text += "Pastikan perubahan tersebut direncanakan dan bukan akibat masalah keamanan atau konfigurasi yang tidak sengaja."
        data.append(["Perubahan Konfigurasi", changes_text])
    
    # Check for interfaces with MTU issues
    mtu_texts = []
    for interface in df['interface'].unique():
        interface_df = df[df['interface'] == interface]
        if len(interface_df) == 0:
            continue
            
        # Get MTU value
        mtu = interface_df['mtu'].iloc[-1]
        if mtu < 1500:
            mtu_texts.append(f"{interface} (MTU: {mtu})")
    
    if len(mtu_texts) > 0:
        mtu_recommendation = f"Interface berikut memiliki MTU di bawah standar 1500: {', '.join(mtu_texts)}. "
        mtu_recommendation += "Pertimbangkan untuk menyesuaikan MTU untuk performa optimal kecuali ada kebutuhan khusus."
        data.append(["MTU Non-Standar", mtu_recommendation])
    
    # Add general recommendation
    general_text = "Lakukan monitoring rutin terhadap traffic dan error rate untuk mendeteksi anomali sedini mungkin. "
    general_text += f"Dari {interface_count} interface, pastikan semua interface yang seharusnya aktif memiliki alamat IP dan MAC yang valid."
    data.append(["Monitoring Rutin", general_text])
    
    # Create the table
    col_widths = [page_width * 0.2, page_width * 0.6]
    table = Table(data, colWidths=col_widths, hAlign='LEFT')
    
    # Style the table
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
    elements.append(table)
    
    return elements

def plot_interface_comparison(df):
    """
    Create a comparison chart of traffic across all interfaces
    """
    interfaces = df['interface'].unique()
    
    # Calculate total traffic for each interface
    traffic_data = []
    for interface in interfaces:
        interface_df = df[df['interface'] == interface]
        if interface_df.empty:
            continue
            
        last_row = interface_df.iloc[-1]
        rx_gb = last_row['rx_bytes'] / (1024**3)
        tx_gb = last_row['tx_bytes'] / (1024**3)
        
        traffic_data.append({
            'interface': interface,
            'rx_gb': rx_gb,
            'tx_gb': tx_gb,
            'total_gb': rx_gb + tx_gb
        })
    
    # Sort by total traffic
    traffic_data = sorted(traffic_data, key=lambda x: x['total_gb'], reverse=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract data for plotting
    interfaces = [item['interface'] for item in traffic_data]
    rx_values = [item['rx_gb'] for item in traffic_data]
    tx_values = [item['tx_gb'] for item in traffic_data]
    
    # Create bar positions
    x = np.arange(len(interfaces))
    width = 0.35
    
    # Create bars
    rx_bars = ax.bar(x - width/2, rx_values, width, label='RX (GB)', color='blue', alpha=0.7)
    tx_bars = ax.bar(x + width/2, tx_values, width, label='TX (GB)', color='red', alpha=0.7)
    
    # Add labels and title
    ax.set_title('Perbandingan Traffic Antar Interface (GB)')
    ax.set_xlabel('Interface')
    ax.set_ylabel('Traffic (GB)')
    ax.set_xticks(x)
    ax.set_xticklabels(interfaces, rotation=45, ha='right')
    ax.legend()
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    add_labels(rx_bars)
    add_labels(tx_bars)
    
    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png', dpi=150, bbox_inches='tight')
    img_data.seek(0)
    plt.close()
    
    return img_data

def ProcessNetworkData(network_data):
    try:
        start_time = time.time()
        
        logger.info("Memproses data network interface...")
        column_names = ['interface', 'hwaddr', 'inet_addr', 'bcast', 'mask', 'mtu', 'metric',
                      'rx_packets', 'rx_errors', 'rx_dropped', 'rx_overruns', 'rx_frame',
                      'tx_packets', 'tx_errors', 'tx_dropped', 'tx_overruns', 'tx_carrier',
                      'collisions', 'txqueuelen', 'rx_bytes', 'tx_bytes', 'created_at']

        # Convert data to regular Python lists or tuples if needed
        data_list = [tuple(row) for row in network_data]
        
        # Create DataFrame from the data
        df = pd.DataFrame(data_list, columns=column_names)
        
        # Convert created_at to datetime
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        # Convert numeric columns to appropriate types
        numeric_cols = ['mtu', 'metric', 'rx_packets', 'rx_errors', 'rx_dropped', 
                       'rx_overruns', 'rx_frame', 'tx_packets', 'tx_errors', 
                       'tx_dropped', 'tx_overruns', 'tx_carrier', 'collisions', 
                       'txqueuelen', 'rx_bytes','tx_bytes']
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Generate plots and get image data
        traffic_img_data = plot_network_traffic(df)
        errors_img_data = plot_packet_errors(df)
        comparison_img_data = plot_interface_comparison(df)
        
        # Detect changes in network configuration
        changes = detect_network_changes(df)
        
        return traffic_img_data, errors_img_data, comparison_img_data, df, changes
        
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error dalam pembuatan analisis network: {str(e)}\n{tb}")
        return None, None, None, None, None

def GlobalHandler(elements, network_data):
    try:
        traffic_img_data, errors_img_data, comparison_img_data, df, changes = ProcessNetworkData(network_data)
        
        if traffic_img_data and errors_img_data and df is not None:
            elements.append(PageBreak())
            
            # Tambahkan ringkasan interface network
            elements = NetworkInterfaceSummaryTable(elements, df)
            
            # Tambahkan plot traffic interface
            elements.append(PageBreak())
            elements = NetworkDistributionPlot(elements, traffic_img_data, "Analisis Traffic Interface Network")
            
            # Tambahkan plot error rate
            elements.append(PageBreak())
            elements = NetworkDistributionPlot(elements, errors_img_data, "Analisis Error Rate dan Drop Rate")
            
            # Tambahkan perbandingan antar interface
            elements.append(PageBreak())
            elements = NetworkDistributionPlot(elements, comparison_img_data, "Perbandingan Traffic Antar Interface")
            
            # Tambahkan tabel perubahan network jika ada
            if changes:
                elements.append(PageBreak())
                elements = NetworkChangesTable(elements, changes)
            
            # Tambahkan kesimpulan dan rekomendasi
            elements.append(PageBreak())
            elements = NetworkConclusionTable(elements, df, changes)
        else:
            # Jika ada error, tambahkan pesan error ke PDF
            elements.append(Paragraph("Error: Tidak dapat membuat analisis network", getSampleStyleSheet()['Normal']))
            elements.append(Spacer(1, 20))
            
        return elements
        
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error dalam integrasi ke PDF: {str(e)}\n{tb}")
        
        elements.append(Paragraph(f"Error: {str(e)}", getSampleStyleSheet()['Normal']))
        elements.append(Spacer(1, 20))
        return elements