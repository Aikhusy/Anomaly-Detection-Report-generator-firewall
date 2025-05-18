import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from reportlab.platypus import Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm

def capacity_optimization_conclusions(df):
    """Generate capacity optimization conclusions and recommendations."""
    
    # Calculate key metrics
    current_avg = df['fw_vals'].mean()
    peak_avg = df['fw_peaks'].mean()
    max_peak = df['fw_peaks'].max()
    
    # Calculate growth rate (trend)
    x = range(len(df))
    z_values = np.polyfit(x, df['fw_vals'], 1)
    growth_rate = z_values[0]
    growth_percent = (growth_rate / current_avg) * 100 if current_avg > 0 else 0
    
    # Calculate session links ratio
    slinks_avg = df['fw_slinks'].mean()
    slinks_ratio = slinks_avg / current_avg if current_avg > 0 else 0
    
    # Default values
    connection_limit = None
    current_utilization = None
    peak_utilization = None
    
    # Try to get valid connection limit if available
    try:
        limit_val = df['fw_limit'].iloc[0]
        if limit_val and limit_val != '' and not (isinstance(limit_val, str) and limit_val.lower() == 'unlimited'):
            try:
                connection_limit = float(limit_val)
                if connection_limit > 0:
                    current_utilization = (current_avg / connection_limit) * 100
                    peak_utilization = (max_peak / connection_limit) * 100
            except (ValueError, TypeError):
                pass
    except (KeyError, IndexError):
        pass
    
    # Determine status based on utilization or growth
    if connection_limit is not None and current_utilization is not None:
        # Base status on utilization
        if peak_utilization < 50:
            status = "Low Utilization"
            color = "green"
            recommendation = "Current capacity is significantly higher than needed. Consider reducing resources allocated to this firewall if other services need them."
        elif peak_utilization < 70:
            status = "Optimal Utilization"
            color = "blue"
            recommendation =def plot_connections_usage(df):
    """Generate a plot showing connection usage vs. limits."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create an index for the x-axis (representing time periods)
    x = range(len(df))
    
    # Plot connections values
    ax.plot(x, df['fw_vals'], marker='o', linestyle='-', color='blue', 
            label='Current Connections')
    
    # Plot peak connections
    ax.plot(x, df['fw_peaks'], marker='^', linestyle='-', color='red', 
            label='Peak Connections')
    
    # Plot session links
    ax.plot(x, df['fw_slinks'], marker='s', linestyle='-', color='green', 
            label='Session Links')
    
    # Get statistics
    mean_vals = df['fw_vals'].mean()
    mean_peaks = df['fw_peaks'].mean()
    max_peaks = df['fw_peaks'].max()
    
    # Add horizontal line for mean values
    ax.axhline(y=mean_vals, color='blue', linestyle='-.', 
               label=f'Avg Connections: {mean_vals:.2f}')
    
    # Add horizontal line for mean peaks
    ax.axhline(y=mean_peaks, color='red', linestyle='-.', 
               label=f'Avg Peak: {mean_peaks:.2f}')
    
    # Statistics text without utilization percentages
    stats_text = (
        f"Statistics:\n"
        f"Avg Connections: {mean_vals:.2f}\n"
        f"Avg Peak: {mean_peaks:.2f}\n"
        f"Max Peak: {max_peaks}"
    )
    
    # Try to add limit information if available and valid
    try:
        limit_val = df['fw_limit'].iloc[0]
        # Check if limit is a non-empty string or a number
        if limit_val and limit_val != '':
            if isinstance(limit_val, str):
                if limit_val.lower() == 'unlimited':
                    stats_text += f"\nLimit: Unlimited"
                    # Add text annotation for unlimited
                    ax.text(len(df) * 0.8, max_peaks * 1.1, "Limit: Unlimited", 
                            color='purple', fontweight='bold')
                # Try to convert to float if it looks like a number
                elif limit_val.replace('.', '', 1).isdigit():
                    numeric_limit = float(limit_val)
                    stats_text += f"\nLimit: {numeric_limit}"
                    # Add limit line
                    ax.axhline(y=numeric_limit, color='purple', linestyle='--',
                              label=f'Connection Limit: {numeric_limit}')
                    # Calculate utilization
                    utilization_percent = (mean_vals / numeric_limit) * 100
                    peak_utilization_percent = (max_peaks / numeric_limit) * 100
                    stats_text += f"\nUtilization: {utilization_percent:.2f}%"
                    stats_text += f"\nPeak Utilization: {peak_utilization_percent:.2f}%"
            elif isinstance(limit_val, (int, float)) and limit_val > 0:
                stats_text += f"\nLimit: {limit_val}"
                # Add limit line
                ax.axhline(y=limit_val, color='purple', linestyle='--',
                          label=f'Connection Limit: {limit_val}')
                # Calculate utilization
                utilization_percent = (mean_vals / limit_val) * 100
                peak_utilization_percent = (max_peaks / limit_val) * 100
                stats_text += f"\nUtilization: {utilization_percent:.2f}%"
                stats_text += f"\nPeak Utilization: {peak_utilization_percent:.2f}%"
    except (KeyError, ValueError, TypeError, IndexError):
        # If any error occurs, just skip the limit information
        pass
    
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Configure the plot
    ax.set_title('Firewall Connection Analysis')
    ax.set_xlabel('Time Period')
    ax.set_ylabel('Number of Connections')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Save the plot to a BytesIO object
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png', dpi=150, bbox_inches='tight')
    img_data.seek(0)
    plt.close()
    
    return img_data

def plot_connections_distribution(df):
    """Generate a plot showing the distribution of connections."""
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    
    # Current Connections Distribution
    ax1 = axes[0, 0]
    df['fw_vals'].plot(kind='kde', ax=ax1, color='blue', linewidth=2,
                      title='Distribution of Current Connections')
    
    # Calculate statistics
    mean_val = df['fw_vals'].mean()
    median_val = df['fw_vals'].median()
    q1 = df['fw_vals'].quantile(0.25)
    q3 = df['fw_vals'].quantile(0.75)
    
    # Add vertical lines for statistics
    ax1.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
    ax1.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
    ax1.axvline(q1, color='orange', linestyle='-.', label=f'Q1: {q1:.2f}')
    ax1.axvline(q3, color='purple', linestyle='-.', label=f'Q3: {q3:.2f}')
    
    ax1.set_xlabel('Current Connections')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Peak Connections Distribution
    ax2 = axes[0, 1]
    df['fw_peaks'].plot(kind='kde', ax=ax2, color='red', linewidth=2,
                       title='Distribution of Peak Connections')
    
    # Calculate statistics
    mean_val = df['fw_peaks'].mean()
    median_val = df['fw_peaks'].median()
    q1 = df['fw_peaks'].quantile(0.25)
    q3 = df['fw_peaks'].quantile(0.75)
    
    # Add vertical lines for statistics
    ax2.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
    ax2.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
    ax2.axvline(q1, color='orange', linestyle='-.', label=f'Q1: {q1:.2f}')
    ax2.axvline(q3, color='purple', linestyle='-.', label=f'Q3: {q3:.2f}')
    
    ax2.set_xlabel('Peak Connections')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Session Links Distribution
    ax3 = axes[1, 0]
    df['fw_slinks'].plot(kind='kde', ax=ax3, color='green', linewidth=2,
                        title='Distribution of Session Links')
    
    # Calculate statistics
    mean_val = df['fw_slinks'].mean()
    median_val = df['fw_slinks'].median()
    q1 = df['fw_slinks'].quantile(0.25)
    q3 = df['fw_slinks'].quantile(0.75)
    
    # Add vertical lines for statistics
    ax3.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
    ax3.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
    ax3.axvline(q1, color='orange', linestyle='-.', label=f'Q1: {q1:.2f}')
    ax3.axvline(q3, color='purple', linestyle='-.', label=f'Q3: {q3:.2f}')
    
    ax3.set_xlabel('Session Links')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Utilization Percentage - Only if limit is valid
    ax4 = axes[1, 1]
    
    # Default message if utilization can't be calculated
    ax4.text(0.5, 0.5, "Connection usage trends\n(No utilization data available)", 
             ha='center', va='center', fontsize=12)
    
    # Try to calculate utilization if possible
    try:
        limit_val = df['fw_limit'].iloc[0]
        
        # Only proceed if limit_val seems valid
        if limit_val and limit_val != '' and not (isinstance(limit_val, str) and limit_val.lower() == 'unlimited'):
            # Convert to numeric if it's a string number
            numeric_limit = float(limit_val) if isinstance(limit_val, str) else limit_val
            
            if numeric_limit > 0:
                # Create utilization columns
                df['utilization_pct'] = (df['fw_vals'] / numeric_limit) * 100
                df['peak_utilization_pct'] = (df['fw_peaks'] / numeric_limit) * 100
                
                # Clear previous content
                ax4.clear()
                
                df['utilization_pct'].plot(kind='kde', ax=ax4, color='blue', linewidth=2,
                                        label='Current Utilization %')
                df['peak_utilization_pct'].plot(kind='kde', ax=ax4, color='red', linewidth=2,
                                            label='Peak Utilization %')
                
                ax4.set_xlabel('Utilization (%)')
                ax4.legend(loc='upper right')
                ax4.grid(True, alpha=0.3)
    except (KeyError, ValueError, TypeError, IndexError):
        # If any error occurs, just leave the default message
        pass
    
    ax4.set_title('Distribution of Capacity Utilization')
    
    plt.tight_layout()
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png', dpi=150, bbox_inches='tight')
    img_data.seek(0)
    plt.close()
    
    return img_data

def plot_capacity_forecast(df):
    """Generate a capacity forecast plot."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create an index for the x-axis (representing time periods)
    x = range(len(df))
    x_extended = range(len(df) + 10)  # Extend for forecast
    
    # Plot current connections
    ax.plot(x, df['fw_vals'], marker='o', linestyle='-', color='blue', 
            label='Current Connections')
    
    # Plot peak connections
    ax.plot(x, df['fw_peaks'], marker='^', linestyle='-', color='red', 
            label='Peak Connections')
    
    # Calculate linear trend for forecasting
    z = np.polyfit(x, df['fw_peaks'], 1)
    p = np.poly1d(z)
    
    # Generate forecast
    forecast = [p(i) for i in x_extended]
    
    # Plot forecast line
    ax.plot(x_extended, forecast, linestyle='--', color='purple', 
            label='Forecast Trend')
    
    # Mark the forecast section
    ax.axvspan(len(df)-1, len(df)+9, alpha=0.2, color='gray')
    
    # Default limit reached text
    limit_reached_text = "Connections are trending at rate of {:.2f}/period".format(z[0])
    
    # Try to add limit line if available
    limit_line_added = False
    try:
        limit_val = df['fw_limit'].iloc[0]
        # Check if limit is valid for plotting
        if limit_val and limit_val != '' and not (isinstance(limit_val, str) and limit_val.lower() == 'unlimited'):
            # Convert to numeric if possible
            try:
                numeric_limit = float(limit_val)
                # Add connection limit line
                ax.axhline(y=numeric_limit, color='black', linestyle='--',
                          label=f'Connection Limit: {numeric_limit}')
                limit_line_added = True
                
                # Calculate when forecast will reach limit
                if z[0] > 0:  # Positive slope = growing trend
                    periods_to_limit = (numeric_limit - z[1]) / z[0] - len(df)
                    if periods_to_limit > 0:
                        limit_reached_text = f"Limit may be reached in {periods_to_limit:.1f} periods"
                    else:
                        limit_reached_text = "Limit may already be exceeded based on trend"
            except (ValueError, TypeError):
                pass
    except (KeyError, IndexError):
        pass
    
    # Stats and forecast info
    if limit_line_added:
        current_val = df['fw_vals'].iloc[-1]
        peak_val = df['fw_peaks'].max()
        numeric_limit = float(limit_val)  # We know it's numeric from above
        
        stats_text = (
            f"Trend Analysis:\n"
            f"Growth Rate: {z[0]:.2f} connections/period\n"
            f"{limit_reached_text}\n"
            f"Current Utilization: {(current_val/numeric_limit)*100:.2f}%\n"
            f"Peak Utilization: {(peak_val/numeric_limit)*100:.2f}%"
        )
    else:
        stats_text = (
            f"Trend Analysis:\n"
            f"Growth Rate: {z[0]:.2f} connections/period\n"
            f"{limit_reached_text}"
        )
    
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Configure the plot
    ax.set_title('Firewall Connection Forecast')
    ax.set_xlabel('Time Period')
    ax.set_ylabel('Number of Connections')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Save the plot to a BytesIO object
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png', dpi=150, bbox_inches='tight')
    img_data.seek(0)
    plt.close()
    
    return img_data
    
    # Configure the plot
    ax.set_title('Firewall Connection Forecast')
    ax.set_xlabel('Time Period')
    ax.set_ylabel('Number of Connections')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Save the plot to a BytesIO object
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png', dpi=150, bbox_inches='tight')
    img_data.seek(0)
    plt.close()
    
    return img_data

def capacity_optimization_conclusions(df):
    """Generate capacity optimization conclusions and recommendations."""
    
    # Calculate key metrics
    connection_limit = df['fw_limit'].iloc[0]
    current_avg = df['fw_vals'].mean()
    peak_avg = df['fw_peaks'].mean()
    max_peak = df['fw_peaks'].max()
    
    # Check if limit is unlimited or numeric
    is_unlimited = False
    if isinstance(connection_limit, str) and connection_limit.lower() == 'unlimited':
        is_unlimited = True
    else:
        # Ensure connection_limit is numeric
        connection_limit = float(connection_limit)
    
    # Calculate utilization only if there's a numeric limit
    if not is_unlimited:
        current_utilization = (current_avg / connection_limit) * 100
        peak_utilization = (max_peak / connection_limit) * 100
    else:
        current_utilization = None
        peak_utilization = None
    
    # Calculate growth rate (trend)
    x = range(len(df))
    z_values = np.polyfit(x, df['fw_vals'], 1)
    growth_rate = z_values[0]
    growth_percent = (growth_rate / current_avg) * 100 if current_avg > 0 else 0
    
    # Calculate session links ratio
    slinks_avg = df['fw_slinks'].mean()
    slinks_ratio = slinks_avg / current_avg if current_avg > 0 else 0
    
    # Determine status based on utilization or growth if unlimited
    if is_unlimited:
        # For unlimited, base status on growth rate
        if growth_rate > 0:
            if growth_percent > 10:
                status = "High Growth"
                color = "orange"
                recommendation = "Connection usage is growing rapidly. Though limit is unlimited, monitor for potential performance impacts."
            else:
                status = "Moderate Growth"
                color = "blue"
                recommendation = "Connection usage is growing steadily. No immediate concerns with unlimited capacity."
        else:
            status = "Stable or Decreasing"
            color = "green"
            recommendation = "Connection usage is stable or decreasing. No capacity concerns with unlimited connections."
    else:
        # For numeric limits, base status on utilization
        if peak_utilization < 50:
            status = "Low Utilization"
            color = "green"
            recommendation = "Current capacity is significantly higher than needed. Consider reducing resources allocated to this firewall if other services need them."
        elif peak_utilization < 70:
            status = "Optimal Utilization"
            color = "blue"
            recommendation = "Current capacity is appropriately sized for the workload. Continue monitoring for changes in trend."
        elif peak_utilization < 90:
            status = "High Utilization"
            color = "orange"
            recommendation = "Approaching capacity limits. Monitor closely and plan for potential capacity increase if growth trend continues."
        else:
            status = "Critical Utilization"
            color = "red"
            recommendation = "Very close to or exceeding capacity limits. Immediate action recommended to increase capacity or redistribute load."
    
    # Build forecast information
    if not is_unlimited and growth_rate > 0:
        periods_to_limit = (connection_limit - z_values[1]) / growth_rate - len(df) if growth_rate != 0 else float('inf')
        if periods_to_limit > 0:
            forecast_text = f"At current growth rate, connection limit may be reached in approximately {periods_to_limit:.1f} more time periods."
        else:
            forecast_text = "Based on current trend, connection capacity is at risk of being exceeded."
    elif growth_rate > 0:
        forecast_text = "Connection usage is growing. With unlimited capacity, no limit concerns, but monitor for performance impacts."
    else:
        forecast_text = "Connection usage is stable or decreasing. No capacity issues expected in the near future."
    
    # Generate conclusions
    conclusions = {
        "status": status,
        "color": color,
        "current_avg": current_avg,
        "peak_avg": peak_avg,
        "max_peak": max_peak,
        "current_utilization": current_utilization,
        "peak_utilization": peak_utilization,
        "connection_limit": "Unlimited" if is_unlimited else connection_limit,
        "growth_rate": growth_rate,
        "growth_percent": growth_percent,
        "slinks_ratio": slinks_ratio,
        "recommendation": recommendation,
        "forecast": forecast_text
    }

def process_capacity_data(data):
    """Process capacity optimization data and generate analysis."""
    
    # Create DataFrame with the provided data
    column_names = ['fw_hostname', 'fw_names', 'fw_id', 'fw_vals', 
                   'fw_peaks', 'fw_slinks', 'fw_limit']
    
    data_list = [tuple(row) for row in data]

    df = pd.DataFrame(data_list, columns=column_names)
    
    # Generate plots
    usage_img_data = plot_connections_usage(df)
    distribution_img_data = plot_connections_distribution(df)
    forecast_img_data = plot_capacity_forecast(df)
    
    # Generate conclusions
    conclusions = capacity_optimization_conclusions(df)
    
    return usage_img_data, distribution_img_data, forecast_img_data, conclusions, df

def Title(elements, word):
    """Add a title to the PDF elements list."""
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

def CapacityDistributionPlot(elements, img_data, title):
    """Add a distribution plot to the PDF elements list."""
    page_width, _ = A4
    
    # Add title for the plot section if needed
    elements = Title(elements, title)
    elements.append(Spacer(1, 12))
    
    # Create an Image object from image data
    img = Image(img_data, width=page_width*0.8, height=page_width*0.8)
    
    # Wrap image with Table to position it
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

def CapacityConclusionTable(elements, conclusions, df):
    """Add a conclusion table to the PDF elements list."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    
    page_width, _ = A4

    # Create tables for capacity analysis conclusions
    title_capacity = ["Kesimpulan Analisis Kapasitas Koneksi Firewall"]
    data_capacity = [title_capacity]
    data_capacity.append(["Parameter", "Nilai"])
    data_capacity.append(["Total Limit Koneksi", f"{int(conclusions['connection_limit']):,}"])
    data_capacity.append(["Rata-rata Koneksi", f"{conclusions['current_avg']:.2f} ({conclusions['current_utilization']:.2f}%)"])
    data_capacity.append(["Koneksi Tertinggi", f"{conclusions['max_peak']} ({conclusions['peak_utilization']:.2f}%)"])
    data_capacity.append(["Status Utilisasi", f"{conclusions['status']}"])
    data_capacity.append(["Trend Pertumbuhan", f"{conclusions['growth_rate']:.2f} koneksi/periode ({conclusions['growth_percent']:.2f}%)"])
    data_capacity.append(["Rata-rata Session Links", f"{df['fw_slinks'].mean():.2f} (Ratio: {conclusions['slinks_ratio']:.2f})"])

    # Create recommendation table
    title_recommendation = ["Rekomendasi Optimisasi Kapasitas"]
    data_recommendation = [title_recommendation]
    data_recommendation.append(["Aspek", "Rekomendasi"])
    data_recommendation.append(["Kapasitas", conclusions['recommendation']])
    data_recommendation.append(["Forecast", conclusions['forecast']])
    
    # Function to create styled table
    def create_table(data):
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

    elements.append(create_table(data_capacity))
    elements.append(Spacer(1, 15))
    elements.append(create_table(data_recommendation))
    elements.append(Spacer(1, 15))

    return elements

def GlobalHandler(elements, capacity_data):
    """Process capacity data and add results to PDF elements list."""
    try:
        import logging
        import time
        import traceback
        
        logger = logging.getLogger(__name__)
        
        start_time = time.time()
        
        logger.info("Memproses data capacity optimisation firewall...")
        
        # Process the data using our functions
        usage_img_data, distribution_img_data, forecast_img_data, conclusions, df = process_capacity_data(capacity_data)
        
        if usage_img_data and distribution_img_data and forecast_img_data and df is not None:
            elements.append(PageBreak())
            # Add capacity usage plot to PDF elements
            elements = CapacityDistributionPlot(elements, usage_img_data, "Analisis Penggunaan Koneksi Firewall")
            
            # Add distribution plot to PDF elements
            elements.append(PageBreak())
            elements = CapacityDistributionPlot(elements, distribution_img_data, "Distribusi Parameter Koneksi")
            
            # Add forecast plot to PDF elements
            elements.append(PageBreak())
            elements = CapacityDistributionPlot(elements, forecast_img_data, "Peramalan Kapasitas Koneksi")
            
            # Add conclusion analysis
            elements.append(PageBreak())
            elements = Title(elements, "Kesimpulan dan Rekomendasi Kapasitas")
            elements.append(Spacer(1, 10))
            elements = CapacityConclusionTable(elements, conclusions, df)
        else:
            # If there's an error, add error message to PDF
            elements.append(Paragraph("Error: Tidak dapat membuat analisis kapasitas", getSampleStyleSheet()['Normal']))
            elements.append(Spacer(1, 20))
            
        logger.info(f"Analisis capacity selesai dalam {time.time() - start_time:.2f} detik")
        return elements
        
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error dalam integrasi ke PDF: {str(e)}\n{tb}")
        
        elements.append(Paragraph(f"Error: {str(e)}", getSampleStyleSheet()['Normal']))
        elements.append(Spacer(1, 20))
        return elements
