from DBConnect import Connect as Connect 
import matplotlib.pyplot as plt 
from reportlab.lib.pagesizes import letter 
from reportlab.lib.styles import getSampleStyleSheet 
import datetime 
from reportlab.platypus import ( 
    SimpleDocTemplate,
    Paragraph, Spacer 
) 
import pandas as pd 
from reportlab.lib import colors 
from DocumentHeader import GlobalHandler as DocumentHeader 
from DocumentGeneral import GlobalHandler as DocumentGeneral 
from Uptime.UptimeAnomalyDetect import GlobalHandler as UptimeAnomaly 
from Uptime.UptimeAnalysis import GlobalHandler as UptimeAnalysis
from Memory.MemoryAnalysis import GlobalHandler as MemoryAnalysis
from Storage.StorageAnalysis import GlobalHandler as StorageAnalysis
from Cpu.CpuAnalysis import GlobalHandler as CpuAnalysis
from Network.NetworkAnalysis import GlobalHandler as NetworkAnalysis
# from UptimeAnomalyLSTM import GlobalHandler as UptimeLSTM

def FetchData(startdate, enddate, fk_m_firewall=None):
    try:
        conn = Connect()
        if isinstance(conn, str):
            print(f"Connection error: {conn}")
            return {"error": f"Database connection failed: {conn}"}, None, None, None, None
        
        cursor = conn.cursor()
        
        counted_rows = None
        current_status = None
        uptime = None
        avg_uptime_results = None
        memory = None
        
        # Prepare date filter for SQL queries
        date_filter = f"BETWEEN '{startdate}' AND '{enddate}'"
        
        # Prepare firewall filter clause
        fw_filter = ""
        fw_where_clause = ""
        if fk_m_firewall is not None:
            if isinstance(fk_m_firewall, list):
                # Convert list to comma-separated string for IN clause
                fw_ids = ','.join(map(str, fk_m_firewall))
                fw_filter = f"AND fk_m_firewall IN ({fw_ids})"
                fw_where_clause = f"WHERE fk_m_firewall IN ({fw_ids})"
            else:
                # Single firewall ID
                fw_filter = f"AND fk_m_firewall = {fk_m_firewall}"
                fw_where_clause = f"WHERE fk_m_firewall = {fk_m_firewall}"
        
        try:
            # Query 1: Get firewall row counts
            query = f"""
                SELECT f.fw_name, counts.total_row 
                FROM ( 
                    SELECT fk_m_firewall, COUNT(*) AS total_row 
                    FROM tbl_t_firewall_uptime
                    WHERE tbl_t_firewall_uptime.created_at  
                    {date_filter}
                    {fw_filter}
                    GROUP BY fk_m_firewall 
                ) AS counts 
                INNER JOIN tbl_m_firewall AS f ON counts.fk_m_firewall = f.id
            """
            cursor.execute(query)
            counted_rows = cursor.fetchall()
            
            if not counted_rows:
                print("Warning: No data retrieved for counted_rows")
            
            # Query 2: Get uptime data
            uptime_query = f"""
                SELECT 
                    fw_days_uptime,  
                    fw_number_of_users,  
                    fw_load_avg_1_min,  
                    fw_load_avg_5_min,  
                    fw_load_avg_15_min,  
                    created_at 
                FROM tbl_t_firewall_uptime 
                WHERE created_at 
                {date_filter}
                {fw_filter}
                ORDER BY created_at ASC
            """
            cursor.execute(uptime_query)
            uptime = cursor.fetchall()
            
            if not uptime:
                print("Warning: No data retrieved for uptime")
            
            # Query 3: Get current firewall status
            status_query = f"""
                SELECT  
                    f.fw_name, 
                    cs.uptime, 
                    cs.fwtmp, 
                    cs.varloglog, 
                    cs.ram, 
                    cs.swap, 
                    cs.memory_error, 
                    cs.cpu, 
                    cs.rx_error_total, 
                    cs.tx_error_total, 
                    cs.sync_mode, 
                    cs.sync_state, 
                    cs.license_expiration_status, 
                    cs.raid_state, 
                    cs.hotfix_module 
                FROM  
                    tbl_t_firewall_current_status AS cs 
                INNER JOIN  
                    tbl_m_firewall AS f  
                    ON cs.fk_m_firewall = f.id
                {fw_where_clause}
            """
                           
            cursor.execute(status_query)
            current_status = cursor.fetchall()
            
            if not current_status:
                print("Warning: No data retrieved for current_status")
                
            
            # Query 5: Get memory data
            memory_query = f"""
                SELECT TOP 200  
                    mem_type,  
                    mem_total,  
                    mem_used,  
                    mem_free,  
                    mem_shared,
                    mem_cache,
                    mem_available,  
                    created_at 
                FROM tbl_t_firewall_memory
                WHERE tbl_t_firewall_memory.created_at 
                {date_filter}
                {fw_filter}
                ORDER BY created_at ASC
            """
            cursor.execute(memory_query)
            memory = cursor.fetchall()

            # Query 6: Get Storage data
            storage_query = f"""
                SELECT 
                fw_filesystem, fw_mounted_on, fw_total, 
                fw_available, fw_used, fw_used_percentage,
                created_at
                FROM tbl_t_firewall_diskspace
                WHERE created_at 
                {date_filter}
                {fw_filter}
                ORDER BY created_at ASC
            """
            cursor.execute(storage_query)
            storage = cursor.fetchall()

            cpu_query = f"""
                SELECT 
                fw_cpu_user_time_percentage,
                       fw_cpu_system_time_percentage,
                       fw_cpu_idle_time_percentage,
                       fw_cpu_usage_percentage,
                       fw_cpu_queue_length,
                       fw_cpu_interrupt_per_sec,
                       fw_cpu_number,
                       created_at
                FROM tbl_t_firewall_cpu
                WHERE created_at 
                {date_filter}
                {fw_filter}
                ORDER BY created_at ASC
            """
            cursor.execute(cpu_query)
            cpu = cursor.fetchall()

            network_query = f"""
                SELECT 
                interface, hwaddr, inet_addr, bcast, mask, mtu, metric,
                rx_packets, rx_errors, rx_dropped, rx_overruns, rx_frame,
                tx_packets, tx_errors, tx_dropped, tx_overruns, tx_carrier,
                collisions, txqueuelen, rx_bytes, tx_bytes, created_at
                FROM tbl_t_firewall_rxtx
                WHERE created_at 
                {date_filter}
                {fw_filter}
                ORDER BY created_at ASC
            """
            cursor.execute(network_query)
            network = cursor.fetchall()
                
        except Exception as e:
            print(f"Query execution error: {str(e)}")
            return {"error": f"Query execution failed: {str(e)}"}, None, None, None, None
        finally:
            # Ensure connection is closed regardless of success or exception
            try:
                conn.close()
                print("Database connection closed successfully")
            except Exception as close_err:
                print(f"Error closing database connection: {str(close_err)}")
        
        # Return the fetched data
        return counted_rows, current_status, uptime, memory, storage, cpu, network
    
    except Exception as e:
        print(f"Unexpected error in FetchData: {str(e)}")
        return {"error": f"Data retrieval failed: {str(e)}"}, None, None, None, None
    
def ExportToPDF(
    filename="firewall_report.pdf", 
    report_time=None,
    firewall_id=None,
    site_name="BRCG01",
    start_date="2025-01-01",
    end_date=None,
    month=None,
    year=None,
    logo_path="logo.png",
    record_limit=200
):

    try:
        # Set default report time if not provided
        if report_time is None:
            report_time = datetime.datetime.now()
            
        # Set default end date if not provided
        if end_date is None:
            end_date = report_time.strftime("%Y-%m-%d")
            
        # Set default month and year if not provided
        if month is None:
            month = report_time.strftime("%B")
        if year is None:
            year = report_time.strftime("%Y")
            
        # Get data from database with custom firewall filter
        data_result = FetchData('2024-04-01', '2025-04-14', 1)
        
        # Check if an error occurred during data fetching
        if isinstance(data_result[0], dict) and "error" in data_result[0]:
            error_message = data_result[0]["error"]
            print(f"Error: {error_message}")
            
            # Create a simple error report if data fetching failed
            doc = SimpleDocTemplate(filename, pagesize=letter)
            elements = []
            
            styles = getSampleStyleSheet()
            styleH1 = styles["Heading1"]
            styleN = styles["BodyText"]
            
            # Add error title and description
            elements.append(Paragraph("Firewall Report - Error", styleH1))
            elements.append(Spacer(1, 12))
            elements.append(Paragraph(f"Failed to generate report: {error_message}", styleN))
            elements.append(Spacer(1, 12))
            elements.append(Paragraph(f"Report generation attempted on: {report_time.strftime('%Y-%m-%d %H:%M:%S')}", styleN))
            
            # Build the error report document
            doc.build(elements)
            print(f"Error report generated: {filename}")
            return filename
        
        counted_rows, current_status, uptime, memory, storage, cpu, network = data_result
        # Unpack the data if no error occurred
        
        # Check if any required data is missing
        if not counted_rows or not current_status:
            # Create a warning report if data is incomplete
            doc = SimpleDocTemplate(filename, pagesize=letter)
            elements = []
            
            styles = getSampleStyleSheet()
            styleH1 = styles["Heading1"]
            styleN = styles["BodyText"]
            
            elements.append(Paragraph("Firewall Report - Incomplete Data", styleH1))
            elements.append(Spacer(1, 12))
            
            warning_message = "Report may be incomplete due to missing data:"
            if not counted_rows:
                warning_message += " No firewall count data available."
            if not current_status:
                warning_message += " No current status data available."
            if not uptime:
                warning_message += " No uptime data available."
                
            elements.append(Paragraph(warning_message, styleN))
            elements.append(Spacer(1, 12))
            elements.append(Paragraph(f"Report generated on: {report_time.strftime('%Y-%m-%d %H:%M:%S')}", styleN))
            
            # Continue with available data if any
            datass = {
                "counted_rows": counted_rows or [],
                "current_status": current_status or []
            }
            
            if uptime:
                try:
                    dataAnomaly = UptimeAnomaly(uptime)
                    print(f"Found {len(dataAnomaly.get('anomalies', []))} anomalies")
                except Exception as anomaly_err:
                    print(f"Warning: Failed to process uptime anomaly data: {str(anomaly_err)}")
                    dataAnomaly = None
            else:
                dataAnomaly = None
            
            inputs = {
                "sitename": site_name,
                "startdate": start_date,
                "enddate": end_date,
                "exportdate": report_time.strftime("%Y-%m-%d"),
                "totalfw": len(counted_rows) if counted_rows else 0,
                "month": month,
                "year": year,
                "image_path": logo_path
            }
            
            # Add available data sections
            try:
                elements = DocumentHeader(elements, inputs)
                elements = DocumentGeneral(elements, datass)
            except Exception as e:
                elements.append(Paragraph(f"Error generating report sections: {str(e)}", styleN))
            
            doc.build(elements)
            print(f"Partial report generated with warnings: {filename}")
            return filename
        
        # Process the data for the report if everything is available
        datass = {
            "counted_rows": counted_rows,
            "current_status": current_status
        }
        
        try:
            if uptime:
                dataAnomaly = UptimeAnomaly(uptime)
                # lstmAnomaly = UptimeLSTM(uptime)
        except Exception as e:
            print(f"Warning: Failed to process uptime anomaly data: {str(e)}")
            dataAnomaly = None
        
        doc = SimpleDocTemplate(filename, pagesize=letter)
        elements = []
        
        styles = getSampleStyleSheet()
        styleN = styles["BodyText"]
        styleH1 = styles["Heading1"]
        styleH2 = styles["Heading2"]
        
        # Count total firewalls from current_status data
        total_firewalls = len(set([status[0] for status in current_status])) if current_status else 0
        
        inputs = {
            "sitename": site_name,
            "startdate": start_date,
            "enddate": end_date,
            "exportdate": report_time.strftime("%Y-%m-%d"),
            "totalfw": total_firewalls,
            "month": month,
            "year": year,
            "image_path": logo_path
        }
        
        try:
            elements = DocumentHeader(elements, inputs)
            elements = DocumentGeneral(elements, datass)
            
            if uptime:
                elements = UptimeAnalysis(elements=elements, uptime_data=uptime)
                
            if memory:
                elements = MemoryAnalysis(elements=elements, memory_data=memory)

            if storage:
                elements = StorageAnalysis(elements=elements,storage_data=storage)

            if cpu :
                elements = CpuAnalysis(elements=elements, cpu_data=cpu)
            
            if network :
                elements = NetworkAnalysis(elements= elements, network_data=network)
                
            doc.build(elements)
            print(f"PDF successfully created: {filename}")
            return filename
            
        except Exception as e:
            print(f"Error building PDF document: {str(e)}")
            
            # Create a simpler error report if PDF generation fails
            error_doc = SimpleDocTemplate(filename, pagesize=letter)
            error_elements = []
            
            error_elements.append(Paragraph("Firewall Report - Error", styleH1))
            error_elements.append(Spacer(1, 12))
            error_elements.append(Paragraph(f"Failed to generate complete report: {str(e)}", styleN))
            
            error_doc.build(error_elements)
            print(f"Error report generated: {filename}")
            return filename
            
    except Exception as e:
        print(f"Unexpected error in ExportToPDF: {str(e)}")
        
        # Create a very basic error document as a last resort
        try:
            doc = SimpleDocTemplate(filename, pagesize=letter)
            elements = []
            
            styles = getSampleStyleSheet()
            elements.append(Paragraph("Critical Error", styles["Heading1"]))
            elements.append(Paragraph(f"Report generation failed: {str(e)}", styles["BodyText"]))
            
            doc.build(elements)
        except:
            print(f"Failed to create even an error report. Critical failure.")
        
        return filename