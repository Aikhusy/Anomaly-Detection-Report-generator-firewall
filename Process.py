from DBConnect import Connect as Connect 
import matplotlib.pyplot as plt 
from reportlab.lib.pagesizes import letter 
from reportlab.lib.styles import getSampleStyleSheet 
import datetime 
from reportlab.platypus import ( 
    SimpleDocTemplate, Table, TableStyle, Image, 
    Paragraph, Spacer 
) 
import pandas as pd 
from reportlab.lib import colors 
from DocumentHeader import GlobalHandler as DocumentHeader 
from DocumentGeneral import GlobalHandler as DocumentGeneral 
from Uptime.UptimeAnomalyDetect import GlobalHandler as UptimeAnomaly 
from Uptime.UptimeAnalysis import GlobalHandler as UptimeAnalysis
from Memory.MemoryAnalysis import GlobalHandler as MemoryAnalysis
# from UptimeAnomalyLSTM import GlobalHandler as UptimeLSTM
 
def FetchData():
    try:
        conn = Connect()
        if isinstance(conn, str):
            print(f"Connection error: {conn}")
            return {"error": f"Database connection failed: {conn}"}, None, None
        
        cursor = conn.cursor()
        
        # Initialize variables to store query results
        counted_rows = None
        current_status = None
        uptime = None
        memory = None
        
        try:
            # Query 1: Get firewall row counts
            query = """
                SELECT f.fw_name, counts.total_row 
                FROM ( 
                    SELECT fk_m_firewall, COUNT(*) AS total_row 
                    FROM tbl_t_firewall_uptime 
                    GROUP BY fk_m_firewall 
                ) AS counts 
                INNER JOIN tbl_m_firewall AS f ON counts.fk_m_firewall = f.id 
            """
            cursor.execute(query)
            counted_rows = cursor.fetchall()
            
            if not counted_rows:
                print("Warning: No data retrieved for counted_rows")
            
            # Query 2: Get uptime data
            query = """
                SELECT TOP 200  
                    fw_days_uptime,  
                    fw_number_of_users,  
                    fw_load_avg_1_min,  
                    fw_load_avg_5_min,  
                    fw_load_avg_15_min,  
                    created_at 
                FROM tbl_t_firewall_uptime 
                WHERE fk_m_firewall = 1 
                ORDER BY created_at ASC
            """
            cursor.execute(query)
            uptime = cursor.fetchall()
            
            if not uptime:
                print("Warning: No data retrieved for uptime")
            
            # Query 3: Get current firewall status
            query = """
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
            """
            
            cursor.execute(query)
            current_status = cursor.fetchall()
            
            if not current_status:
                print("Warning: No data retrieved for current_status")
                
            avg_uptime_query = """
                            SELECT 
                f.fw_name,
                AVG(fu.fw_number_of_users) AS average_users,
                AVG(fu.fw_load_avg_1_min) AS average_load_1min,
                AVG(fu.fw_load_avg_5_min) AS average_load_5min,
                AVG(fu.fw_load_avg_15_min) AS average_load_15min,
                MIN(fu.fw_days_uptime) AS min_uptime,
                MAX(fu.fw_days_uptime) AS max_uptime,
                AVG(fu.fw_days_uptime) AS average_uptime,
                COUNT(*) AS record_count,
                MIN(fu.created_at) AS earliest_record,
                MAX(fu.created_at) AS latest_record
            FROM 
                tbl_t_firewall_uptime AS fu
            INNER JOIN 
                tbl_m_firewall AS f ON fu.fk_m_firewall = f.id
            WHERE 
                fu.fk_m_firewall = 1  -- You can remove this line to get averages for all firewalls
            GROUP BY 
                f.fw_name
            ORDER BY 
                f.fw_name;
            """
            cursor.execute(avg_uptime_query)
            avg_uptime_results = cursor.fetchall()
            
            query = """
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
                WHERE fk_m_firewall = 1 
                ORDER BY created_at ASC
            """
            cursor.execute(query)
            memory = cursor.fetchall()
            # Query for highest and lowest uptime values per firewall
                
        except Exception as e:
            print(f"Query execution error: {str(e)}")
            return {"error": f"Query execution failed: {str(e)}"}, None, None
        finally:
            # Ensure connection is closed regardless of success or exception
            try:
                conn.close()
                print("Database connection closed successfully")
            except Exception as close_err:
                print(f"Error closing database connection: {str(close_err)}")
        
        # Return the fetched data
        return counted_rows, current_status, uptime, avg_uptime_results,memory
    
    except Exception as e:
        print(f"Unexpected error in FetchData: {str(e)}")
        return {"error": f"Data retrieval failed: {str(e)}"}, None, None
 
def ExportToPDF(filename="firewall_report.pdf", time=datetime.datetime.now()):
    try:
        # Get data from database
        data_result = FetchData()
        
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
            elements.append(Paragraph(f"Report generation attempted on: {time.strftime('%Y-%m-%d %H:%M:%S')}", styleN))
            
            # Build the error report document
            doc.build(elements)
            print(f"Error report generated: {filename}")
            return filename
        
        # Unpack the data if no error occurred
        counted_rows, current_status, uptime, avg_uptime_results, memory = data_result
        
        
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
            elements.append(Paragraph(f"Report generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}", styleN))
            
            # Continue with available data if any
            datass = {
                "counted_rows": counted_rows or [],
                "current_status": current_status or []
            }
            
            if uptime:
                dataAnomaly = UptimeAnomaly(uptime)
                print(dataAnomaly['anomalies'])
            else:
                dataAnomaly = None
            
            inputs = {
                "sitename": "BRCG01",
                "startdate": "2025-01-01",
                "enddate": "2025-04-01",
                "exportdate": "2025-04-10",
                "totalfw": 5,
                "month": "April",
                "year": "2025",
                "image_path": "logo.png"
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
        if time is None:
            time = datetime.datetime.now()
            
        datass = {
            "counted_rows": counted_rows,
            "current_status": current_status
        }
        
        try:
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
        
        inputs = {
            "sitename": "BRCG01",
            "startdate": "2025-01-01",
            "enddate": "2025-04-01",
            "exportdate": "2025-04-10",
            "totalfw": 5,
            "month": "April",
            "year": "2025",
            "image_path": "logo.png"
        }
        
        try:
            elements = DocumentHeader(elements, inputs)
            elements = DocumentGeneral(elements, datass)
            elements = UptimeAnalysis(elements=elements,uptime_data=uptime)
            elements = MemoryAnalysis ( elements=elements, memory_data= memory)

            doc.build(elements)
            print(f"PDF berhasil dibuat: {filename}")
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