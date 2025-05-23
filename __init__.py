import json
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
from LogManagement import tulis_log as logs
from DBConnect import Connect as Connect
from Process import ExportToPDF as export

def FetchData():
    try:
        conn = Connect()
        if isinstance(conn, str):
            return None, None, f"Database connection failed: {conn}"

        cursor = conn.cursor()

        status_query = """
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

        cursor.execute(status_query)
        columns = [desc[0] for desc in cursor.description]  # Get column names
        data = cursor.fetchall()
        conn.close()

        # Convert all data to strings to avoid TclError
        converted_data = []
        for row in data:
            converted_data.append(tuple(str(item) if item is not None else "" for item in row))

        return columns, converted_data, None

    except Exception as e:
        return None, None, str(e)


def update_config(log_level="INFO", message="Konfigurasi database diperbarui"):
    try:
        with open("config.json", "r") as file:
            data = json.load(file)

        data["LastExport"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data["LogLevel"] = log_level
        data["Message"] = message

        with open("config.json", "w") as file:
            json.dump(data, file, indent=4)

        logs("User Logout", level="info")

    except Exception as e:
        print(f"Gagal memperbarui konfigurasi: {e}")


def tampilkan_data(tree):
    # Clear previous data
    tree.delete(*tree.get_children())
    
    # Get new data
    columns, data, error = FetchData()
    
    if error:
        messagebox.showerror("Kesalahan", error)
    elif data:
        # Configure columns
        tree["columns"] = columns
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor="w", stretch=tk.YES)
        
        # Insert data
        for row in data:
            tree.insert('', 'end', values=row)


def main():
    root = tk.Tk()
    root.title("Status Firewall Saat Ini")
    root.geometry("1200x600")
    
    # Create frame for treeview and scrollbars
    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Create treeview
    tree = ttk.Treeview(frame, height=20)
    
    # Add scrollbars
    vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
    
    # Layout
    tree.grid(row=0, column=0, sticky='nsew')
    vsb.grid(row=0, column=1, sticky='ns')
    hsb.grid(row=1, column=0, sticky='ew')
    
    # Configure grid
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    
    # Button frame
    button_frame = tk.Frame(root)
    button_frame.pack(fill=tk.X, padx=10, pady=5)
    
    refresh_btn = tk.Button(
        button_frame, 
        text="Muat Ulang Data", 
        command=lambda: tampilkan_data(tree),
        padx=10,
        pady=5
    )
    refresh_btn.pack(side=tk.LEFT)
    tombol = tk.Button(root, text="Export Data", command=export,padx=10,pady=5)
    tombol.pack(side=tk.LEFT)
    tampilkan_data(tree)
    
    root.mainloop()
    update_config()


if __name__ == "__main__":
    main()