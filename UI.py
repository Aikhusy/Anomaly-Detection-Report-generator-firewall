import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import os
import pyodbc
from threading import Thread
from datetime import datetime

class DatabaseConnectionManager:
    def __init__(self, root):
        self.root = root
        self.root.title("SQL Server Database Connection Manager")
        self.root.geometry("800x600")
        
        # File untuk menyimpan database info
        self.json_file = "HashedDBInfo.json"
        self.db_connections = []
        self.selected_index = -1
        
        # Load existing connections
        self.load_connections()
        
        # Setup UI
        self.setup_ui()
        
        # Refresh list
        self.refresh_connection_list()
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="SQL Server Database Connection Manager", 
                               font=("Arial", 14, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Connection List Frame
        list_frame = ttk.LabelFrame(main_frame, text="Database Connections", padding="10")
        list_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        list_frame.columnconfigure(0, weight=1)
        
        # Listbox with scrollbar
        listbox_frame = ttk.Frame(list_frame)
        listbox_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        listbox_frame.columnconfigure(0, weight=1)
        listbox_frame.rowconfigure(0, weight=1)
        
        self.connection_listbox = tk.Listbox(listbox_frame, height=8)
        self.connection_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.connection_listbox.bind('<<ListboxSelect>>', self.on_select_connection)
        
        # Scrollbar for listbox
        scrollbar = ttk.Scrollbar(listbox_frame, orient="vertical")
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.connection_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.connection_listbox.yview)
        
        # Connection buttons
        button_frame = ttk.Frame(list_frame)
        button_frame.grid(row=1, column=0, pady=(10, 0))
        
        ttk.Button(button_frame, text="Create New", command=self.create_connection).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Edit Selected", command=self.edit_connection).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Delete Selected", command=self.delete_connection).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Test Connection", command=self.test_connection).pack(side=tk.LEFT, padx=(5, 0))
        
        # Connection Form Frame
        self.form_frame = ttk.LabelFrame(main_frame, text="Connection Details", padding="10")
        self.form_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        self.form_frame.columnconfigure(1, weight=1)
        
        # Form fields
        self.create_form_fields()
        
        # Action buttons
        action_frame = ttk.Frame(main_frame)
        action_frame.grid(row=3, column=0, columnspan=2, pady=(0, 10))
        
        ttk.Button(action_frame, text="Save Connection", command=self.save_connection).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(action_frame, text="Clear Form", command=self.clear_form).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Connect & Return Index", command=self.connect_and_return).pack(side=tk.LEFT, padx=(5, 0))
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Configure grid weights
        main_frame.rowconfigure(1, weight=1)
    
    def create_form_fields(self):
        row = 0
        
        # Driver
        ttk.Label(self.form_frame, text="Driver:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.driver_var = tk.StringVar(value="ODBC Driver 17 for SQL Server")
        driver_combo = ttk.Combobox(self.form_frame, textvariable=self.driver_var, width=40)
        driver_combo['values'] = ("ODBC Driver 17 for SQL Server", "ODBC Driver 13 for SQL Server", "SQL Server")
        driver_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=(10, 0))
        row += 1
        
        # Server
        ttk.Label(self.form_frame, text="Server:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.server_var = tk.StringVar(value="localhost")
        ttk.Entry(self.form_frame, textvariable=self.server_var, width=40).grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=(10, 0))
        row += 1
        
        # Database
        ttk.Label(self.form_frame, text="Database:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.database_var = tk.StringVar()
        ttk.Entry(self.form_frame, textvariable=self.database_var, width=40).grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=(10, 0))
        row += 1
        
        # Trusted Connection
        ttk.Label(self.form_frame, text="Trusted Connection:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.trusted_conn_var = tk.StringVar(value="no")
        trusted_combo = ttk.Combobox(self.form_frame, textvariable=self.trusted_conn_var, width=40)
        trusted_combo['values'] = ("yes", "no")
        trusted_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=(10, 0))
        row += 1
        
        # UID
        ttk.Label(self.form_frame, text="Username (UID):").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.uid_var = tk.StringVar()
        ttk.Entry(self.form_frame, textvariable=self.uid_var, width=40).grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=(10, 0))
        row += 1
        
        # PWD
        ttk.Label(self.form_frame, text="Password (PWD):").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.pwd_var = tk.StringVar()
        ttk.Entry(self.form_frame, textvariable=self.pwd_var, width=40, show="*").grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=(10, 0))
        row += 1
        
        # Encrypt
        ttk.Label(self.form_frame, text="Encrypt:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.encrypt_var = tk.StringVar(value="no")
        encrypt_combo = ttk.Combobox(self.form_frame, textvariable=self.encrypt_var, width=40)
        encrypt_combo['values'] = ("yes", "no")
        encrypt_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=(10, 0))
        row += 1
        
        # TrustServerCertificate
        ttk.Label(self.form_frame, text="Trust Server Certificate:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.trust_cert_var = tk.StringVar(value="no")
        trust_combo = ttk.Combobox(self.form_frame, textvariable=self.trust_cert_var, width=40)
        trust_combo['values'] = ("yes", "no")
        trust_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=(10, 0))
    
    def load_connections(self):
        """Load connections from JSON file"""
        try:
            if os.path.exists(self.json_file):
                with open(self.json_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        # Handle both array format and individual objects
                        if content.startswith('['):
                            self.db_connections = json.loads(content)
                        else:
                            # Handle multiple JSON objects separated by commas
                            content = '[' + content + ']'
                            content = content.replace('},{', '},{')
                            self.db_connections = json.loads(content)
                    else:
                        self.db_connections = []
            else:
                self.db_connections = []
        except Exception as e:
            messagebox.showerror("Error", f"Error loading connections: {str(e)}")
            self.db_connections = []
    
    def save_connections(self):
        """Save connections to JSON file"""
        try:
            with open(self.json_file, 'w') as f:
                json.dump(self.db_connections, f, indent=4)
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Error saving connections: {str(e)}")
            return False
    
    def refresh_connection_list(self):
        """Refresh the connection listbox"""
        self.connection_listbox.delete(0, tk.END)
        for i, conn in enumerate(self.db_connections):
            display_text = f"{i+1}. {conn.get('Server', 'Unknown')} - {conn.get('Database', 'Unknown')}"
            self.connection_listbox.insert(tk.END, display_text)
    
    def on_select_connection(self, event):
        """Handle connection selection"""
        selection = self.connection_listbox.curselection()
        if selection:
            self.selected_index = selection[0]
            conn = self.db_connections[self.selected_index]
            
            # Populate form with selected connection data
            self.driver_var.set(conn.get('Driver', 'ODBC Driver 17 for SQL Server'))
            self.server_var.set(conn.get('Server', ''))
            self.database_var.set(conn.get('Database', ''))
            self.trusted_conn_var.set(conn.get('Trusted_Connection', 'no'))
            self.uid_var.set(conn.get('UID', ''))
            self.pwd_var.set(conn.get('PWD', ''))
            self.encrypt_var.set(conn.get('Encrypt', 'no'))
            self.trust_cert_var.set(conn.get('TrustServerCertificate', 'no'))
            
            self.status_var.set(f"Selected connection {self.selected_index + 1}")
    
    def create_connection(self):
        """Create new connection"""
        self.clear_form()
        self.selected_index = -1
        self.status_var.set("Creating new connection")
    
    def edit_connection(self):
        """Edit selected connection"""
        if self.selected_index >= 0:
            self.status_var.set(f"Editing connection {self.selected_index + 1}")
        else:
            messagebox.showwarning("Warning", "Please select a connection to edit")
    
    def delete_connection(self):
        """Delete selected connection"""
        if self.selected_index >= 0:
            if messagebox.askyesno("Confirm Delete", "Are you sure you want to delete this connection?"):
                del self.db_connections[self.selected_index]
                self.save_connections()
                self.refresh_connection_list()
                self.clear_form()
                self.selected_index = -1
                self.status_var.set("Connection deleted")
        else:
            messagebox.showwarning("Warning", "Please select a connection to delete")
    
    def save_connection(self):
        """Save current connection"""
        if not self.database_var.get().strip():
            messagebox.showerror("Error", "Database name is required")
            return
        
        connection_data = {
            "Driver": self.driver_var.get(),
            "Server": self.server_var.get(),
            "Database": self.database_var.get(),
            "Trusted_Connection": self.trusted_conn_var.get(),
            "UID": self.uid_var.get(),
            "PWD": self.pwd_var.get(),
            "Encrypt": self.encrypt_var.get(),
            "TrustServerCertificate": self.trust_cert_var.get()
        }
        
        if self.selected_index >= 0:
            # Update existing connection
            self.db_connections[self.selected_index] = connection_data
            message = f"Connection {self.selected_index + 1} updated"
        else:
            # Add new connection
            self.db_connections.append(connection_data)
            message = "New connection added"
        
        if self.save_connections():
            self.refresh_connection_list()
            self.status_var.set(message)
            messagebox.showinfo("Success", message)
    
    def clear_form(self):
        """Clear form fields"""
        self.driver_var.set("ODBC Driver 17 for SQL Server")
        self.server_var.set("localhost")
        self.database_var.set("")
        self.trusted_conn_var.set("no")
        self.uid_var.set("")
        self.pwd_var.set("")
        self.encrypt_var.set("no")
        self.trust_cert_var.set("no")
        self.selected_index = -1
    
    def build_connection_string(self, conn_data):
        """Build connection string from connection data"""
        conn_str_parts = []
        
        for key, value in conn_data.items():
            if value:  # Only add non-empty values
                conn_str_parts.append(f"{key}={value}")
        
        return ";".join(conn_str_parts)
    
    def test_connection_thread(self, connection_data, callback):
        """Test connection in separate thread"""
        try:
            conn_str = self.build_connection_string(connection_data)
            connection = pyodbc.connect(conn_str, timeout=10)
            connection.close()
            callback(True, "Connection successful!")
        except Exception as e:
            callback(False, f"Connection failed: {str(e)}")
    
    def test_connection(self):
        """Test selected connection"""
        if self.selected_index < 0:
            messagebox.showwarning("Warning", "Please select a connection to test")
            return
        
        connection_data = self.db_connections[self.selected_index]
        self.status_var.set("Testing connection...")
        
        def callback(success, message):
            self.root.after(0, lambda: self.test_connection_callback(success, message))
        
        thread = Thread(target=self.test_connection_thread, args=(connection_data, callback))
        thread.daemon = True
        thread.start()
    
    def test_connection_callback(self, success, message):
        """Callback for test connection result"""
        self.status_var.set(message)
        if success:
            messagebox.showinfo("Connection Test", message)
        else:
            messagebox.showerror("Connection Test", message)
    
    def connect_and_return(self):
        """Connect to selected database and return index"""
        if self.selected_index < 0:
            messagebox.showwarning("Warning", "Please select a connection")
            return
        
        connection_data = self.db_connections[self.selected_index]
        self.status_var.set("Connecting...")
        
        def callback(success, message):
            self.root.after(0, lambda: self.connect_and_return_callback(success, message))
        
        thread = Thread(target=self.test_connection_thread, args=(connection_data, callback))
        thread.daemon = True
        thread.start()
    
    def connect_and_return_callback(self, success, message):
        """Callback for connect and return result"""
        if success:
            self.status_var.set(f"Connected successfully! Selected index: {self.selected_index}")
            
            # Open the current status window
            self.open_current_status_window()
            
            return self.selected_index
        else:
            self.status_var.set("Connection failed")
            messagebox.showerror("Connection Error", message)
            return -1
    
    def open_current_status_window(self):
        """Open window to display current status data"""
        status_window = FirewallStatusWindow(self.root, self.db_connections[self.selected_index], self.selected_index)
        status_window.show()

from Process import ExportToPDF as ExportToPdf
class FirewallStatusWindow:
    def __init__(self, parent, connection_data, connection_index):
        self.parent = parent
        self.connection_data = connection_data
        self.connection_index = connection_index
        self.window = None
        
    def show(self):
        """Show the firewall status window"""
        self.window = tk.Toplevel(self.parent)
        self.window.title(f"Status Firewall - {self.connection_data['Database']}")
        self.window.geometry("1200x600")
        
        # Make window modal
        self.window.transient(self.parent)
        self.window.grab_set()
        
        self.setup_ui()
        self.load_data()
        
    def setup_ui(self):
        # Header frame
        header_frame = ttk.Frame(self.window, padding="10")
        header_frame.pack(fill=tk.X)
        
        # Title and connection info
        title_label = ttk.Label(header_frame, 
                               text=f"Status Firewall Saat Ini - {self.connection_data['Database']}", 
                               font=("Arial", 14, "bold"))
        title_label.pack(anchor=tk.W)
        
        conn_info_label = ttk.Label(header_frame, 
                                   text=f"Server: {self.connection_data['Server']} | Index: {self.connection_index}",
                                   font=("Arial", 10))
        conn_info_label.pack(anchor=tk.W)
        
        # Create frame for treeview and scrollbars
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create treeview
        self.tree = ttk.Treeview(main_frame, height=20)
        
        # Add scrollbars
        vsb = ttk.Scrollbar(main_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(main_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Layout
        self.tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        
        # Configure grid
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Button frame
        button_frame = ttk.Frame(self.window)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(button_frame, text="Muat Ulang Data", 
                  command=self.refresh_data).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Export Data", 
                  command=self.export_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Tutup", 
                  command=self.close_window).pack(side=tk.RIGHT)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.window, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, padx=10, pady=(0, 5))
    
    def build_connection_string(self):
        """Build connection string for current database"""
        conn_str_parts = []
        for key, value in self.connection_data.items():
            if value:
                conn_str_parts.append(f"{key}={value}")
        return ";".join(conn_str_parts)
    
    def fetch_data(self):
        """Fetch firewall status data from database"""
        try:
            conn_str = self.build_connection_string()
            conn = pyodbc.connect(conn_str, timeout=10)
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
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            conn.close()

            # Convert all data to strings to avoid TclError
            converted_data = []
            for row in data:
                converted_data.append(tuple(str(item) if item is not None else "" for item in row))

            return columns, converted_data, None

        except Exception as e:
            return None, None, str(e)
    
    def load_data(self):
        """Load data into treeview"""
        self.status_var.set("Loading data...")
        
        # Clear previous data
        self.tree.delete(*self.tree.get_children())
        
        # Get new data
        columns, data, error = self.fetch_data()
        
        if error:
            self.status_var.set("Error loading data")
            messagebox.showerror("Kesalahan", f"Error loading data: {error}")
        elif data:
            # Configure columns
            self.tree["columns"] = columns
            self.tree["show"] = "headings"  # Hide the default first column
            
            for col in columns:
                self.tree.heading(col, text=col)
                self.tree.column(col, width=120, anchor="w", stretch=tk.YES)
            
            # Insert data
            for row in data:
                self.tree.insert('', 'end', values=row)
            
            self.status_var.set(f"Loaded {len(data)} records")
        else:
            self.status_var.set("No data found")
    
    def refresh_data(self):
        """Refresh data in treeview"""
        self.load_data()
    
    def export_data(self):
        """Export data to file"""
        try:
            # Get data from treeview
            data = []
            columns = []
            
            if self.tree["columns"]:
                columns = list(self.tree["columns"])
                for item in self.tree.get_children():
                    values = self.tree.item(item)["values"]
                    data.append(values)
            
            if not data:
                messagebox.showwarning("Warning", "No data to export")
                return
            
            # Create export data structure
            export_data = {
                "database": self.connection_data['Database'],
                "server": self.connection_data['Server'],
                "export_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "connection_index": self.connection_index,
                "columns": columns,
                "data": data
            }
            
            # Save to JSON file
            filename = f"firewall_status_{self.connection_data['Database']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=4, ensure_ascii=False)

            ExportToPdf(
                report_time=datetime.now(),
                start_date="2025-01-01",
                end_date=datetime.now().strftime("%Y-%m-%d"),
                connection_data=self.connection_data
            )

            messagebox.showinfo("Export Success", f"Data exported to {filename}")
            self.status_var.set(f"Data exported to {filename}")            
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data: {str(e)}")
    
    def close_window(self):
        """Close the status window"""
        self.window.destroy()


def main():
    root = tk.Tk()
    app = DatabaseConnectionManager(root)
    root.mainloop()

if __name__ == "__main__":
    main()