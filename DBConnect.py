import pyodbc
from JsonFileImporter import GlobalHandler as JsonFileImporter


def GetDBData():
    data=JsonFileImporter("HashedDBInfo.json")
    if isinstance(data, (dict, list)):  
        return data
    else:  
        return 0

def GetDBType():
    try:
        data = GetDBData()
        return data
    except KeyError:
        return "Error: 'dbName' not found in the data."
    except Exception as e:
        return f"Unexpected Error: {str(e)}"

def Connect():
    connection_data = GetDBType()
    if isinstance(connection_data, dict):
        try:
            connection_string = (
                f"DRIVER={connection_data['Driver']};"
                f"SERVER={connection_data['Server']};"
                f"DATABASE={connection_data['Database']};"
                f"UID={connection_data['UID']};"
                f"PWD={connection_data['PWD']};"
                f"Encrypt={connection_data['Encrypt']};"
                f"TrustServerCertificate={connection_data['TrustServerCertificate']};"
            )
            conn = pyodbc.connect(connection_string)
            print("Koneksi ke SQL Server berhasil!")
            conn.close()
            return connection_string
        except KeyError as e:
            return f"Error: Missing key {str(e)} in the database config."
        except Exception as e:
            return f"Unexpected Error: {str(e)}"
    else:
        return "Invalid database configuration format."

def GlobalHandler():
    return Connect()