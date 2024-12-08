import odbc 
from JsonFileImporter import GlobalHandler as JsonFileImporter


def GetDBData():
    data=JsonFileImporter("HashedDBInfo.json")
    print(data)
    if isinstance(data, (dict, list)):  
        return data
    else:  
        return 0

def GetDBType(index):
    try:
        data = GetDBData()
        print(data)
        if 0 <= index < len(data):
            return data[index]["dbName"]
        else:
            return "Error: Index out of range."
    except KeyError:
        return "Error: 'dbName' not found in the data."
    except Exception as e:
        return f"Unexpected Error: {str(e)}"

def GlobalHandler():
    return GetDBType(0)