import json
import os

def AbsolutePathJsonFileImporter(path):
    if os.path.isabs(path):  
        try:
            with open(path, 'r') as file:
                data = json.load(file)  
            return data
        except FileNotFoundError:
            return "Error: File not found."
        except json.JSONDecodeError:
            return "Error: Failed to decode JSON."
    else:
        return "Error: Provided path is not an absolute path."

def RelativePathJsonFileImporter(path):
    if not os.path.isabs(path):  
        try:
            with open(path, 'r') as file:
                data = json.load(file)  
            return data
        except FileNotFoundError:
            return "Error: File not found."
        except json.JSONDecodeError:
            return "Error: Failed to decode JSON."
    else:
        return "Error: Provided path is not a relative path."

def JsonPathImporterHandler(path):
    
    result = AbsolutePathJsonFileImporter(path) 
    result2 = RelativePathJsonFileImporter(path)

    if isinstance(result, (dict, list)):  
        return result
    
    if isinstance(result2, (dict, list)):  
        return result2
    
    return "Error: Not a valid relative or absolute path"


        
def GlobalHandler(path):
    return JsonPathImporterHandler(path)