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
    
    if os.path.isabs(path):
        result = AbsolutePathJsonFileImporter(path)
        if isinstance(result, dict):  
            return result
        else:  
            return f"AbsolutePath Error: {result}"
    else:
        
        result = RelativePathJsonFileImporter(path)
        if isinstance(result, dict):  
            return result
        else:  
            return f"RelativePath Error: {result}"
        
def GlobalHandler(path):
    JsonPathImporterHandler(path)