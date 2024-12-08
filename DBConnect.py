import odbc 
from JsonFileImporter import GlobalHandler as JsonFileImporter


def TranslateHash():
    JsonFileImporter("HashedDBInfo.Json")
    return True

def GlobalHandler():
    return TranslateHash()