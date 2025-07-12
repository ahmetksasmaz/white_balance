import json
from configuration import *

class Database:
    def __init__(self, filename='database.json'):
        self.filename = filename
        self.data = {}
        self.saved_updates = True
        self.__load()
    
    def __load(self):
        try:
            with open(self.filename, 'r') as file:
                self.data = json.load(file)
                info(f"Database loaded from '{self.filename}'.")
        except FileNotFoundError:
            error(f"Database file '{self.filename}' not found. Exiting.")
            exit(1)
        except json.JSONDecodeError:
            error(f"Database file '{self.filename}' is not a valid JSON. Exiting.")
            exit(1)
    
    def save(self):
        try:
            with open(self.filename, 'w') as file:
                json.dump(self.data, file, indent=4)
            self.saved_updates = True
            info(f"Updates saved to database: '{self.filename}'.")
        except IOError as e:
            error(f"Failed to save database file '{self.filename}': {e}. Exiting.")
            exit(1)