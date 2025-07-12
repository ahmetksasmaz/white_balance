import argparse
import os
import sys
from configuration import *
from database import Database

def show_help():
    print("Available commands:")
    print("  help - Show this help message")
    print("  exit - Exit the tool")
    print("  save - Save the current state of the database")

def main():
    global CFG_VERBOSE
    parser = argparse.ArgumentParser(description="White Balance Experimentation Tool")
    parser.add_argument('--version', action='version', version=CFG_VERSION, help='Show the version of the tool')
    parser.add_argument('--verbose', type=int, help='Verbosity code ([ERROR_BIT][WARN_BIT][INFO_BIT]) 7 -> error, warn, info | 4 -> error | 6 error, warn', default=CFG_VERBOSE)
    parser.add_argument('--database', type=str, default='database.json', help='Path to the database file')

    args = parser.parse_args()

    if args.verbose < 0 or args.verbose > 7:
        print("Invalid verbosity code. Must be between 0 and 7.")
        sys.exit(1)
    CFG_VERBOSE = args.verbose

    database = Database(filename=args.database)

    while True:
        if not database.saved_updates:
            warn("Database has unsaved changes.")
        command = input("Enter command (or 'exit' to quit): ").strip().lower()
        if command == 'exit':
            print("Exiting the tool. Goodbye!")
            break
        elif command == 'help':
            show_help()
        elif command == 'save':
            database.save()
        else:
            warn(f"Unknown command: {command}. Type 'help' for a list of commands.")

if __name__ == "__main__":
    main()