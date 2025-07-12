CFG_VERSION = "1.0.0"
CFG_VERBOSE = 7

def error(message):
    if (CFG_VERBOSE & 4) != 0:
        print(f"ERROR: {message}")
def warn(message):
    if (CFG_VERBOSE & 2) != 0:
        print(f"WARN: {message}")
def info(message):
    if (CFG_VERBOSE & 1) != 0:
        print(f"INFO: {message}")