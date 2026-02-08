import os

def scan_project(folder_path: str):
    files = []

    for root, dirs, filenames in os.walk(folder_path):
        for name in filenames:
            files.append(os.path.join(root, name))

    return files
