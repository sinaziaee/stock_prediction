import os

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path