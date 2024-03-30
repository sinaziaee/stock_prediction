import os
import json

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def save_dict_to_json(dictionary, path):
    with open(path, 'w') as f:
        json.dump(dictionary, f)

def load_dict_from_json(path):
    with open(path, 'r') as f:
        return json.load(f)