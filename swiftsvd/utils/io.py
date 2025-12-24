import json
from pathlib import Path

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)
    
def ensure_file_path(file_path: str) -> None:
    """
    Ensures that all parent directories of the given file path exist.
    If they don't, creates them recursively.
    
    Args:
        file_path (str): Full path to a file
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)