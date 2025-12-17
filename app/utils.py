import os
import pandas as pd

def get_column_names(csv_path: str):
    """grab the header row from a csv (super basic)"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}")
    try:
        df = pd.read_csv(csv_path, nrows=1)
        return df.columns.tolist()
    except Exception as e:
        raise RuntimeError(f"Failed to read {csv_path}: {e}")

def ensure_dir(path: str):
    """make a folder if it isn't there already"""
    os.makedirs(path, exist_ok=True)

def log(message: str):
    """tiny logger because print() is fine"""
    print(f"[JMP Copilot] {message}")
