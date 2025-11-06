import os
import pandas as pd

def get_column_names(csv_path: str):
    """Return column names from a CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}")
    try:
        df = pd.read_csv(csv_path, nrows=1)
        return df.columns.tolist()
    except Exception as e:
        raise RuntimeError(f"Failed to read {csv_path}: {e}")

def ensure_dir(path: str):
    """Create a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def log(message: str):
    """Simple logger."""
    print(f"[JMP Copilot] {message}")
