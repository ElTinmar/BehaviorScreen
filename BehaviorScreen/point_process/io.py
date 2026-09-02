import re
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

def slugify(name: str) -> str:
    """Turn an arbitrary model/condition name into a filesystem-safe filename fragment."""
    name = name.strip()
    name = re.sub(r"[^\w\-.]+", "_", name)   # anything not alnum/_/-/. -> underscore
    return re.sub(r"_+", "_", name).strip("_")

def save_fig(fig: plt.Figure, out_dir: Path, filename: str, dpi: int = 150) -> Path:
    """Save fig as PNG under out_dir/filename (creating out_dir if needed) and return the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{slugify(filename)}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path

def save_csv(df: pd.DataFrame, out_dir: Path, filename: str) -> Path:
    """Save df as CSV under out_dir/filename.csv (creating out_dir if needed)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{slugify(filename)}.csv"
    df.to_csv(path, index=False)
    return path