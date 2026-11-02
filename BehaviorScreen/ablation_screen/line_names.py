"""
Display names of the line folders, from line_names.csv (folder, short, full):
short for plots (matplotlib math text: $\mathit{atoh7}^{\mathit{th241}}$), full for the methods
table. Folders stay the identifiers everywhere else; an empty or missing
entry falls back to the short name, then to the folder.
"""
from pathlib import Path

import pandas as pd

LINE_NAMES = pd.read_csv(Path(__file__).with_name("line_names.csv"), dtype=str, keep_default_na=False).set_index("folder", verify_integrity=True)


def line_name(line: str, full: bool = False) -> str:
    if line not in LINE_NAMES.index:
        return line
    short, long = LINE_NAMES.loc[line, ["short", "full"]]
    return (full and long) or short or line
