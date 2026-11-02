"""
Names of the line folders, from line_names.csv:
    short       plot label (matplotlib math text: $\\mathit{atoh7}^{\\mathit{th241}}$)
    mapzebrain  name in mapzebrain.org
    zfin        ZFIN name of the line (feature or genotype)
    zfin_id     its ZFIN ID (ZDB-ALT-..., ZDB-GENO-...), zfin.org/<zfin_id>
Folders stay the identifiers everywhere else; an empty or missing entry falls
back to the short name, then to the folder.
"""
from pathlib import Path

import pandas as pd

LINE_NAMES = pd.read_csv(Path(__file__).with_name("line_names.csv"), dtype=str, keep_default_na=False).set_index("folder", verify_integrity=True)


def line_name(line: str, column: str = "short") -> str:
    if line not in LINE_NAMES.index:
        return line
    return LINE_NAMES.at[line, column] or LINE_NAMES.at[line, "short"] or line
