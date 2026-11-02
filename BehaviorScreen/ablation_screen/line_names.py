"""
Names of the line folders, from line_names.csv (edited as a spreadsheet,
exported as CSV UTF-8):
    folder      the line's data folder: its identifier everywhere in the code
    short       plot label; ^ starts the allele superscript (atoh7^th241), & joins intersected drivers
    type        one of LINE_TYPES; gene drivers, intersectional lines and mutants are set in italics
    mapzebrain  name in mapzebrain.org
    zfin        ZFIN name of the line
    zfin_id     its ZFIN ID: zfin.org/<zfin_id>
An empty entry falls back to the short name, then to the folder.
"""
import warnings
from pathlib import Path
from typing import Iterable

import pandas as pd

LINE_TYPES = {"gene driver", "intersectional", "mutant", "enhancer trap", "MPN line", "wild type"}
ITALIC_TYPES = {"gene driver", "intersectional", "mutant"}
COLUMNS = ["folder", "short", "type", "mapzebrain", "zfin", "zfin_id"]

LINE_NAMES = pd.read_csv(Path(__file__).with_name("line_names.csv"), dtype=str, keep_default_na=False, encoding="utf-8-sig")
if missing := set(COLUMNS) - set(LINE_NAMES.columns):
    raise ValueError(f"line_names.csv: missing columns {sorted(missing)}")
if unknown := set(LINE_NAMES["type"]) - LINE_TYPES:
    raise ValueError(f"line_names.csv: unknown types {sorted(unknown)}; use one of {sorted(LINE_TYPES)}")
LINE_NAMES = LINE_NAMES.set_index("folder", verify_integrity=True)


def _italic(name: str) -> str:
    """'atoh7^th241' -> matplotlib math text with gene and allele in italics."""
    italic = lambda part: r" \cap ".join(rf"\mathit{{{word.strip()}}}" for word in part.split("&"))
    gene, _, allele = name.partition("^")
    return f"${italic(gene)}" + (f"^{{{italic(allele)}}}" if allele else "") + "$"


def line_name(line: str, column: str = "short") -> str:
    if line not in LINE_NAMES.index:
        return line
    row = LINE_NAMES.loc[line]
    if column == "short":
        return _italic(row["short"] or line) if row["type"] in ITALIC_TYPES else row["short"] or line
    return row[column] or row["short"] or line


def warn_unnamed(lines: Iterable[str]) -> None:
    """Warn about line folders in the data without a row in line_names.csv (their plots show the folder name)."""
    if unnamed := sorted(set(lines) - set(LINE_NAMES.index)):
        warnings.warn(f"line_names.csv has no row for {unnamed}")
