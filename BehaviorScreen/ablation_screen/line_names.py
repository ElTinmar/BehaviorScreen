"""
Names of the screen's lines, from two tables edited as spreadsheets and
exported as CSV UTF-8:

transgenes.csv, one row per transgene, mutant allele or background:
    transgene   the lab's handle for it, as used in lines.csv
    type        one of TYPES; gene-named types are set in italics in plot labels
    mpn, mapzebrain, zfin, zfin_id, notes

lines.csv, one row per line folder:
    folder      the data folder: the line's identifier everywhere in the code
    short       plot label; ^ starts the allele superscript (atoh7^th241), & joins intersected drivers
    driver, recombinase, effector, mutant, background
                what the recorded larvae carry: one transgene of transgenes.csv per cell
    notes

A label is italic when the line's driver, recombinase and mutant all have
gene-named types; then the gene part of each &-joined piece is italic
(atf5b:QF2 -> italic atf5b, upright :QF2).
"""
import warnings
from pathlib import Path
from typing import Iterable, List

import pandas as pd

ROLES = ["driver", "recombinase", "effector", "mutant", "background"]
TYPES = {"enhancer trap", "gene trap", "MPN line", "CRISPR knock-in", "BAC Tg", "gene driver",
         "ENU point mutation", "mutant", "Tg", "wild type"}
ITALIC_TYPES = {"CRISPR knock-in", "BAC Tg", "gene driver", "ENU point mutation", "mutant"}
NAMED_ROLES = ["driver", "recombinase", "mutant"]  # the transgenes a label names


def _read(name: str, columns: List[str], index: str) -> pd.DataFrame:
    table = pd.read_csv(Path(__file__).with_name(name), dtype=str, keep_default_na=False, encoding="utf-8-sig")
    table = table.rename(columns=str.strip).apply(lambda column: column.str.strip())  # spreadsheets leave stray spaces
    if missing := set(columns) - set(table.columns):
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    return table.set_index(index, verify_integrity=True)


TRANSGENES = _read("transgenes.csv", ["transgene", "type", "mpn", "mapzebrain", "zfin", "zfin_id"], "transgene")
LINES = _read("lines.csv", ["folder", "short"] + ROLES, "folder")
if unknown := set(TRANSGENES["type"]) - TYPES - {""}:
    raise ValueError(f"transgenes.csv: unknown type {sorted(unknown)}; use one of {sorted(TYPES)}")
if undefined := sorted(set(LINES[ROLES].to_numpy().ravel()) - set(TRANSGENES.index) - {""}):
    raise ValueError(f"lines.csv names transgenes missing from transgenes.csv: {undefined}")


def _italic(label: str) -> str:
    """'atoh7^th241' or 'mafaa & ebf1b:Cre' -> matplotlib text with the gene names (and alleles) in italics."""
    def piece(part: str) -> str:
        gene, colon, construct = part.strip().partition(":")
        gene, caret, allele = gene.partition("^")
        return rf"$\mathit{{{gene}}}" + (rf"^{{\mathit{{{allele}}}}}" if caret else "") + "$" + colon + construct
    return r" $\cap$ ".join(piece(part) for part in label.split("&"))


def is_italic(line: str) -> bool:
    types = [TRANSGENES.at[t, "type"] for t in LINES.loc[line, NAMED_ROLES] if t]
    return bool(types) and all(t in ITALIC_TYPES for t in types)


def line_name(line: str, column: str = "short") -> str:
    """
    Plot label (column 'short'), or the line's transgenes joined by '; ' by
    their mpn, mapzebrain, zfin or zfin_id entry (their handle where empty).
    Folders without a row keep their folder name.
    """
    if line not in LINES.index:
        return line
    if column == "short":
        short = LINES.at[line, "short"] or line
        return _italic(short) if is_italic(line) else short
    return "; ".join(TRANSGENES.at[t, column] or t for t in LINES.loc[line, ROLES] if t)


def warn_unnamed(lines: Iterable[str]) -> None:
    """Warn about line folders in the data without a row in lines.csv (their plots show the folder name)."""
    if unnamed := sorted(set(lines) - set(LINES.index)):
        warnings.warn(f"lines.csv has no row for {unnamed}")
