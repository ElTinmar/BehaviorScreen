"""
Names of the screen's lines, from two tables edited as spreadsheets and
exported as CSV UTF-8:

facility_lines.csv, one row per line kept in the fish facility:
    line          its name in the facility (as in the cross log)
    type          one of TYPES; gene-named types are set in italics in plot labels
    facility_ids  its stock IDs in the facility database
    mpn, zfin, zfin_id, mapzebrain, notes
    Every line needs an mpn or a ZFIN ID: python -m BehaviorScreen.ablation_screen.line_names
    lists the ones to ask for.

screen_lines.csv, one row per data folder:
    folder              the line's identifier everywhere in the code
    short               plot label; ^ starts the allele superscript (atoh7^th241), & joins intersected drivers
    parent_1, parent_2  the facility lines crossed to make the larvae (parent_2 empty for an in-cross)
    notes

A label is italic when its parents, effector and wild-type lines aside, all
have gene-named types; then the gene part of each &-joined piece is italic
(atf5b:QF2 -> italic atf5b, upright :QF2).
"""
import warnings
from pathlib import Path
from typing import Iterable, List

import pandas as pd

TYPES = {"enhancer trap", "gene trap", "MPN line", "CRISPR knock-in", "BAC Tg", "gene driver",
         "ENU point mutation", "mutant", "Tg", "wild type"}
ITALIC_TYPES = {"CRISPR knock-in", "BAC Tg", "gene driver", "ENU point mutation", "mutant"}
UNNAMED_TYPES = {"Tg", "wild type"}  # effectors and backgrounds: never what a label names
PARENTS = ["parent_1", "parent_2"]


def _read(name: str, columns: List[str], index: str) -> pd.DataFrame:
    table = pd.read_csv(Path(__file__).with_name(name), dtype=str, keep_default_na=False, encoding="utf-8-sig")
    table = table.rename(columns=str.strip).apply(lambda column: column.str.strip())  # spreadsheets leave stray spaces
    if missing := set(columns) - set(table.columns):
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    if duplicated := sorted(set(table.loc[table[index].duplicated(), index])):
        raise ValueError(f"{name}: {index} listed twice: {duplicated}")
    return table.set_index(index)


FACILITY = _read("facility_lines.csv", ["line", "type", "facility_ids", "mpn", "zfin", "zfin_id", "mapzebrain"], "line")
SCREEN = _read("screen_lines.csv", ["folder", "short"] + PARENTS, "folder")
if unknown := set(FACILITY["type"]) - TYPES - {""}:
    raise ValueError(f"facility_lines.csv: unknown type {sorted(unknown)}; use one of {sorted(TYPES)}")
if undefined := sorted(set(SCREEN[PARENTS].to_numpy().ravel()) - set(FACILITY.index) - {""}):
    raise ValueError(f"screen_lines.csv names lines missing from facility_lines.csv: {undefined}")


def _italic(label: str) -> str:
    """'atoh7^th241' or 'mafaa & ebf1b:Cre' -> matplotlib text with the gene names (and alleles) in italics."""
    def piece(part: str) -> str:
        gene, colon, construct = part.strip().partition(":")
        gene, caret, allele = gene.partition("^")
        return rf"$\mathit{{{gene}}}" + (rf"^{{\mathit{{{allele}}}}}" if caret else "") + "$" + colon + construct
    return r" $\cap$ ".join(piece(part) for part in label.split("&"))


def _parents(folder: str) -> List[str]:
    return [p for p in SCREEN.loc[folder, PARENTS] if p]


def is_italic(folder: str) -> bool:
    named = [t for t in FACILITY.loc[_parents(folder), "type"] if t not in UNNAMED_TYPES]
    return bool(named) and all(t in ITALIC_TYPES for t in named)


def line_name(folder: str, column: str = "short") -> str:
    """
    Plot label (column 'short'), or the cross by its parents' entries in a
    facility_lines.csv column (mpn, zfin, zfin_id, mapzebrain; their name where
    empty), e.g. 'mpn202Tg x 81C'. Folders without a row keep their folder name.
    """
    if folder not in SCREEN.index:
        return folder
    if column == "short":
        short = SCREEN.at[folder, "short"] or folder
        return _italic(short) if is_italic(folder) else short
    return " x ".join(FACILITY.at[p, column] or p for p in _parents(folder))


def warn_unnamed(folders: Iterable[str]) -> None:
    """Warn about data folders without a row in screen_lines.csv (their plots show the folder name)."""
    if unnamed := sorted(set(folders) - set(SCREEN.index)):
        warnings.warn(f"screen_lines.csv has no row for {unnamed}")


def unregistered() -> pd.DataFrame:
    """Facility lines (wild types aside) with neither an mpn nor a ZFIN ID: the ones to ask for."""
    lacking = (FACILITY["mpn"] == "") & (FACILITY["zfin_id"] == "") & (FACILITY["type"] != "wild type")
    return FACILITY.loc[lacking, ["type", "facility_ids", "notes"]]


if __name__ == "__main__":
    print(unregistered().to_string())
