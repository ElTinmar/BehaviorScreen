import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


def create_parser():
    parser = argparse.ArgumentParser(description="Concatenate multiple CSV files.")
    parser.add_argument("root", type=Path, help="Root directory containing the files")
    parser.add_argument("inputs", nargs="+", help="Input CSV files (relative to root)")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output CSV file (relative to root or absolute)",
    )
    parser.add_argument(
        "--no-header-check", action="store_true", help="Do not enforce matching headers"
    )
    return parser


def merge(root: Path, files: Iterable[Path], check_headers: bool = True) -> pd.DataFrame:
    """The files' tables stacked, each with its line and condition: the first two folders below root."""
    tables, headers = [], None
    for file in files:
        parts = file.relative_to(root).parts
        if len(parts) < 3:
            print(f"Skipping {file}: not in a <line>/<condition> folder of {root}")
            continue
        table = pd.read_csv(file)
        if check_headers and headers is not None and list(table.columns) != headers:
            raise ValueError(f"CSV headers do not match in {file}. Use --no-header-check to override.")
        headers = list(table.columns)
        table.insert(0, "line", parts[0])
        table.insert(1, "condition", parts[1])
        tables.append(table)
    if not tables:
        raise ValueError(f"no table to merge under {root}")
    return pd.concat(tables, ignore_index=True)


def main(args):
    output = Path(args.output)
    if not output.is_absolute():
        output = args.root / output
    print(f"Concatenating into {output}...")
    merge(args.root, [args.root / i for i in args.inputs], not args.no_header_check).to_csv(output, index=False)


if __name__ == "__main__":
    main(create_parser().parse_args())
