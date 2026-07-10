import argparse
import pandas as pd
from pathlib import Path

def create_parser():
    parser = argparse.ArgumentParser(description="Concatenate multiple CSV files.")
    parser.add_argument("root", type=Path, help="Root directory containing the files")
    parser.add_argument("inputs", nargs="+", help="Input CSV files (relative to root)")
    parser.add_argument("-o", "--output", required=True, help="Output CSV file (relative to root or absolute)")
    parser.add_argument("--no-header-check", action="store_true", help="Do not enforce matching headers")
    return parser

def main(args):

    output = Path(args.output)
    if not output.is_absolute():
        output = args.root / output
    
    dfs = []
    headers = None

    for input_str in args.inputs:
        input_file = args.root / input_str
        
        try:
            rel_parts = input_file.relative_to(args.root).parts
            if len(rel_parts) < 2:
                raise ValueError(f"File {input_str} must be at least 2 levels deep inside the root directory.")
            line, condition = rel_parts[0], rel_parts[1]
        except ValueError as e:
            print(f"Skipping {input_str}: {e}")
            continue

        df = pd.read_csv(input_file)        
        
        if not args.no_header_check:
            current_headers = list(df.columns)
            if headers is None:
                headers = current_headers
            elif current_headers != headers:
                raise ValueError(f"CSV headers do not match in {input_str}. Use --no-header-check to override.")

        df.insert(0, 'line', line)
        df.insert(1, 'condition', condition)
        dfs.append(df)

    if not dfs:
        print("No valid DataFrames to concatenate.")
        return

    print(f"Concatenating {len(dfs)} files into {output}...")
    pd.concat(dfs, ignore_index=True).to_csv(output, index=False)

if __name__ == "__main__":
    main(create_parser().parse_args())