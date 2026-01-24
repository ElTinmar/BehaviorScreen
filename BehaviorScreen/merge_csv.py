import argparse
import pandas as pd

def create_parser():
    
    parser = argparse.ArgumentParser(description="Concatenate multiple CSV files.")
    parser.add_argument("inputs", nargs="+", help="Input CSV files")
    parser.add_argument("-o", "--output", required=True, help="Output CSV file")
    parser.add_argument("--no-header-check", action="store_true", help="Do not enforce matching headers")
    return parser

def main(args):

    dfs = [pd.read_csv(f) for f in args.inputs]

    if not args.no_header_check:
        headers = [list(df.columns) for df in dfs]
        if not all(h == headers[0] for h in headers):
            raise ValueError("CSV headers do not match. Use --no-header-check to override.")

    pd.concat(dfs, ignore_index=True).to_csv(args.output, index=False)

if __name__ == "__main__":

    main(create_parser().parse_args())
