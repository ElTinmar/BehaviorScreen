import argparse
import pandas as pd
from pathlib import Path

def create_parser():
    
    parser = argparse.ArgumentParser(description="Concatenate multiple CSV files.")
    parser.add_argument("root", type=Path, help="root dorectory")
    parser.add_argument("-n", "--name", default='bout_frequency.csv', help="input CSV files")
    parser.add_argument("-o", "--output", required=True, help="Output CSV file")
    return parser

def main(args):

    all_files = [f for f in args.root.rglob(args.name)]
    dfs = []
    for f in all_files:
        rel_parts = f.relative_to(args.root).parts          
        line = rel_parts[0]
        condition = rel_parts[1]
        
        print(line, condition)
        df = pd.read_csv(f)        
        df.insert(0, 'line', line)
        df.insert(1, 'condition', condition)
        
        dfs.append(df)

    pd.concat(dfs, ignore_index=True).to_csv(args.output, index=False)

if __name__ == "__main__":

    main(create_parser().parse_args())
