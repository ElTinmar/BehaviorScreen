from typing import Any
from pathlib import Path
from dataclasses import dataclass
import operator
import yaml
import argparse
import pandas as pd

def pd_series_in(s: pd.Series, v: Any) -> pd.Series:
    return s.isin(v)

def pd_series_not_in(s: pd.Series, v: Any) -> pd.Series:
    return ~s.isin(v)

_OPS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
    "in": pd_series_in,
    "not_in": pd_series_not_in,
}

@dataclass
class Rule:
    column: str
    operator: str
    value: Any

    def get_mask(self, df: pd.DataFrame) -> pd.Series:
        op_func = _OPS[self.operator]
        return op_func(df[self.column], self.value)

@dataclass
class RuleSet: 
    rules: tuple[Rule, ...]

    def get_mask(self, df: pd.DataFrame) -> pd.Series:
        mask = pd.Series(True, index=df.index)

        for rule in self.rules:
            mask &= rule.get_mask(df)

        return mask
    
    def __repr__(self):
        return "_".join([f"{r.column}{r.operator}{r.value}" for r in self.rules])
    

def load_bouts(bout_csv: Path) -> pd.DataFrame:
    return pd.read_csv(bout_csv)

def parse_rules(cfg: dict) -> RuleSet:
    rules = []

    for column, rule_dict in cfg.items():
        for op_name, value in rule_dict.items():
            rules.append(Rule(column, op_name, value))

    return RuleSet(tuple(rules))

def filter_bouts(quality_control: Path, bouts: pd.DataFrame, cfg: dict) -> pd.DataFrame:

    filtered = bouts.copy()
    n0 = len(filtered)
    print(f'TOTAL NUM BOUTS: {n0}')

    if quality_control.exists():
        qc = pd.read_csv(quality_control)
        before = len(filtered)
        filtered = filtered[~filtered['file'].isin(qc['file'])]
        after = len(filtered)
        removed = before - after
        frac_total = removed / n0 if n0 else 0
        print(f"Quality control → removed {removed:6d} ({frac_total:6.2%})")

    filters = parse_rules(cfg)
    for rule in filters.rules:
        before = len(filtered)
        mask = rule.get_mask(filtered)
        filtered = filtered[mask]
        after = len(filtered)
        removed = before - after
        frac_total = removed / n0 if n0 else 0
        print(f"{rule.column:25s} {rule.operator:>2} {rule.value} → removed {removed:6d} ({frac_total:6.2%})")

    return filtered

def load_yaml_config(path: Path) -> dict:
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_parser() -> argparse.ArgumentParser:
    
    parser = argparse.ArgumentParser(
        description="Filter bouts"
    )

    parser.add_argument(
        "root",
        type=Path,
        help="Directory containing bouts.csv, qc.csv",
    )

    parser.add_argument(
        "--input",
        default='bouts.csv',
        help="input CSV file",
    )

    parser.add_argument(
        "--yaml",
        default='BehaviorScreen/filters.yaml',
        help="filter config file",
    )

    parser.add_argument(
        "--qc",
        default='qc.csv',
        help="quality control: remove fish not moving or with tracking issues",
    )
    
    parser.add_argument(
        "--output",
        default='bouts_filtered.csv',
        help="output CSV filtered bouts",
    )

    return parser

def filter(
        input_csv: Path,
        config_yaml: Path,
        quality_control: Path,
        output_csv: Path
    ) -> None:

    cfg = load_yaml_config(config_yaml)
    bouts = load_bouts(input_csv)
    filtered_bouts = filter_bouts(quality_control, bouts, cfg)
    filtered_bouts.to_csv(output_csv, index=False)

def main(args: argparse.Namespace) -> None:

    input_csv = args.root / args.input
    quality_control = args.root / args.qc
    output_csv = args.root / args.output
    config_yaml = Path(args.yaml)

    filter(
        input_csv,
        config_yaml,
        quality_control,
        output_csv
    )


if __name__ == "__main__":

    main(build_parser().parse_args())
