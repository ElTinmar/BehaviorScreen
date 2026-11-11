"""
What quality control and the tracking filters remove, per arm: an exclusion that depends on the behavior
(a fish that does not move, a trial tracked badly, a bout away from the centre) and differs between the
arms of a contrast changes the comparison.

Per (line, condition), from the merged tables:

    fish_recorded, fish_excluded and one column per quality-control flag (qc_all.csv)
    trials_presented, trials_untracked    presented trials of the kept fish, and those not tracking_ok
    bouts, bouts_dropped_<column>         bouts in trials of the kept fish, and those each tracking filter
                                          of the models removes (their trial still counts as observed)

and each one's share of its total; the same pooled per condition is printed.

    python -m BehaviorScreen.ablation_screen.exclusions DATA_ROOT [--output ./exclusions.csv]
"""
import argparse
from pathlib import Path

import pandas as pd

from BehaviorScreen.point_process.analysis.model_config import EVENT_TABLES
from BehaviorScreen.stim_specs import parse_rules

ARM = ["line", "condition"]
FLAGS = ["not_moving", "centroid_issue", "heading_issue", "well_not_found", "frame_order_issue"]


def counts(root: Path) -> pd.DataFrame:
    """Per arm: fish recorded and excluded, trials presented and untracked, bouts and bouts dropped by each filter."""
    read = lambda name, **kwargs: pd.read_csv(root / name, dtype={"line": str, "condition": str}, **kwargs)
    filters = EVENT_TABLES["bouts"]["event_filters"]
    trials, qc = read("valid_trials_all.csv"), read("qc_all.csv")
    bouts = read("bouts_all.csv", usecols=["file", *ARM, "epoch_name", *filters]).dropna(subset=["epoch_name"])

    fish = trials.drop_duplicates("file")[["file", *ARM]].merge(qc[["file", *FLAGS]], on="file", how="left")
    fish[FLAGS] = fish[FLAGS].fillna(False).astype(bool)
    fish["excluded"] = fish[FLAGS].any(axis=1)
    kept = fish.loc[~fish["excluded"], "file"]
    trials = trials[trials["file"].isin(kept) & trials["presented"].astype(bool)]
    bouts = bouts[bouts["file"].isin(kept)]
    dropped = pd.DataFrame({f"bouts_dropped_{rule.column}": ~rule.get_mask(bouts).fillna(False).astype(bool) for rule in parse_rules(filters).rules})
    return pd.concat([
        fish.groupby(ARM).size().rename("fish_recorded"),
        fish.groupby(ARM)[["excluded", *FLAGS]].sum().add_prefix("fish_"),
        trials.groupby(ARM).size().rename("trials_presented"),
        (~trials["tracking_ok"].astype(bool)).groupby([trials[c] for c in ARM]).sum().rename("trials_untracked"),
        bouts.groupby(ARM).size().rename("bouts"),
        dropped.groupby([bouts[c] for c in ARM]).sum(),
    ], axis=1).fillna(0).astype(int)


def with_shares(table: pd.DataFrame) -> pd.DataFrame:
    """Each count's share of its total (fish recorded, trials presented, bouts), in share_<column>."""
    totals = {"fish_": "fish_recorded", "trials_": "trials_presented", "bouts_": "bouts"}
    shares = {f"share_{c}": table[c] / table[total] for prefix, total in totals.items() for c in table.columns if c.startswith(prefix) and c != total}
    return table.assign(**shares)


def main(args: argparse.Namespace) -> None:
    table = counts(args.root)
    with_shares(table).reset_index().to_csv(args.output, index=False)
    print(with_shares(table.groupby("condition").sum()).filter(like="share_").T.round(3).to_string())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fish, trials and bouts removed by quality control and tracking filters, per arm.")
    parser.add_argument("root", type=Path, help="Folder of the merged tables (bouts_all.csv, valid_trials_all.csv, qc_all.csv).")
    parser.add_argument("--output", type=Path, default=Path("./exclusions.csv"))
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
