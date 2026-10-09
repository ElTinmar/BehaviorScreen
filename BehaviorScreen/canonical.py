"""Canonical events, participation and fish tables: the one data layer every screen analysis reads."""

import argparse
from pathlib import Path

import pandas as pd
from megabouts.utils import bouts_category_name_short

from BehaviorScreen.stim_specs import (
    FILTER_SECTIONS,
    load_valid_trials,
    load_yaml_config,
    parse_rules,
)

TRIAL_KEY = ["file", "epoch_name", "trial_num"]
FISH_COLUMNS = ["line", "condition", "dpf", "day", "cos_daytime", "sin_daytime"]
QC_COLUMNS = ["not_moving", "centroid_issue", "heading_issue"]
TABLES = ("events", "participation", "fish")


def load_participation(path: Path) -> pd.DataFrame:
    trials = load_valid_trials(path)
    if trials.duplicated(TRIAL_KEY).any():
        raise ValueError(f"{path}: duplicated (file, epoch_name, trial_num) rows")
    origin = trials.groupby("file")["start_timestamp"].transform("min")
    for edge in ("start", "stop"):
        trials[f"t_{edge}_s"] = (trials[f"{edge}_timestamp"] - origin) * 1e-9
    return trials


def load_events(path: Path, participation: pd.DataFrame, config: dict) -> pd.DataFrame:
    events = pd.read_csv(path, dtype={"line": str})
    origin = participation.groupby("file")["start_timestamp"].min()
    unknown = set(events["file"]) - set(origin.index)
    if unknown:
        raise ValueError(
            f"{path}: {len(unknown)} files missing from participation, e.g. {sorted(unknown)[:3]}"
        )

    events["t_start_s"] = (
        events["event_timestamp"] - events["file"].map(origin)
    ) * 1e-9
    events["t_stop_s"] = (
        events["t_start_s"] + events["time_stop"] - events["time_start"]
    )
    events["bout_type"] = events["category"].map(
        dict(enumerate(bouts_category_name_short))
    )
    events = events.sort_values(["file", "t_start_s"], ignore_index=True)

    previous = events.groupby("file")[["t_stop_s", "bout_type"]].shift()
    events["age_s"] = events["t_start_s"] - previous["t_stop_s"]
    events["previous_type"] = previous["bout_type"]

    for section in FILTER_SECTIONS["bout"]:
        for rule in parse_rules(config.get(section)).rules:
            name = f"ok_{rule.column}"
            mask = rule.get_mask(events).fillna(False).astype(bool)
            events[name] = mask & events[name] if name in events else mask
    events["ok"] = events.filter(regex="^ok_").all(axis=1)

    usable = participation.set_index(TRIAL_KEY)["usable"]
    events["usable_trial"] = (
        usable.reindex(pd.MultiIndex.from_frame(events[TRIAL_KEY]))
        .fillna(False)
        .to_numpy(dtype=bool)
    )
    return events


def build_fish(events: pd.DataFrame, qc_path: Path) -> pd.DataFrame:
    per_file = events.groupby("file")[FISH_COLUMNS]
    varying = per_file.nunique().gt(1).any(axis=1)
    if varying.any():
        raise ValueError(
            f"fish metadata varies within files: {list(varying[varying].index)}"
        )

    fish = per_file.first()
    fish["clutch"] = fish["line"] + "_" + fish["day"].astype(str)
    fish = fish.join(pd.read_csv(qc_path, index_col="file")[QC_COLUMNS])
    fish[QC_COLUMNS] = fish[QC_COLUMNS].fillna(False).astype(bool)
    fish["qc_excluded"] = fish[QC_COLUMNS].any(axis=1)
    return fish.reset_index()


def summarize(
    events: pd.DataFrame, participation: pd.DataFrame, fish: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    arms = participation.merge(fish, on="file", how="left", validate="many_to_one")
    kept = arms[arms["usable"] & arms["qc_excluded"].eq(False)]
    flags = [column for column in events if column.startswith("ok_")] + [
        "ok",
        "usable_trial",
    ]
    retained = events[events["ok"] & events["usable_trial"]]
    return {
        "fish_without_bouts": pd.DataFrame(
            {"file": sorted(set(participation["file"]) - set(fish["file"]))}
        ),
        "presentations": kept.groupby(["line", "condition", "epoch_name"])
        .agg(n_fish=("file", "nunique"), n_usable=("file", "size"))
        .reset_index(),
        "qc_exclusion": fish.groupby(["line", "condition"])["qc_excluded"]
        .agg(["sum", "size"])
        .reset_index(),
        "retention": events.groupby(["condition", "bout_type"])[flags]
        .mean()
        .reset_index(),
        "laterality": retained.groupby(["epoch_name", "bout_type", "laterality"])
        .size()
        .unstack(fill_value=0)
        .reset_index(),
    }


def load_tables(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return tuple(pd.read_parquet(directory / f"{name}.parquet") for name in TABLES)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the canonical events, participation and fish tables."
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("yaml", type=Path)
    parser.add_argument("--bouts-csv", default="bouts.csv")
    parser.add_argument("--valid-trials-csv", default="valid_trials.csv")
    parser.add_argument("--qc-csv", default="qc.csv")
    parser.add_argument("--out", default="tables")
    args = parser.parse_args()

    participation = load_participation(args.root / args.valid_trials_csv)
    events = load_events(
        args.root / args.bouts_csv, participation, load_yaml_config(args.yaml)
    )
    fish = build_fish(events, args.root / args.qc_csv)

    out = args.root / args.out
    (out / "summary").mkdir(parents=True, exist_ok=True)
    for name, table in zip(TABLES, (events, participation, fish)):
        table.to_parquet(out / f"{name}.parquet", index=False)
    for name, table in summarize(events, participation, fish).items():
        table.to_csv(out / "summary" / f"{name}.csv", index=False)


if __name__ == "__main__":
    main()
