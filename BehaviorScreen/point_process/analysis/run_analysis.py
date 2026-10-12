"""Command line: one experiment's diagnostics or fit, or consolidation across experiments."""
import argparse
from pathlib import Path

from BehaviorScreen.point_process.analysis.model_config import get_model_config
from BehaviorScreen.point_process.analysis.tasks import (
    run_consolidation_task,
    run_diagnostics_task,
    run_fit_task,
)
from BehaviorScreen.point_process.point_process.dataset import BehavioralDataLoader


def make_loader(args: argparse.Namespace) -> BehavioralDataLoader:
    return BehavioralDataLoader(
        events_csv=args.data_root / args.events,
        valid_trials_csvs=[args.data_root / p for p in args.valid_trials],
        qc_csvs=[args.data_root / p for p in args.qc],
        category_column=args.category_column,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Point-process analysis of one screen")
    parser.add_argument("--exp", help="Experiment key (e.g. prey_capture_ipsi); needed by fit and diagnostics.")
    parser.add_argument("--mode", choices=["diagnostics", "fit", "consolidate"], default="fit")
    parser.add_argument(
        "-d", "--data-root", type=Path, dest="data_root", required=True,
        help="Directory the file arguments below are relative to.",
    )
    parser.add_argument("--events", default="bouts_control.csv", help="Event table (bouts or saccades).")
    parser.add_argument(
        "--valid-trials", nargs="+", default=["valid_trials_control.csv"],
        help="Trial tables from quality control, one per screen folder.",
    )
    parser.add_argument("--qc", nargs="*", default=["qc_control.csv"], help="Fish-level exclusion tables from quality control.")
    parser.add_argument("--category-column", default="category", help="Event category column (saccades: saccade_category).")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("./figures"), help="Output figures and CSVs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loader = make_loader(args)
    dead_time_s = loader.dead_time()
    print(f"Dead time from all events: {dead_time_s:g} s")
    model_config = get_model_config(dead_time_s)

    if args.mode == "consolidate":
        run_consolidation_task(model_config, loader, args.output_dir)
        return
    if args.exp not in model_config:
        raise ValueError(f"--exp must be one of {list(model_config)} in {args.mode} mode, got {args.exp!r}")

    config = model_config[args.exp]
    dataset = loader.prepare_dataset(**config["dataset"])
    if args.mode == "diagnostics":
        run_diagnostics_task(args.exp, dataset, args.output_dir)
    else:
        run_fit_task(args.exp, config, dataset, args.output_dir)


if __name__ == "__main__":
    main()
