"""Command line: one experiment's diagnostics or fit, or consolidation across experiments."""
import argparse
import sys
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Callable, TextIO

from BehaviorScreen.point_process.analysis.model_config import EVENT_TABLES, get_model_config
from BehaviorScreen.point_process.analysis.tasks import (
    Replicates,
    code_version,
    run_consolidation_task,
    run_diagnostics_task,
    run_fit_task,
    run_plot_task,
)
from BehaviorScreen.point_process.point_process.dataset import BehavioralDataLoader


class Tee:
    """A stream that also appends what it writes to a log file."""

    def __init__(self, stream: TextIO, log: TextIO) -> None:
        self.stream, self.log = stream, log

    def write(self, text: str) -> int:
        self.log.write(text)
        return self.stream.write(text)

    def flush(self) -> None:
        self.log.flush()
        self.stream.flush()

    def __getattr__(self, name: str):
        return getattr(self.stream, name)


def make_loaders(args: argparse.Namespace) -> Callable[[str], BehavioralDataLoader]:
    """Loader of each event table (EVENT_TABLES key), read on first use."""

    @cache
    def loader(events: str) -> BehavioralDataLoader:
        loader = BehavioralDataLoader(
            events_csv=args.data_root / getattr(args, events),
            valid_trials_csvs=[args.data_root / p for p in args.valid_trials],
            qc_csvs=[args.data_root / p for p in args.qc],
            **EVENT_TABLES[events],
        )
        return loader.subset(conditions=args.conditions) if args.conditions else loader

    return loader


def add_data_arguments(parser: argparse.ArgumentParser, required: bool) -> None:
    """The data tables make_loaders reads."""
    parser.add_argument(
        "-d", "--data-root", type=Path, dest="data_root", required=required,
        help="Directory the file arguments below are relative to.",
    )
    parser.add_argument("--bouts", "--events", default="bouts_control.csv", help="Bout table.")
    parser.add_argument("--saccades", default="saccades_augmented_control.csv", help="Saccade table.")
    parser.add_argument(
        "--valid-trials", nargs="+", default=["valid_trials_control.csv"],
        help="Trial tables from quality control, one per screen folder.",
    )
    parser.add_argument("--qc", nargs="*", default=["qc_control.csv"], help="Fish-level exclusion tables from quality control.")
    parser.add_argument("--conditions", nargs="+", help="Conditions kept (e.g. vehicle danieau on the merged _all tables); default: every fish.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Point-process analysis of one screen")
    parser.add_argument("--exp", help="Experiment key (e.g. prey_capture_ipsi); needed by fit and diagnostics.")
    parser.add_argument(
        "--mode", choices=["diagnostics", "fit", "plot", "consolidate"], default="fit",
        help="plot: every rung's checks and plots from a finished fit run's parameters (in -o), without refitting it.",
    )
    add_data_arguments(parser, required=True)
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("./figures"), help="Output figures and CSVs.")
    parser.add_argument(
        "--refit", nargs="+", default=["missing"],
        help="fit mode: 'missing' (default) fits the rungs without a saved fit or whose saved NLL no longer reproduces; "
             "'all' every rung; rung numbers (e.g. 5 6) those and the missing ones.",
    )
    parser.add_argument("--gof-replicates", type=int, default=300, help="Simulate-and-refit replicates of each rung's goodness of fit.")
    parser.add_argument("--bootstrap-replicates", type=int, default=200, help="Fish-bootstrap replicates of the best model.")
    parser.add_argument("--new-replicates", action="store_true", help="Discard the saved replicates instead of reusing them.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Worker processes for fits and bootstraps (-1: all cores).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loader = make_loaders(args)
    model_config = get_model_config()

    if args.mode == "consolidate":
        run_consolidation_task(model_config, loader, args.output_dir)
        return
    if args.exp not in model_config:
        raise ValueError(f"--exp must be one of {list(model_config)} in {args.mode} mode, got {args.exp!r}")

    log = args.output_dir / args.exp / "run.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log, "a", encoding="utf-8")
    sys.stdout, sys.stderr = Tee(sys.stdout, log_file), Tee(sys.stderr, log_file)
    print(f"\n##### {datetime.now():%Y-%m-%d %H:%M} --mode {args.mode} --exp {args.exp} (code {code_version()}) #####")
    config = model_config[args.exp]
    replicates = Replicates(args.gof_replicates, args.bootstrap_replicates, args.new_replicates)
    dataset = loader(config["events"]).prepare_dataset(**config["dataset"])
    if args.mode == "diagnostics":
        run_diagnostics_task(args.exp, dataset, args.output_dir)
    elif args.mode == "plot":
        run_plot_task(args.exp, config, dataset, args.output_dir, n_jobs=args.n_jobs, replicates=replicates)
    else:
        refit = args.refit[0] if args.refit in (["missing"], ["all"]) else [int(r) for r in args.refit]
        run_fit_task(args.exp, config, dataset, args.output_dir, n_jobs=args.n_jobs, refit=refit, replicates=replicates)


if __name__ == "__main__":
    main()
