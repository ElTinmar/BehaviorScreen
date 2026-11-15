"""
Tier 1 on the cluster: one Slurm job per behavior of experiments.txt (or of those named), then one that merges every
behavior saved so far into the screen's tables and maps, once those jobs have all succeeded:
    python -m BehaviorScreen.screen.line_tests.submit_tier1 [behavior ...]
    python -m BehaviorScreen.screen.line_tests.submit_tier1 --combine-only
A resubmitted behavior skips the cells it has already saved.
"""
import argparse

from BehaviorScreen.cluster import DATA_ROOT, submit
from BehaviorScreen.screen.control_models.submit_fits import EXPERIMENTS

RESOURCES = ["--cpus-per-task=24", "--mem=120000"]
RUN = ["python", "-m", "BehaviorScreen.screen.line_tests.run_screen", "--data-root", str(DATA_ROOT),
       "--fits-root", "./figures", "--output-dir", "./tier1_results"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("behaviors", nargs="*", choices=EXPERIMENTS, metavar="behavior", help="default: all")
    parser.add_argument("--combine-only", action="store_true", help="Only merge the behaviors already saved.")
    args = parser.parse_args()
    jobs = [] if args.combine_only else [
        submit([*RUN, "--behavior", behavior], f"tier1_{behavior}", RESOURCES, one_thread=True) for behavior in args.behaviors or EXPERIMENTS
    ]
    submit([*RUN, "--combine"], "tier1_combine", RESOURCES, after=jobs, one_thread=True)


if __name__ == "__main__":
    main()
