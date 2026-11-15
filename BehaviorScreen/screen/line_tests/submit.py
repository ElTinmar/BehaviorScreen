"""
Test each line against the controls (Tier 1), one job per behavior, then one merging their results.
On Slurm or here (BehaviorScreen/machines.yaml); the merge runs once the behaviors' jobs have all succeeded and covers
every behavior saved so far:
    python -m BehaviorScreen test-lines [behavior ...] [--local]
    python -m BehaviorScreen test-lines --combine-only
A resubmitted behavior skips the cells it has already saved.
"""
import argparse

from BehaviorScreen.jobs import Jobs, add_local_argument
from BehaviorScreen.screen.control_models.submit import EXPERIMENTS

RESOURCES = ["--cpus-per-task=24", "--mem=120000"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("behaviors", nargs="*", choices=EXPERIMENTS, metavar="behavior", help="default: all")
    parser.add_argument("--combine-only", action="store_true", help="Only merge the behaviors already saved.")
    add_local_argument(parser)
    args = parser.parse_args()
    jobs = Jobs(args.local)
    run = ["python", "-m", "BehaviorScreen.screen.line_tests.run_screen", "--data-root", str(jobs.data_root),
           "--fits-root", "./figures", "--output-dir", "./tier1_results"]
    behaviors = [] if args.combine_only else args.behaviors or EXPERIMENTS
    done = [jobs.submit([*run, "--behavior", b], f"tier1_{b}", RESOURCES, one_thread=True) for b in behaviors]
    jobs.submit([*run, "--combine"], "tier1_combine", RESOURCES, after=done, one_thread=True)
    jobs.finish()


if __name__ == "__main__":
    main()
