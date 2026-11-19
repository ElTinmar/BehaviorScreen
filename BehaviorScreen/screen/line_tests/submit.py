"""
Test each line against the controls (Tier 1), one job per behavior; once they are done, merge every behavior saved
so far into the screen's tables and maps. On Slurm or here (BehaviorScreen/machines.yaml):
    python -m BehaviorScreen test-lines [behavior ...] [--model best|best-poisson|base] [--local]
    python -m BehaviorScreen test-lines --combine [--model ...] [--local]
    python -m BehaviorScreen test-lines --compare
A resubmitted behavior skips the cells it has already saved. --model runs the screen with another model of each
behavior (paths.VARIANTS) into its own folder, without the plate-simulated p-value (it simulates from the model);
--compare then sets the merged variants' hits side by side, here (compare_variants: tables only, seconds).
"""
import argparse

from BehaviorScreen.jobs import Jobs, add_local_argument
from BehaviorScreen.screen.control_models.submit import EXPERIMENTS
from BehaviorScreen.screen.line_tests import compare_variants
from BehaviorScreen.screen.paths import CONTROL_FITS, VARIANTS, line_tests_dir

RESOURCES = ["--cpus-per-task=24", "--mem=120000"]
COMBINE_RESOURCES = ["--cpus-per-task=4", "--mem=32000"]  # tables and plots: about 20 GB, minutes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("behaviors", nargs="*", metavar="behavior", help="of experiments.txt (default: all)")
    parser.add_argument("--combine", action="store_true", help="Merge the behaviors already saved, instead of running any.")
    parser.add_argument("--compare", action="store_true", help="Set the merged variants' hits side by side (compare_variants), here.")
    parser.add_argument("--model", choices=VARIANTS, default="best", help="Model of each behavior (default: the best).")
    add_local_argument(parser)
    args = parser.parse_args()
    if unknown := sorted(set(args.behaviors) - set(EXPERIMENTS)):
        parser.error(f"not in experiments.txt: {unknown}")
    if args.combine and args.behaviors:
        parser.error("--combine runs no behavior: submit them first, then --combine once they are done")
    if args.compare:
        compare_variants.main()
        return
    jobs = Jobs(args.local)
    run = ["python", "-m", "BehaviorScreen.screen.line_tests.run_screen", "--data-root", str(jobs.data_root),
           "--fits-root", str(CONTROL_FITS), "--model", args.model, "--output-dir", str(line_tests_dir(args.model))]
    if args.model != "best":
        run += ["--n-sim", "0"]
    prefix = "tier1" if args.model == "best" else "tier1_" + args.model.replace("-", "_")
    if args.combine:
        jobs.submit([*run, "--combine"], f"{prefix}_combine", COMBINE_RESOURCES, one_thread=True)
    else:
        for behavior in args.behaviors or EXPERIMENTS:
            jobs.submit([*run, "--behavior", behavior], f"{prefix}_{behavior}", RESOURCES, one_thread=True)
    jobs.finish()


if __name__ == "__main__":
    main()
