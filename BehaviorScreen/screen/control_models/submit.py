"""
Fit the control models, one job per experiment and mode, on Slurm or here (BehaviorScreen/machines.yaml):
    python -m BehaviorScreen fit-controls [experiment ...] [--modes diagnostics fit plot] [--local]
--modes plot redraws a finished fit's rung plots from its saved parameters and GOF replicates, without refitting.
On Slurm, GPU nodes throughout (the GPUs stay idle: easier to get than CPU nodes), at most 18 cores per GPU: renewal
rungs on long trials or large grids get a full node, the other renewal ladders half of one, ladders without renewal
rungs a quarter. One fit worker per core; each holds the data and its renewal grid (2 GB or more for phototaxis).
"""
import argparse
from pathlib import Path

from BehaviorScreen.jobs import Jobs, add_local_argument
from BehaviorScreen.screen.paths import CONTROL_FITS

EXPERIMENTS = Path(__file__).with_name("experiments.txt").read_text().split()
HEAVY = {"after_looming", "phototaxis"}
LIGHT = {"prey_capture_convergent_saccade", "prey_capture_divergent_saccade", "okr_conjugate_saccades", "looming",
         "looming_divergent_saccade", "dark_flash"}
MEMORY = {4: "500000", 2: "160G", 1: "32G"}  # by GPUs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiments", nargs="*", metavar="experiment", help="of experiments.txt (default: all)")
    parser.add_argument("--modes", nargs="+", choices=["diagnostics", "fit", "plot"], default=["fit"], help="run_analysis modes, in this order")
    add_local_argument(parser)
    args = parser.parse_args()
    if unknown := sorted(set(args.experiments) - set(EXPERIMENTS)):
        parser.error(f"not in experiments.txt: {unknown}")
    jobs = Jobs(args.local)
    for experiment in args.experiments or EXPERIMENTS:
        gpus = 4 if experiment in HEAVY else 1 if experiment in LIGHT else 2
        for mode in args.modes:
            jobs.submit(["python", "-m", "BehaviorScreen.screen.control_models.run_analysis", "--exp", experiment, "--mode", mode,
                         "--data-root", str(jobs.data_root), "--output-dir", str(CONTROL_FITS), "--n-jobs", str(jobs.workers(18 * gpus))],
                        f"{experiment}_{mode}",
                        ["--constraint=gpu", f"--gres=gpu:a100:{gpus}", f"--cpus-per-task={18 * gpus}", f"--mem={MEMORY[gpus]}"],
                        one_thread=True)
    jobs.finish()


if __name__ == "__main__":
    main()
