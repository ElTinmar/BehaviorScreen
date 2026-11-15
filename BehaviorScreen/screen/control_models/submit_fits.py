"""
Control fits on the cluster, one Slurm job per experiment of experiments.txt (or of those named), each with the resources its ladder needs:
    python -m BehaviorScreen.screen.control_models.submit_fits [experiment ...]
GPU nodes throughout (the GPUs stay idle: easier to get than CPU nodes), at most 18 cores per GPU: renewal rungs on long
trials or large grids get a full node, the other renewal ladders half of one, ladders without renewal rungs a quarter.
One fit worker per core; each holds the data and its renewal grid (2 GB or more for phototaxis).
"""
import argparse
from pathlib import Path

from BehaviorScreen.cluster import DATA_ROOT, submit

EXPERIMENTS = Path(__file__).with_name("experiments.txt").read_text().split()
HEAVY = {"after_looming", "phototaxis"}
LIGHT = {"prey_capture_convergent_saccade", "prey_capture_divergent_saccade", "okr_conjugate_saccades", "looming",
         "looming_divergent_saccade", "dark_flash"}
MEMORY = {4: "500000", 2: "160G", 1: "32G"}  # by GPUs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiments", nargs="*", choices=EXPERIMENTS, metavar="experiment", help="default: all")
    for experiment in parser.parse_args().experiments or EXPERIMENTS:
        gpus = 4 if experiment in HEAVY else 1 if experiment in LIGHT else 2
        submit(["python", "-m", "BehaviorScreen.screen.control_models.run_analysis", "--exp", experiment, "--mode", "fit",
                "--data-root", str(DATA_ROOT), "--output-dir", "./figures", "--n-jobs", str(18 * gpus)],
               experiment, ["--constraint=gpu", f"--gres=gpu:a100:{gpus}", f"--cpus-per-task={18 * gpus}", f"--mem={MEMORY[gpus]}"],
               one_thread=True)


if __name__ == "__main__":
    main()
