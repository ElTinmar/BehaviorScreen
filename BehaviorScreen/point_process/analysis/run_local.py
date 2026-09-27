import subprocess
from multiprocessing import Pool

EXPERIMENTS = [
    # "prey_capture_ipsi",
    # "prey_capture_contra",
    # "phototaxis_ipsi",
    # "phototaxis_contra",
    "omr_lateral_ipsi",
    "omr_lateral_contra",
    "omr_forward",
    "okr_ipsi",
    "okr_contra",
    # "looming_ipsi",
    # "looming_contra",
    # "dark_flash",
    # "spont_dark",
    # "spont_bright",
    # "after_looming",
]

DATA_DIR = "/media/martin/DATA_18TB/Screen"
OUT_DIR = "./figures"
N_WORKERS = 1


def run_experiment(exp_name):
    cmd = [
        "python", "-m", "BehaviorScreen.point_process.analysis.run_analysis",
        "--exp", exp_name,
        "--mode", "fit",
        "--data-root", DATA_DIR,
        "--output-dir", OUT_DIR,
    ]
    print(f"Starting {exp_name}")
    subprocess.run(cmd, check=True)
    print(f"Finished {exp_name}")


if __name__ == "__main__":
    with Pool(N_WORKERS) as pool:
        pool.map(run_experiment, EXPERIMENTS)