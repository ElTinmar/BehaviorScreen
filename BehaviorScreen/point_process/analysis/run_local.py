import os
import subprocess

EXPERIMENTS = [
    # "prey_capture_ipsi",
    # "prey_capture_contra",
    "prey_capture_convergent_saccade",
    "phototaxis_ipsi",
    "phototaxis_contra",
    "omr_lateral_ipsi",
    "omr_lateral_contra",
    "omr_forward",
    "okr_ipsi",
    "okr_contra",
    "okr_conjugate_ipsi_saccades",
    "okr_conjugate_contra_saccades",
    "looming_ipsi",
    "looming_contra",
    "dark_flash",
    "spont_dark",
    "spont_bright",
    "after_looming",
]

DATA_DIR = "/media/martin/DATA_18TB/Screen"
EVENTS = "bouts_control.csv"
VALID_TRIALS = "valid_trials_control.csv"
QC = "qc_control.csv"
OUT_DIR = "./figures"
PROCESSES = 16  # parallel fits per experiment
ENV = os.environ | {
    "MPLBACKEND": "Agg",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


def run_experiment(exp_name: str) -> int:
    cmd = [
        "python", "-m", "BehaviorScreen.point_process.analysis.run_analysis",
        "--exp", exp_name,
        "--mode", "fit",
        "--data-root", DATA_DIR,
        "--events", EVENTS,
        "--valid-trials", VALID_TRIALS,
        "--qc", QC,
        "--output-dir", OUT_DIR,
        "--n-jobs", str(PROCESSES),
    ]
    print(f"Starting {exp_name}")
    code = subprocess.run(cmd, env=ENV).returncode
    print(f"{'Finished' if code == 0 else f'FAILED ({code})'} {exp_name}")
    return code


if __name__ == "__main__":
    failed = [exp for exp in EXPERIMENTS if run_experiment(exp) != 0]
    print(f"Failed: {failed}" if failed else "All experiments finished.")