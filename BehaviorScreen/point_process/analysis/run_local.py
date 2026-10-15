import os
import subprocess

EXPERIMENTS = [
    # Directional stimuli: both sides in one sided model (Tier 1 columns)
    "prey_capture",
    "phototaxis",
    "omr_lateral",
    "okr",
    "okr_conjugate_saccades",
    "looming",  # gained the sided candidates
    # Non-directional
    "prey_capture_convergent_saccade",
    "prey_capture_divergent_saccade",
    "omr_forward",
    "looming_divergent_saccade",
    "after_looming",
    "dark_flash",
    "spont_dark",
    "spont_bright",
    # One side at a time: no longer used by Tier 1; uncomment to refit
    # "prey_capture_ipsi",
    # "prey_capture_contra",
    # "phototaxis_ipsi",
    # "phototaxis_contra",
    # "omr_lateral_ipsi",
    # "omr_lateral_contra",
    # "okr_ipsi",
    # "okr_contra",
    # "okr_conjugate_ipsi_saccades",
    # "okr_conjugate_contra_saccades",
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
#MODES = ["diagnostics", "fit"]
MODES = ["diagnostics"]

def run_experiment(exp_name: str) -> int:
    for mode in MODES:
        cmd = [
            "python", "-m", "BehaviorScreen.point_process.analysis.run_analysis",
            "--exp", exp_name,
            "--mode", mode,
            "--data-root", DATA_DIR,
            "--events", EVENTS,
            "--valid-trials", VALID_TRIALS,
            "--qc", QC,
            "--output-dir", OUT_DIR,
            "--n-jobs", str(PROCESSES),
        ]
        print(f"Starting {mode}")
        code = subprocess.run(cmd, env=ENV).returncode
        print(f"{'Finished' if code == 0 else f'FAILED ({code})'} {exp_name}: {mode}")
    return code


if __name__ == "__main__":
    failed = [exp for exp in EXPERIMENTS if run_experiment(exp) != 0]
    print(f"Failed: {failed}" if failed else "All experiments finished.")