from .config import FOLDERS, CONFIG_YAML
from BehaviorScreen.plot import plot_heatmaps
from multiprocessing import Pool

N = 8

def process_folder(folder):
    print(f"processing {folder}")

    input_csv = folder / "bouts.csv"
    output_png = folder / "bouts.png"
    qc_csv = folder / "qc.csv"
    valid_trials_csv = folder / "valid_trials.csv"

    plot_heatmaps(
        quality_control=qc_csv,
        input_csv=input_csv,
        valid_trials_csv=valid_trials_csv,
        config_yaml = CONFIG_YAML,
        output_png = output_png,
    )

if __name__ == "__main__":
    with Pool(N) as pool:
        pool.map(process_folder, FOLDERS)