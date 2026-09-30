from .config import FOLDERS, CONFIG_YAML, SACCADE_MODEL
from BehaviorScreen.eyes.run_analysis import run_analysis
from multiprocessing import Pool

N = 12

def process_folder(folder):
    print(f"processing {folder}")

    run_analysis(
        root=folder,
        config_yaml=CONFIG_YAML,
        model_path=SACCADE_MODEL,
    )

if __name__ == "__main__":
    with Pool(N) as pool:
        pool.map(process_folder, FOLDERS)