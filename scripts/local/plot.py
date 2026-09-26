from .config import FOLDERS, CONFIG_YAML
from BehaviorScreen.plot import run_plot
from multiprocessing import Pool

N = 6

def process_folder(folder):
    print(f"processing {folder}")

    input_csv = folder / "bouts.csv"
    output_png = folder / "bouts.png"
    qc_csv = folder / "qc.csv"

    run_plot(
        qc_csv=qc_csv,
        bouts_csv=input_csv, 
        bouts_png=output_png,
        config_yaml=CONFIG_YAML, 
        root = folder,
        metadata = "results",
        stimuli = "results",
        tracking = "results",
        lightning_pose = "lightning_pose",
        temperature = "results",
        video = "results",
        video_timestamp = "results",
        results = "results",
        plots = "results",
        interactive=False
    )
    
if __name__ == "__main__":
    with Pool(N) as pool:
        pool.map(process_folder, FOLDERS)