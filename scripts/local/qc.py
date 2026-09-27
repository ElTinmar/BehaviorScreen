from .config import FOLDERS
from BehaviorScreen.qc import quality_control
from multiprocessing import Pool

N = 6

def process_folder(folder):
    print(f"processing {folder}")

    epoch_presentation_csv = folder / "epoch_trial_presentation.csv"
    qc_csv = folder / "qc.csv"

    quality_control(
        root=folder,
        output_csv=qc_csv,
        epoch_presentation_csv=epoch_presentation_csv, 
        metadata = "results",
        stimuli = "results",
        tracking = "results",
        lightning_pose = "lightning_pose",
        temperature = "results",
        video = "results",
        video_timestamp = "results",
        results = "results",
        plots = "results",
    )
    
if __name__ == "__main__":
    with Pool(N) as pool:
        pool.map(process_folder, FOLDERS)