from .config import FOLDERS
from BehaviorScreen.megabouts import run_megabouts
from multiprocessing import Pool

N = 6

def process_folder(folder):

    print(f"processing {folder}")

    run_megabouts(
        root = folder,
        output_csv = "bouts.csv",
        metadata = "results",
        stimuli = "results",
        tracking = "results",
        lightning_pose = "lightning_pose",
        temperature = "results",
        video = "results",
        video_timestamp = "results",
        results = "results",
        plots = "results",
        cpu = False,
    )
    
if __name__ == "__main__":
    with Pool(N) as pool:
        pool.map(process_folder, FOLDERS)