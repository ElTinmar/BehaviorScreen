from .config import FOLDERS
from BehaviorScreen.export import export_single_animals
from multiprocessing import Pool

N = 4

def process_folder(folder):

    print(f"processing {folder}")

    export_single_animals(
        root = folder,
        metadata = "data",
        stimuli = "data",
        tracking = "data",
        temperature = "results",
        video = "video",
        video_timestamp = "video",
        results = "results",
        plots = "results",
        tracking_flag = False,
        timestamps_flag = False,
        stimuli_flag = False,
        metadata_flag = True,
        videos_flag = False,
        overwrite = True,
    )
    
if __name__ == "__main__":
    with Pool(N) as pool:
        pool.map(process_folder, FOLDERS)