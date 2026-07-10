from .config import ROOT, FOLDERS, CONFIG_YAML
from BehaviorScreen.plot import plot_heatmap
from multiprocessing import Pool, cpu_count

def process_folder(folder):
    print(f"processing {folder}")

    input_csv = folder / "bouts.csv"
    output_png = folder / "bouts.png"

    plot_heatmap(input_csv, CONFIG_YAML, output_png)
    
if __name__ == "__main__":
    with Pool(cpu_count()) as pool:
        pool.map(process_folder, FOLDERS)