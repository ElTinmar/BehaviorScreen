from .config import ROOT, FOLDERS, CONFIG_YAML
from BehaviorScreen.plot import plot_heatmap

for f in FOLDERS:
    target = ROOT / f
    input_csv = target / "bouts.csv"
    output_png = target / "bouts.png"
    plot_heatmap(input_csv, CONFIG_YAML, output_png)
