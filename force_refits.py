"""
One-off, outside the repo: after the bound changes of 0089, forget the saved fits (and their
replicates) of the rungs that sat on an old bound, and the fish bootstraps whose resamples did.
Run from the repo root, then resubmit spont_dark, spont_bright, after_looming and phototaxis.
"""
import csv
import json
import shutil
from pathlib import Path

FIGURES = Path("figures")
REFIT = {"spont_dark": ["06", "07", "08"], "spont_bright": ["06", "07", "08"], "after_looming": ["04", "06"]}
NEW_BOOTSTRAP = {"after_looming": "05", "phototaxis": "06"}

for experiment in sorted(set(REFIT) | set(NEW_BOOTSTRAP)):
    models = FIGURES / experiment / "models"
    with open(models / "rung_index.csv") as f:
        folders = {row["folder"][:2]: (row["folder"], row["model"]) for row in csv.DictReader(f)}
    records_path = models / "fit_records.json"
    records = json.loads(records_path.read_text())
    for rung in REFIT.get(experiment, []):
        folder, name = folders[rung]
        print(f"{experiment}/{folder}: refit" + ("" if records.pop(name, None) else " (no saved fit)"))
        for store in ("gof_replicates", "bootstrap_replicates"):
            shutil.rmtree(models / folder / store, ignore_errors=True)
    records_path.write_text(json.dumps(records, indent=2))
    if experiment in NEW_BOOTSTRAP:
        folder = folders[NEW_BOOTSTRAP[experiment]][0]
        shutil.rmtree(models / folder / "bootstrap_replicates", ignore_errors=True)
        print(f"{experiment}/{folder}: new fish bootstrap")
