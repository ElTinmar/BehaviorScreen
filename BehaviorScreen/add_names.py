from pathlib import Path
from BehaviorScreen.load import load_stimuli
from BehaviorScreen.core import Stim
import json
from typing import List, Dict

ROOT = Path("/media/martin/datastore_baier_group/_Projects/Martin_Privat/DATA/Behavioral_screen/DATA/Screen")
ROOT = Path("/media/martin/DATA_18TB/Screen")

# TODO check that on the setup
labels_full = ["adaptation", "ramp 0"]
labels_full += 5 * ["prey capture right", "prey capture break 0", "prey capture left", "prey capture break 1"]
labels_full += ["ramp 1"]
labels_full += 10 * ["phototaxis bright right", "phototaxis break 0", "phototaxis bright left", "phototaxis break 1"]
labels_full += ["ramp 2"]
labels_full += 10 * ["spontaneous dark"]
labels_full += ["ramp 3"]
labels_full += 5 * ["flash dark", "flash ramp", "flash bright"]
labels_full += ["ramp 4"]
labels_full += 5 * ["grating left", "grating break 0", "grating right", "grating break 1", "grating forward", "grating break 2"]
labels_full += ["ramp 5"]
labels_full += 10 * ["spontaneous bright"]
labels_full += ["ramp 6"]
labels_full += 5 * ["pinwheel clockwise", "pinwheel break 0", "pinwheel counter-clockwise", "pinwheel break 1"]
labels_full += ["ramp 7"]
labels_full += 7 * ["looming right", "looming break 0", "looming left", "looming break 1"]

# phototaxis only
labels_ptx = ["adaptation"]
labels_ptx += 10 * ["phototaxis bright right", "phototaxis break 0", "phototaxis bright left", "phototaxis break 1"]

def save_stimuli(stim_file: Path, stimuli: List[Dict]) -> None:
    with open(stim_file, 'w') as f:
        for stim in stimuli:
            f.write(json.dumps(stim) + '\n')

for file in ROOT.rglob("stim_*.json"):

    if not "results" in file.parts:
        continue

    stim = load_stimuli(file) # this is rounding float to 6th decimal
    num_stim = len(stim)

    labels = labels_ptx if num_stim == 41 else labels_full
    for s,l in zip(stim, labels):
        if s['stim_select'] == Stim.PHOTOTAXIS:
            l += f" {s['foreground_color'][0]}"
        s['name'] = l
    
    backup_file = file.with_stem(file.stem + '_old')
    file.rename(backup_file)

    save_stimuli(file, stim)


# revert
# for backup_file in ROOT.rglob("stim_*_old.json"):

#     if "results" not in backup_file.parts:
#         continue

#     original_file = backup_file.with_stem(backup_file.stem[:-4])
#     backup_file.replace(original_file)