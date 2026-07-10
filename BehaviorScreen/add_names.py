from pathlib import Path
from BehaviorScreen.load import load_stimuli
from BehaviorScreen.protocol import protocol, protocol_ptx
import json
from typing import List, Dict

ROOT = Path("/media/martin/datastore_baier_group/_Projects/Martin_Privat/DATA/Behavioral_screen/DATA/Screen")
ROOT = Path("/media/martin/DATA_18TB/Screen")

def save_stimuli(stim_file: Path, stimuli: List[Dict]) -> None:
    with open(stim_file, 'w') as f:
        for stim in stimuli:
            f.write(json.dumps(stim) + '\n')

for file in ROOT.rglob("stim_*.json"):

    if not "results" in file.parts:
        continue

    stim = load_stimuli(file) # this is rounding float to 6th decimal
    num_stim = len(stim)

    labels = protocol_ptx if num_stim == len(protocol_ptx) else protocol
    for s,l in zip(stim, labels):
        s['name'] = l
    
    # backup_file = file.with_stem(file.stem + '_old')
    # file.rename(backup_file)

    save_stimuli(file, stim)


# revert
# for backup_file in ROOT.rglob("stim_*_old.json"):

#     if "results" not in backup_file.parts:
#         continue

#     original_file = backup_file.with_stem(backup_file.stem[:-4])
#     backup_file.replace(original_file)