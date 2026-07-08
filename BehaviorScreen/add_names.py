from pathlib import Path
from BehaviorScreen.load import load_stimuli, stimuli_filename_regexp, parse_filename
from datetime import datetime
import pandas as pd

ROOT = Path("/media/martin/datastore_baier_group/_Projects/Martin_Privat/DATA/Behavioral_screen/DATA/Screen")
protocol_change_data = datetime(2026, 1, 28)

# old protocol
labels_old = ["adaptation", "ramp 0"]
labels_old += 5 * ["prey capture left", "prey capture break 0", "prey capture right", "prey capture break 1"]
labels_old += ["ramp 1"]
labels_old += 10 * ["phototaxis bright 0.2 right", "phototaxis break 0", "phototaxis bright 0.2 left", "phototaxis break 1"]
labels_old += ["ramp 2"]
labels_old += 10 * ["spontaneous dark"]
labels_old += ["ramp 3"]
labels_old += 5 * ["flash dark", "flash ramp", "flash bright"]
labels_old += ["ramp 4"]
labels_old += 5 * ["grating right", "grating break 0", "grating left", "grating break 1", "grating forward", "grating break 2"]
labels_old += ["ramp 5"]
labels_old += 10 * ["spontaneous bright"]
labels_old += ["ramp 6"]
labels_old += 5 * ["pinwheel counter-clockwise", "pinwheel break 0", "pinwheel clockwise", "pinwheel break 1"]
labels_old += ["ramp 7"]
labels_old += 7 * ["looming left", "looming break 0", "looming right", "looming break 1"]

# new protocol
labels_new = ["adaptation", "ramp 0"]
labels_new += 5 * ["prey capture left", "prey capture break 0", "prey capture right", "prey capture break 1"]
labels_new += ["ramp 1"]
labels_new += 10 * ["phototaxis bright 0.1 right", "phototaxis break 0", "phototaxis bright 0.1 left", "phototaxis break 1"]
labels_new += ["ramp 2"]
labels_new += 10 * ["spontaneous dark"]
labels_new += ["ramp 3"]
labels_new += 5 * ["flash dark", "flash ramp", "flash bright"]
labels_new += ["ramp 4"]
labels_new += 5 * ["grating right", "grating break 0", "grating left", "grating break 1", "grating forward", "grating break 2"]
labels_new += ["ramp 5"]
labels_new += 10 * ["spontaneous bright"]
labels_new += ["ramp 6"]
labels_new += 5 * ["pinwheel counter-clockwise", "pinwheel break 0", "pinwheel clockwise", "pinwheel break 1"]
labels_new += ["ramp 7"]
labels_new += 7 * ["looming left", "looming break 0", "looming right", "looming break 1"]

# phototaxis only
labels_ptx = ["adaptation"]
labels_ptx += 10 * ["phototaxis bright 0.1 right", "phototaxis break 0", "phototaxis bright 0.1 left", "phototaxis break 1"]

def old_protocol(stim, file_info) -> bool:
    before_change = file_info.to_datetime() <= protocol_change_data 
    num_stim = len(stim)
    return (num_stim == 182) and before_change

def new_protocol(stim, file_info) -> bool:
    after_change = file_info.to_datetime() > protocol_change_data 
    num_stim = len(stim)
    return (num_stim == 182) and after_change

def new_phototaxis(stim, file_info) -> bool:
    after_change = file_info.to_datetime() > protocol_change_data
    num_stim = len(stim)
    # TODO check intensity levels
    return (num_stim == 41) and after_change

stim_items = []
for file in ROOT.rglob("stim_*.json"):
    if "results" in file.parts:
        stim = load_stimuli(file)
        file_info = parse_filename(file, stimuli_filename_regexp)

        if old_protocol(stim, file_info):
            break
            
        if new_protocol(stim, file_info):
            break

        if new_phototaxis(stim, file_info):
            break

        # stim_items.append((
        #     old_protocol(stim, file_info),
        #     new_protocol(stim, file_info),
        #     new_phototaxis(stim, file_info)
        # ))

#pd.DataFrame(stim_items, columns=("old", "new", "ptx"))