from BehaviorScreen.load import (
    Directories, 
    BehaviorData,
    BehaviorFiles,
    find_files, 
    load_data
)
from pathlib import Path
from typing import List

ROOT = Path('/media/martin/DATA1/Behavioral_screen/DATA/WT_dec_2025/chronus')

directories = Directories(
    root = ROOT,
    metadata='results',
    stimuli='results',
    tracking='results',
    full_tracking='lightning_pose',
    temperature='results',
    video='results',
    video_timestamp='results',
    results='results',
    plots=''
)
behavior_files: List[BehaviorFiles] = find_files(directories)
behavior_data: BehaviorData = load_data(behavior_files[0])