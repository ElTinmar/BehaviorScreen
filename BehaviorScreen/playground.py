from BehaviorScreen.load import (
    Directories, 
    BehaviorData,
    BehaviorFiles,
    find_files, 
    load_data
)
from pathlib import Path
from typing import List

#ROOT = Path('/media/martin/DATA1/Behavioral_screen/DATA/WT/danieau')
ROOT = Path('/media/martin/MARTIN_8TB_0/Work/Baier/DATA/Behavioral_screen/DATA/WT/danieau')

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
files: List[BehaviorFiles] = find_files(directories)
behavior_file = files[0]
behavior_data: BehaviorData = load_data(behavior_file)