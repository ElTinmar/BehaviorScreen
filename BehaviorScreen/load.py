import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, NamedTuple, Optional
import re
from re import Pattern
from video_tools import OpenCV_VideoReader
from datetime import datetime
from BehaviorScreen.core import TIME_TOLERANCE_S

class BehaviorData(NamedTuple):
    metadata: Dict
    stimuli: List[Dict]
    tracking: pd.DataFrame
    full_tracking: pd.DataFrame
    video: OpenCV_VideoReader #NOTE not sure yet about this
    video_timestamps: pd.DataFrame
    temperature: pd.DataFrame
    
class BehaviorFiles(NamedTuple):
    metadata: Path
    stimuli: Path
    tracking: Path
    full_tracking: Path
    video: Path
    video_timestamps: Path
    temperature: Optional[Path]

class Directories:
    def __init__(
            self, 
            root: Path,
            metadata: str = '',
            stimuli: str = '',
            tracking: str = '',
            full_tracking: str = '',
            temperature: str = '',
            video: str = '',
            video_timestamp: str = '',
            results: str = '',
            plots: str = ''
        ) -> None:

        self.root: Path = Path(root)
        self.metadata: Path = self.root / metadata
        self.stimuli: Path = self.root / stimuli
        self.tracking: Path = self.root / tracking
        self.full_tracking: Path = self.root / full_tracking
        self.temperature: Path = self.root / temperature 
        self.video: Path = self.root / video
        self.video_timestamps: Path = self.root / video_timestamp
        self.results: Path = self.root / results
        self.plots: Path = self.root / plots

class FileNameInfo(NamedTuple):
    fish_id: int
    age: int
    line: str
    weekday: str
    day: int
    month: str
    year: int
    hour: int
    minute: int
    second: int
    extra: Optional[str]

    def to_datetime(self) -> datetime:
        return datetime.strptime(
            f"{self.day} {self.month} {self.year} "
            f"{self.hour}:{self.minute}:{self.second}",
            "%d %b %Y %H:%M:%S",
        )

    def matches(self, other: "FileNameInfo", time_tolerance_s: Optional[float] = None) -> bool:

        if not (
            self.fish_id == other.fish_id and
            self.age == other.age and
            self.line == other.line and
            self.day == other.day and
            self.month == other.month and
            self.year == other.year and
            self.extra == other.extra
        ):
            return False
        
        if time_tolerance_s is None:
            return True
        
        dt = (self.to_datetime() - other.to_datetime()).total_seconds()
        return abs(dt) <= time_tolerance_s
    
def filename_regexp(prefix: str, extension: str) -> Pattern:
    regexp = re.compile(
        f"^{prefix}"
        r"(?P<fish_id>\d{2})_"
        r"(?P<age>[0-9]+)dpf_"
        r"(?P<line>[^_]+)_"
        r"(?P<weekday>[A-Za-z]{3})_"
        r"(?P<day>\d{2})_"
        r"(?P<month>[A-Za-z]{3})_"
        r"(?P<year>\d{4})_"
        r"(?P<hour>\d{2})h"
        r"(?P<minute>\d{2})min"
        r"(?P<second>\d{2})sec"
        r"(?:_(?P<extra>[^.]+))?\."
        f"{extension}$"
    )
    return regexp


metadata_filename_regexp = filename_regexp('','metadata')
stimuli_filename_regexp = filename_regexp('stim_','json')
tracking_filename_regexp = filename_regexp('tracking_','csv')
full_tracking_filename_regexp = filename_regexp('','csv')
video_timestamps_filename_regexp = filename_regexp('','csv')
temperature_filename_regexp = filename_regexp('temperature_','csv')
video_filename_regexp = filename_regexp('','mp4')
video_timestamps_filename_regexp = filename_regexp('','csv')

def parse_filename(path: Path, regexp: Pattern) -> FileNameInfo:
    m = regexp.match(path.name)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {path.name}")
    g = m.groupdict()
    return FileNameInfo(
        fish_id = int(g["fish_id"]),
        age = int(g["age"]),
        line = g["line"],
        weekday = g["weekday"],
        day = int(g["day"]),
        month = g["month"],
        year = int(g["year"]),
        hour = int(g["hour"]),
        minute = int(g["minute"]),
        second = int(g["second"]),
        extra = g["extra"]
    )

def load_metadata(metadata_file: Path) -> Dict:
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    return metadata

def load_stimuli(stim_file: Path) -> List[Dict]:
    stimuli = []
    with open(stim_file) as f:
        for line in f:
            stimuli.append(json.loads(line))
    return stimuli

def load_tracking(tracking_file: Path) -> pd.DataFrame:
    return pd.read_csv(tracking_file)

def load_lightning_pose(full_tracking_file: Path) -> pd.DataFrame:
    df = pd.read_csv(full_tracking_file, header=[0,1,2])
    return df

def load_full_tracking(full_tracking_file: Optional[Path]) -> pd.DataFrame:
    # TODO normalize SLEAP/DLC/lightning pose
    # Dont want to assume one specific organisation of CSV
    if full_tracking_file is None:
        return pd.DataFrame()
    return load_lightning_pose(full_tracking_file)

def load_video(video_file: Path) -> OpenCV_VideoReader:
    reader = OpenCV_VideoReader()
    if video_file is None:
        return reader
    reader.open_file(str(video_file))
    return reader 

def load_video_timestamps(video_timestamp_file: Path) -> pd.DataFrame:
    return pd.read_csv(video_timestamp_file)

def load_temperature(temperature_file: Optional[Path]) -> pd.DataFrame:
    if temperature_file is None:
        return pd.DataFrame()
    return pd.read_csv(temperature_file)

def load_data(files: BehaviorFiles) -> BehaviorData:
    return BehaviorData(
        metadata = load_metadata(files.metadata),
        stimuli = load_stimuli(files.stimuli),
        tracking = load_tracking(files.tracking),
        full_tracking = load_full_tracking(files.full_tracking),
        video = load_video(files.video),
        video_timestamps = load_video_timestamps(files.video_timestamps),
        temperature = load_temperature(files.temperature)
    )

def find_file(
    file_info: FileNameInfo,
    dir: Path,
    regexp: Pattern,
    required: bool = True,
) -> Optional[Path]:

    for file in dir.iterdir():

        if not file.is_file():
            continue

        try:
            info = parse_filename(file, regexp)
        except ValueError:
            continue  

        if info.matches(file_info, time_tolerance_s=TIME_TOLERANCE_S):
            return file

    if required:
        raise FileNotFoundError(
            f"No matching file in {dir} for {file_info}"
        )

    return None

def find_files(dir: Directories) -> List[BehaviorFiles]:
    metadata_files = list(dir.metadata.glob("*.metadata"))
    experiments = []
    for metadata_file in metadata_files:
        file_info = parse_filename(metadata_file, metadata_filename_regexp)
        exp = BehaviorFiles(
            metadata = metadata_file,
            stimuli = find_file(file_info, dir.stimuli, stimuli_filename_regexp), # type: ignore
            tracking = find_file(file_info, dir.tracking, tracking_filename_regexp), # type: ignore
            full_tracking = find_file(file_info, dir.full_tracking, full_tracking_filename_regexp, required=False), # type: ignore
            video = find_file(file_info, dir.video, video_filename_regexp, required=False), # type: ignore
            video_timestamps = find_file(file_info, dir.video_timestamps, video_timestamps_filename_regexp, required=False), # type: ignore
            temperature = find_file(file_info, dir.temperature, temperature_filename_regexp, required=False)
        )
        experiments.append(exp)
    return experiments
