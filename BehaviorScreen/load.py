import json
import re
import time
from datetime import datetime
from pathlib import Path
from re import Match, Pattern
from typing import Any, NamedTuple, Optional

import numpy as np
import pandas as pd
from video_tools import OpenCV_VideoReader

from BehaviorScreen.core import TIME_TOLERANCE_S

MONTH_MAP = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "okt": 10,
    "nov": 11,
    "dec": 12,
    "dez": 12,
}


class BehaviorFiles(NamedTuple):
    metadata: Path
    stimuli: Path
    tracking: Path
    full_tracking: Optional[Path]
    eyes_tracking: Optional[Path]
    video: Optional[Path]
    video_timestamps: Optional[Path]
    temperature: Optional[Path]


class Directories:
    def __init__(
        self,
        root: Path,
        metadata: str = "data",
        stimuli: str = "data",
        tracking: str = "data",
        full_tracking: str = "lightning_pose",
        eyes_tracking: str = "lightning_pose",
        temperature: str = "data",
        video: str = "video",
        video_timestamp: str = "video",
        results: str = "results",
        plots: str = "plots",
    ) -> None:

        self.root: Path = Path(root)
        self.metadata: Path = self.root / metadata
        self.stimuli: Path = self.root / stimuli
        self.tracking: Path = self.root / tracking
        self.full_tracking: Path = self.root / full_tracking
        self.eyes_tracking: Path = self.root / eyes_tracking
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
        return datetime(
            year=self.year,
            month=MONTH_MAP[self.month.strip().lower()],
            day=self.day,
            hour=self.hour,
            minute=self.minute,
            second=self.second,
        )

    def matches(
        self, other: "FileNameInfo", time_tolerance_s: Optional[float] = None
    ) -> bool:

        if not (
            self.fish_id == other.fish_id
            and self.age == other.age
            and self.line == other.line
            and self.day == other.day
            and self.month == other.month
            and self.year == other.year
            and self.extra == other.extra
        ):
            return False

        if time_tolerance_s is None:
            return True

        dt = (self.to_datetime() - other.to_datetime()).total_seconds()
        return abs(dt) <= time_tolerance_s


base_regexp = (
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
    r"(?:_(?P<extra>[^.]+))?"
)


def filename_regexp(prefix: str, extension: str | None = None) -> Pattern:
    pattern = f"^{re.escape(prefix)}{base_regexp}"
    if extension is not None:
        pattern += f"\\.{extension}$"
    return re.compile(pattern)


metadata_filename_regexp = filename_regexp("", "metadata")
stimuli_filename_regexp = filename_regexp("stim_", "json")
tracking_filename_regexp = filename_regexp("tracking_", "csv")
temperature_filename_regexp = filename_regexp("temperature_", "csv")
video_filename_regexp = filename_regexp("", "mp4")
lightningpose_full_filename_regexp = filename_regexp("", "csv")
lightningpose_eyes_filename_regexp = filename_regexp("eyes_", "csv")
video_timestamps_filename_regexp = filename_regexp("", "csv")


def _file_name_info(m: Match) -> FileNameInfo:
    g = m.groupdict()
    return FileNameInfo(
        fish_id=int(g["fish_id"]),
        age=int(g["age"]),
        line=g["line"],
        weekday=g["weekday"],
        day=int(g["day"]),
        month=g["month"],
        year=int(g["year"]),
        hour=int(g["hour"]),
        minute=int(g["minute"]),
        second=int(g["second"]),
        extra=g["extra"],
    )


def parse_filename(path: Path, regexp: Pattern) -> FileNameInfo:
    m = regexp.match(path.name)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {path.name}")
    return _file_name_info(m)


def parse_fish(fish: str) -> FileNameInfo:
    m = re.match(base_regexp, fish)
    if m is None:
        raise RuntimeError(f"failed to parse: {fish}")
    return _file_name_info(m)


def encode_time_of_day(info: FileNameInfo) -> tuple[float, float]:
    seconds = info.hour * 3600 + info.minute * 60 + info.second
    theta = 2 * np.pi * (seconds / (24 * 3600))
    return float(np.cos(theta)), float(np.sin(theta))


def fish_metadata(fish: str) -> dict[str, Any]:
    """Per-recording covariates, derived from the file name alone."""
    info = parse_fish(fish)
    cos_daytime, sin_daytime = encode_time_of_day(info)
    return {
        "file": fish,
        "dpf": info.age,
        "day": f"{info.day}.{info.month}.{info.year}",
        "cos_daytime": cos_daytime,
        "sin_daytime": sin_daytime,
    }


def load_metadata(metadata_file: Path) -> dict:
    with open(metadata_file, "r") as f:
        return json.load(f)


def load_stimuli(stim_file: Path) -> list[dict]:
    with open(stim_file) as f:
        return [
            json.loads(line, parse_float=lambda x: round(float(x), 6)) for line in f
        ]


def load_tracking(tracking_file: Path) -> pd.DataFrame:
    return pd.read_csv(tracking_file)


def load_lightning_pose(
    tracking_file: Optional[Path], n_frames: Optional[int] = None
) -> pd.DataFrame:
    """
    Pose rows, one per video frame. With `n_frames` (the number of timestamped
    frames), rows beyond it are dropped: the acquisition can keep writing video
    after the timestamp file stops, and rows align from the first frame.
    """
    if tracking_file is None:
        return pd.DataFrame()
    pose = pd.read_csv(tracking_file, header=[0, 1, 2])["heatmap_tracker"]
    if n_frames is not None and len(pose) > n_frames:
        print(
            f"[warn] {tracking_file.name}: dropping {len(pose) - n_frames} "
            "trailing rows without timestamp"
        )
        pose = pose.iloc[:n_frames]
    return pose  # type: ignore


def load_video(video_file: Optional[Path]) -> OpenCV_VideoReader:
    reader = OpenCV_VideoReader()
    if video_file is not None:
        reader.open_file(str(video_file))
    return reader


def load_video_timestamps(video_timestamp_file: Path) -> pd.DataFrame:
    """
    One row per frame written to the video, in video order. `index` is the
    camera frame counter and may skip values when frames were dropped, so a
    video frame is always the row position, never the `index` value.
    """
    df = pd.read_csv(video_timestamp_file)
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError(f"Timestamps are not increasing in {video_timestamp_file}")
    if not df["index"].is_unique:
        raise ValueError(f"Duplicate frame indices in {video_timestamp_file}")
    return df


def load_temperature(temperature_file: Optional[Path]) -> pd.DataFrame:
    if temperature_file is None:
        return pd.DataFrame()
    return pd.read_csv(temperature_file)


def find_file(
    file_info: FileNameInfo,
    dir: Path,
    regexp: Pattern,
    required: bool = True,
) -> Optional[Path]:

    if not dir.exists():
        if required:
            raise FileNotFoundError(f"Directory does not exist: {dir}")
        return None

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
        raise FileNotFoundError(f"No matching file in {dir} for {file_info}")

    return None


def find_files(dir: Directories) -> list[BehaviorFiles]:
    experiments = []
    for metadata_file in dir.metadata.glob("*.metadata"):
        info = parse_filename(metadata_file, metadata_filename_regexp)
        experiments.append(
            BehaviorFiles(
                metadata=metadata_file,
                stimuli=find_file(info, dir.stimuli, stimuli_filename_regexp),  # type: ignore
                tracking=find_file(info, dir.tracking, tracking_filename_regexp),  # type: ignore
                full_tracking=find_file(
                    info, dir.full_tracking, lightningpose_full_filename_regexp, False
                ),
                eyes_tracking=find_file(
                    info, dir.eyes_tracking, lightningpose_eyes_filename_regexp, False
                ),
                video=find_file(info, dir.video, video_filename_regexp, False),
                video_timestamps=find_file(
                    info, dir.video_timestamps, video_timestamps_filename_regexp, False
                ),
                temperature=find_file(
                    info, dir.temperature, temperature_filename_regexp, False
                ),
            )
        )
    return experiments


def file_younger_than(file_path: str | Path, hours: float) -> bool:
    path = Path(file_path)
    if not path.is_file():
        return False
    return time.time() - path.stat().st_mtime <= hours * 3600
