"""
What each preprocessing step can work on and what it already wrote, from the
files on disk. Per recording (raw for export, per animal after it): a column per
input and output, and a state:

    done     every output is there
    ready    every input is there, some output missing
    waiting  some input missing

plus the step's single files (tables, figures) with their modification times.
"""
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pandas as pd

from BehaviorScreen.load import (
    find_file, load_metadata, metadata_filename_regexp, parse_filename, stimuli_filename_regexp,
    temperature_filename_regexp, tracking_filename_regexp, video_filename_regexp, video_timestamps_filename_regexp,
)

SACCADES_CSV = "saccades_augmented.csv"  # written by the saccade step (eyes.run_analysis)
Check = Callable[[str], object]
# Raw files of a recording, exported per animal: option of their folder, file name pattern, exported suffix, --no-* flag
RAW = {
    "stimuli": ("stimuli", stimuli_filename_regexp, ".json", "no_stimuli"),
    "tracking": ("tracking", tracking_filename_regexp, ".csv", "no_tracking"),
    "video": ("video", video_filename_regexp, ".mp4", "no_videos"),
    "timestamps": ("video_timestamp", video_timestamps_filename_regexp, ".csv", "no_timestamps"),
}


@dataclass(frozen=True)
class Count:
    """Exported animals out of the recording's wells; true when complete."""
    done: int
    expected: int

    def __bool__(self) -> bool:
        return self.done == self.expected > 0

    def __str__(self) -> str:
        return f"{self.done}/{self.expected}"


@lru_cache(maxsize=None)
def wells(metadata: Path, modified: float) -> int:
    """Wells (ROIs) of a raw recording; cached per file version, the metadata can be large."""
    return len(load_metadata(metadata)["identity"]["ROIs"])


def export_checks(root: Path, v: Dict[str, str]) -> Tuple[List[str], Dict[str, Check], Dict[str, Check], Dict[str, Check]]:
    """Raw recordings: their raw files, and how many animals have each exported file."""
    raw = {p.stem: p for p in sorted((root / v["metadata"]).glob("*.metadata"))}
    info = {stem: parse_filename(p, metadata_filename_regexp) for stem, p in raw.items()}
    found = {name: {stem: find_file(info[stem], root / v[folder], regexp, False) for stem in raw}
             for name, (folder, regexp, _, _) in {**RAW, "temperature": ("temperature", temperature_filename_regexp, "", "")}.items()}
    results = root / v["results"]

    def exported(sources: Dict[str, Path], suffix: str) -> Check:
        def check(stem: str) -> Count:
            source = sources.get(stem)
            done = 0 if source is None else len(list(results.glob(f"{source.stem}_fish_*{suffix}")))
            return Count(done, wells(raw[stem], raw[stem].stat().st_mtime))
        return check

    inputs = {name: (lambda stem, f=found[name]: f[stem] is not None) for name, (*_, flag) in RAW.items() if not v[flag]}
    outputs = {"metadata": exported(raw, ".metadata")} if not v["no_metadata"] else {}
    outputs |= {name: exported(found[name], suffix) for name, (_, _, suffix, flag) in RAW.items() if not v[flag]}
    return list(raw), inputs, outputs, {"temperature": lambda stem: found["temperature"][stem] is not None}


def stems(folder: Path) -> List[str]:
    return sorted(p.stem for p in folder.glob("*.metadata"))


def listed(csv: Path, column: str) -> set:
    """Recordings in a table's column; reads that column only."""
    return set(pd.read_csv(csv, usecols=[column])[column].astype(str)) if csv.exists() else set()


def exists(pattern: str) -> Check:
    return lambda stem: Path(pattern.format(stem)).exists()


def member(csv: Path, column: str) -> Check:
    names = listed(csv, column)
    return lambda stem: stem in names


def count(folder: Path, pattern: str) -> Check:
    return lambda stem: len(list(folder.glob(pattern.format(stem))))


def checks(tab: str, root: Path, v: Dict[str, str]) -> Tuple[List[str], Dict[str, Check], Dict[str, Check], Dict[str, Check]]:
    """(recordings, inputs, outputs, other columns) of a tab; v: option values by dest (the overlay's for Overlay)."""
    if tab == "Overlay":  # optional layers as other columns: the overlay draws what exists
        metadata, pose = root / v["metadata"], root / v["lightning_pose"]
        other = {"eye pose": exists(f"{pose}/eyes_{{}}.csv"), "megabouts": exists(f"{metadata}/{{}}.pkl")}
        if v["saccades"]:
            other["in saccade table"] = member(root / v["saccades"], "fish")
        inputs = {"video": exists(f"{root / v['video']}/{{}}.mp4"), "pose": exists(f"{pose}/{{}}.csv")}
        return stems(metadata), inputs, {"videos": count(root / v["output_dir"], "{}_*s.mp4")}, other
    results, pose = root / v["results"], root / v["lightning_pose"]
    animals = stems(results)
    has_pose, has_eye_pose = exists(f"{pose}/{{}}.csv"), exists(f"{pose}/eyes_{{}}.csv")
    if tab == "Export":
        return export_checks(root, v)
    if tab == "Pose estimation":
        outputs = {"pose": has_pose, "eye video": exists(f"{root / v['eyes_video_dir']}/eyes_{{}}.mp4"), "eye pose": has_eye_pose}
        return animals, {"video": exists(f"{results}/{{}}.mp4")}, outputs, {}
    if tab == "Quality control":
        return animals, {"pose": has_pose}, {"valid trials": member(root / v["valid_trials_csv"], "file")}, {"excluded": member(root / v["qc_csv"], "file")}
    if tab == "Bouts":
        return animals, {"pose": has_pose}, {"megabouts": exists(f"{results}/{{}}.pkl"), "in bouts table": member(root / v["bouts_csv"], "file")}, {}
    if tab == "Saccades":
        return animals, {"eye pose": has_eye_pose}, {"in saccade table": member(root / SACCADES_CSV, "fish")}, {}
    return [], {}, {}, {}


def single_files(tab: str, root: Path, v: Dict[str, str]) -> Dict[str, List[Path]]:
    if tab == "Overlay":
        return {"input": [root / v["qc_csv"]] + ([root / v["saccades"]] if v["saccades"] else [])}
    plots = root / v["plots"]
    tables = [root / v["qc_csv"], root / v["valid_trials_csv"]]
    return {
        "Pose estimation": {"input": [Path(v["full_model_dir"]), Path(v["eyes_model_dir"])]},
        "Quality control": {"output": tables},
        "Bouts": {"output": [root / v["bouts_csv"]]},
        "Saccades": {"input": [Path(v["saccade_model"])], "output": [root / n for n in ("saccades.csv", "saccades_classified.csv", SACCADES_CSV)] + [plots / "saccade_clusters.png"]},
        "Heatmaps": {"input": [root / v["bouts_csv"], root / SACCADES_CSV, *tables, Path(v["yaml"])],
                     "output": [plots / v["bouts_png"], plots / v["saccades_png"]]},
    }.get(tab, {})


def tab_status(tab: str, root: Path, v: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    (one row per recording: its inputs "in: …", outputs "out: …", other columns and
    state; the tab's single files, inputs (models included) then outputs).
    """
    recordings, inputs, outputs, other = checks(tab, root, v)
    inputs, outputs = {f"in: {k}": c for k, c in inputs.items()}, {f"out: {k}": c for k, c in outputs.items()}
    columns = {**inputs, **outputs, **other}
    table = pd.DataFrame({name: [check(r) for r in recordings] for name, check in columns.items()}, index=pd.Index(recordings, name="recording"))
    ready = table[list(inputs)].astype(bool).all(axis=1)
    done = table[list(outputs)].astype(bool).all(axis=1)
    table["state"] = ["done" if d else "ready" if r else "waiting" for d, r in zip(done, ready)]
    files = pd.DataFrame(
        [{"role": "output" if role == "output" else "input", "file": str(p.relative_to(root)) if p.is_relative_to(root) else str(p), "present": p.exists(),
          "modified": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M") if p.exists() else ""}
         for role, paths in single_files(tab, root, v).items() for p in paths],
        columns=["role", "file", "present", "modified"],
    )
    return table, files
