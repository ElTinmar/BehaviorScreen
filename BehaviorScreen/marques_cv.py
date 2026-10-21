"""
Between-fish variability of bout rates and screen readouts in the Marques et al. 2018 tracking data.

    python marques_cv.py download ./marques_data
    python marques_cv.py analyze ./marques_data ./marques_cv
    python marques_cv.py fig3a ./marques_data ./marques_fig3a
    python marques_cv.py inventory ./marques_data ./marques_inventory

`download` fetches the five Mendeley parts and Table S2 (mmc2.xlsx). Run it again to
resume interrupted downloads: only files missing or shorter than listed are fetched. If the API refuses, use "Download All" on each dataset page (links in PARTS).

`analyze` reads the .mat files straight from the zips in each "Experiments with
Continuous Tracking" folder and from "Experiments with Discontinuous Tracking.zip". Conditions are
labelled by experiment (zip name), dataset and protocol names (EnumeratorDataSet.m and
EnumeratorProtocol.m, if found under ROOT) and Table S2 rows matched by bout count.

`analyze` estimates, per fish and stimulus, the bout rate from runs of consecutive
bouts under the same stimulus number: (bouts - 1) / (last onset - first onset) per
run, pooled over runs. Interbout data are not in the files, so the time before the
first and after the last bout of a run is unknown; fish without two bouts in a run
are missing, which biases the CV down for low-rate conditions.

Per (experiment, dataset, protocol, stimulus) with enough fish, the CV of these rates
across fish is reported as observed: it holds both true differences between fish and
counting noise at the number of bouts each fish produced, with no model of the bout
process. It is valid for a protocol giving each fish a similar number of bouts.
Conditions with fewer than MIN_BOUTS_PER_RUN bouts per run or MIN_BOUTS bouts per fish
(rare-event stimuli) are reported but left out of the summary: their run-based rates
are unreliable.

The summary gives the CV distribution by bouts per fish, and the fish per group needed
to detect each FOLD change (default 1.5, 2, 3) of a rate at 80% power, two-sided, at
alpha = 0.05 / LINES (gamma-Poisson counts, as in the screen's frailty model: the per-fish
variance of a rate relative to its mean is CV^2 = c^2 + 1/mu):

    n = 2 (z_alpha + z_0.8)^2 CV^2 / ln(FOLD)^2

Stimulus numbers are mapped to conditions from the rig's protocol files (the authors'
analysis code, ProtocolListFreelyBehaving; STIMULUS_LABELS and the functions above it);
turn_bias.csv gives, per experiment and stimulus, the mean sign of the turn angle
(BoutKinematicParameters column 10, counterclockwise positive: left turns), as a check.

Readouts of the screen (READOUTS, named and pooled as in model_config) are measured per
fish with the same bout categories (BoutInf column 11) and windows, from the first bout
of each presentation (stimulus onsets are not in the files):
    strength   rate of the readout's bouts in the window, or for escapes the fraction
               of presentations with one; its CV across fish gives the fish needed
    direction  for sided readouts, fraction of turning events toward the side the
               protocol says is correct (Readout.side; frontal/rear stimuli left out);
               the SD of its log-odds across fish gives the fish needed to reduce the
               bias above chance FOLD-fold (2-fold: e.g. 93 % -> 72 % correct)
Each is also carried over to the screen's protocol (Readout.screen_trials, screen_window_s):
the counting noise at the Marques event numbers (Poisson 1/mu; binomial (1 - p)/(p k) for
escapes; 1/(turns q (1 - q)) for direction) is subtracted from CV^2 (log-odds variance),
floored at 0, and the counting noise at the screen's event numbers added back.
Only arenas with one fish are used, except for flash_escape (dimming was only recorded in groups of 7).

`fig3a` checks the stimulus mapping against Fig. 3A, using the paper's numbers
(firstOrderStatisticsOfFig3A.mat, in part 5): per stimulus number, the bout-type
distribution (mean over fish of each fish's fractions, with the paper's filters
approximated from the first bout of each presentation) is matched to the closest paper
column of its panel and compared with the column the mapping predicts.
"""
import argparse
import itertools
import io
import json
import re
import tempfile
import zipfile
import zlib
import urllib.error
import urllib.request
from pathlib import Path
from dataclasses import dataclass, field, replace
from typing import Callable, Iterator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
from scipy import stats

TABLE_S2_URL = "https://www.cell.com/cms/10.1016/j.cub.2017.12.002/attachment/d751b425-5672-4343-b3d3-f86449108bf9/mmc2.xlsx"
TABLE_S2 = "mmc2.xlsx"
PARTS = ["r9vn7x287r", "xf2zv4yhv8", "bzh35dd2td", "fgpbbv29rw", "6hsdh9gj89"]
API = "https://data.mendeley.com/public-api/datasets"
VERSION = 1
FPS = 700.0
MIN_BOUTS_PER_RUN = 5
MIN_BOUTS = 10
POWER = 0.8
CONTINUOUS = "Experiments with Continuous Tracking"
DISCONTINUOUS = "Experiments with Discontinuous Tracking"  # bouts captured around stimuli only
KEYS = ["experiment", "dataset", "protocol", "stimulus"]

# BoutInf / FishInf columns (0-based), from Tracking_Data_Information
STIM, FRAME, CATEGORY = 2, 3, 10
DATASET, PROTOCOL, AGE = 2, 3, 10
TURN_ANGLE = 9  # BoutKinematicParameters: angle turned during tail motion (deg, clockwise negative)
CATEGORIES = {
    "SCS": 1, "LCS": 2, "BS": 3, "O": 4, "JT": 5, "SLC": 6, "S1": 7, "RT": 8, "S2": 9, "LLC": 10, "AS": 11, "SAT": 12, "HAT": 13,
}
MIN_TURN_DEG = 10.0  # bouts turning less are left out of turn bias and direction

# Stimulus numbers, from the protocol files: the stimulus number is the protocol line (0-based);
# -1 is the recording before or between stimuli. Angles are egocentric, counterclockwise from
# straight ahead, so +90 deg is the fish's left (from the looming and phototaxis turn biases).
OMR_SPEEDS_MM_S = [0, 2.5, 5, 7.5, 10, 15, 20, 30, 40, 50]
SPONT_LUX = [12, 44, 104, 232, 447, 790, 1890, 4700]  # Table S2; 16 levels in the protocol, pooled in pairs
LOOM_SPEEDS_MM_S = [2.5, 5, 10, 15, 20, 25]
OMR_SCREEN_STIMULI = (7, 19)  # OMRTurns at 10 mm/s, 90 and 270 deg: the screen's gratings
PHOTOTAXIS_LEVELS = [0, 20, 41, 61, 82, 102]  # projector levels; each trial follows 1 s of grey (128)
PHOTOTAXIS_READOUT_LEVELS = (20, 41)  # one side dark, the other dimly lit: the only clear phototaxis here


def odd_index(stimulus: int) -> int:
    """0, 1, 2 ... for the odd stimulus numbers 1, 3, 5 ..."""
    return (stimulus - 1) // 2


def phototaxis_sides(stimulus: int) -> tuple[int, int]:
    """(left, right) levels of a 12 s phototaxis trial: 6 x 6 combinations, left from 102 down to 0, right-major."""
    j = odd_index(stimulus)
    return PHOTOTAXIS_LEVELS[5 - j % 6], PHOTOTAXIS_LEVELS[j // 6]


def loom_angle(experiment: str, stimulus: int) -> int:
    """Direction the spot comes from: 8 directions 45 deg apart, or for each speed front, left, back, right."""
    j = odd_index(stimulus)
    return 45 * j if experiment.startswith("LoomingDifferentDirections") else 90 * (j % 4)


def flash_levels(experiment: str, stimulus: int) -> tuple[int, int]:
    """(3 min background, 1 s level) of a multiFlash stimulus: 6 x 6 levels, background-major; ManyInt3 0-75, ManyInt 0-255."""
    j, step = odd_index(stimulus), 15 if experiment.endswith("3") else 51
    return step * (j // 6), step * (j % 6)


def beep_column(experiment: str, stimulus: int) -> int:
    """Tone frequency index (200-2000 Hz): from the name for one-frequency experiments, else one per block of 5."""
    match = re.search(r"(\d+)Hz", experiment)
    return int(match[1]) // 200 - 1 if match else (stimulus - 1) // 5


def dim_column(experiment: str, stimulus: int) -> int:
    """Fig. 3A dimming column of a step from a lit background to dark (ManyInt3 0-4, ManyInt 5-9), else -1."""
    background, level = flash_levels(experiment, stimulus)
    if stimulus % 2 == 0 or level > 0 or background == 0:
        return -1
    many_int3 = experiment.endswith("3")
    return background // (15 if many_int3 else 51) - 1 + (0 if many_int3 else 5)


def side(angle: float) -> int:
    """+1 for a stimulus on the fish's left, -1 on its right, 0 ahead or behind."""
    return int(np.sign(np.round(np.sin(np.radians(angle)), 9)))


def tone_label(stimulus: int) -> str:
    return f"tone {200 * (beep_column('', stimulus) + 1)} Hz" if stimulus % 5 == 4 else "before tone"


def spont_intensity(stimulus: int) -> int | None:
    """DarkLightTransitions: odd stimuli are 3-min light epochs, two per intensity in increasing order (index into SPONT_LUX)."""
    return (stimulus - 1) // 4 if stimulus > 0 and stimulus % 2 == 1 else None

STIMULUS_LABELS: dict[str, Callable[[int], str]] = {
    "OMRSpeed": lambda s: f"forward grating {OMR_SPEEDS_MM_S[s]} mm/s",
    "OMRTurns": lambda s: "static grating / pre-stimulus (?)" if s == 0 else f"grating {(s - 1) * 15} deg",
    "LoomingDifferentDirections": lambda s: f"loom from {loom_angle('LoomingDifferentDirections', s)} deg" if s % 2 else "interval",
    "LoomingDifferentSpeeds": lambda s: (f"loom {LOOM_SPEEDS_MM_S[odd_index(s) // 4]} mm/s from "
                                         f"{loom_angle('LoomingDifferentSpeeds', s)} deg") if s % 2 else "interval",
    "Phototaxis": lambda s: "left {} / right {}".format(*phototaxis_sides(s)) if s % 2 else "1 s grey",
    "3minLightDark": lambda s: ("light", "dark")[s],
    "DarkLightTransitions": lambda s: f"light {SPONT_LUX[spont_intensity(s)]} lux" if s % 2 else "dark",
    "AcousticLightDark": lambda s: ("tone, " if s % 3 == 1 else "") + ("light", "dark")[s // 3 % 2],
    "Beeps200to2000": tone_label,
    "FlashBeeps200to2000": lambda s: "flash" if s % 5 == 2 else tone_label(s),
    "multiBeep": lambda s: "tone" if s % 5 == 4 else "before tone",
    "multiFlash3min1sManyInt3": lambda s: "background {}, 1 s at {}".format(*flash_levels("3", s)) if s % 2 else "background",
    "multiFlash3min1sManyInt": lambda s: "background {}, 1 s at {}".format(*flash_levels("", s)) if s % 2 else "background",
    "SpontaneousMultiFishDark": lambda s: "dark",
    "SpontaneousMultiFishLight": lambda s: "light",
    "SpontaneousWEyeTracking": lambda s: "spontaneous",
    "PreyCapture": lambda s: "prey session",
}


def stimulus_label(experiment: str, stimulus: int) -> str:
    if stimulus == -1:
        return "before/between stimuli"
    label = next((f for prefix, f in STIMULUS_LABELS.items() if experiment.startswith(prefix)), None)
    return label(stimulus) if label else ""


def phototaxis_readout(stimulus: int) -> bool:
    """Phototaxis trials with one side dark and the other at one of PHOTOTAXIS_READOUT_LEVELS."""
    lit = [[0, level] for level in PHOTOTAXIS_READOUT_LEVELS]
    return stimulus > 0 and stimulus % 2 == 1 and sorted(phototaxis_sides(stimulus)) in lit


def phototaxis_side(experiment: str, stimulus: int) -> int:
    left, right = phototaxis_sides(stimulus)
    return int(np.sign(left - right))


@dataclass(frozen=True)
class Readout:
    """A readout of the screen (model_config names), measured in the Marques experiments that have it."""

    name: str
    experiments: tuple[str, ...]  # zip-name prefixes
    stimuli: Callable[[str, int], bool]
    categories: tuple[str, ...] | None  # None: all bouts
    window_s: float | None  # window end, from the first bout of each presentation; None: end of the run
    start_s: float = 0.0  # window start, from the first bout
    escape: bool = False  # strength = fraction of presentations with >= 1 event, instead of a rate
    side: Callable[[str, int], int] | None = None  # sided readouts: correct turn (+1 left, -1 right, 0 none)
    conditions: Callable[[str, int], object] | None = None  # measure the CV per condition and average over them
    screen_trials: int = 0  # trials per larva in the screen (protocol.py)
    screen_window_s: float = 0.0  # counting window per trial in the screen (model_config), for rates
    multi_fish: bool = False  # keep arenas with several fish (only these exist for this readout; identities may swap)


READOUTS = [
    Readout("omr_forward", ("OMRSpeed",), lambda e, s: s == 4, ("S1", "S2", "BS"), 9.0,
            screen_trials=5, screen_window_s=9.0),
    Readout("omr_lateral", ("OMRTurns",), lambda e, s: s in OMR_SCREEN_STIMULI, ("RT", "HAT"), 9.0,
            side=lambda e, s: side((s - 1) * 15), screen_trials=10, screen_window_s=9.0),
    Readout("phototaxis", ("Phototaxis",), lambda e, s: phototaxis_readout(s), ("RT",), 24.0, side=phototaxis_side,
            screen_trials=20, screen_window_s=24.0),
    Readout("phototaxis_early", ("Phototaxis",), lambda e, s: phototaxis_readout(s), ("RT",), 2.5, side=phototaxis_side,
            screen_trials=20, screen_window_s=2.5),
    Readout("phototaxis_late", ("Phototaxis",), lambda e, s: phototaxis_readout(s), ("RT",), None, 2.5, side=phototaxis_side,
            screen_trials=20, screen_window_s=21.5),
    Readout("looming", ("LoomingDifferentDirections",), lambda e, s: s > 0 and s % 2 == 1 and loom_angle(e, s) in (90, 270),
            ("SLC", "LLC", "O", "SAT"), 9.0, escape=True,  # Marques classify most loom escapes as SAT or O (Fig. 3A)
            side=lambda e, s: -side(loom_angle(e, s)), screen_trials=14),
    # The 16 light levels of DarkLightTransitions change neither the bout rate nor its CV, but a CV from
    # one 180 s epoch and ~26 fish carries ~0.06 of sampling noise, so the CV is averaged over the epochs.
    Readout("spont_dark", ("DarkLightTransitions",), lambda e, s: s >= 0 and s % 2 == 0, None, None,
            conditions=lambda e, s: s, screen_trials=10, screen_window_s=24.0),
    Readout("spont_bright", ("DarkLightTransitions",), lambda e, s: s > 0 and s % 2 == 1, None, None,
            conditions=lambda e, s: s, screen_trials=10, screen_window_s=24.0),
    Readout("prey_capture", ("PreyCapture",), lambda e, s: True, ("JT",), None, screen_trials=10, screen_window_s=24.0),
    Readout("flash_escape", ("multiFlash3min1sManyInt",), lambda e, s: not e.endswith("3") and dim_column(e, s) >= 0,
            ("O", "SAT"), None,
            escape=True, screen_trials=5, multi_fish=True),
]


BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


def open_url(url: str, referer: str = "https://data.mendeley.com/", start: int = 0):
    headers = {**BROWSER_HEADERS, "Referer": referer} | ({"Range": f"bytes={start}-"} if start else {})
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)


def save(url: str, target: Path, size: int | None = None, referer: str = "https://data.mendeley.com/") -> None:
    """
    Download to a .part file renamed on completion, so an interrupted download never
    looks complete. An existing .part file, or a target shorter than `size` (left by an
    earlier interrupted run), is resumed with an HTTP Range request; if the server
    ignores the range, the download restarts from zero.
    """
    partial = target.with_name(target.name + ".part")
    if size and target.exists() and target.stat().st_size < size:
        target.replace(partial)
    start = partial.stat().st_size if partial.exists() else 0
    if size and start > size:
        start = 0
    if size is None or start < size:
        with open_url(url, referer, start) as response:
            resumed = response.status == 206
            if start:
                print(f"    {'resuming at' if resumed else 'server cannot resume, restarting from'} "
                      f"{start / 1e6 if resumed else 0:.0f} MB")
            with partial.open("ab" if resumed else "wb") as f:
                while chunk := response.read(1 << 20):
                    f.write(chunk)
    partial.replace(target)


def intact(path: Path, size: int) -> bool:
    """
    Same size as listed by Mendeley. Zips are not CRC-checked: some large archives on
    Mendeley fail Python's check as served, and downloading them again cannot change that.
    """
    return path.exists() and path.stat().st_size == size


def get_json(url: str) -> list:
    with open_url(url) as response:
        return json.load(response)


def download_folder(dataset: str, folder_id: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for f in get_json(f"{API}/{dataset}/files?folder_id={folder_id}&version={VERSION}"):
        target = out / f["filename"]
        if intact(target, f["size"]):
            continue
        print(f"  {target} ({f['size'] / 1e6:.0f} MB)")
        save(f["content_details"]["download_url"], target, f["size"])
        if not intact(target, f["size"]):
            print(f"    {target}: {target.stat().st_size} bytes, {f['size']} expected")


def subfolders(dataset: str) -> dict[str, Path]:
    """{folder id: path} of a dataset's subfolders; empty if the API refuses to list them."""
    try:
        folders = get_json(f"{API}/{dataset}/folders/{VERSION}")
    except urllib.error.HTTPError as error:
        print(f"  folder listing: {error}; downloading the top folder only")
        return {}
    names = {f["id"]: f["name"] for f in folders}
    parents = {f["id"]: f.get("parent_id") for f in folders}

    def path(fid: str) -> Path:
        return path(parents[fid]) / names[fid] if parents.get(fid) in names else Path(names[fid])

    return {fid: path(fid) for fid in names}


def download(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    table = out / TABLE_S2
    if not table.exists():
        try:
            save(TABLE_S2_URL, table, referer="https://www.cell.com/current-biology/fulltext/S0960-9822(17)31631-6")
        except urllib.error.HTTPError as error:
            print(f"Table S2: {error}. Save it from a browser as {table}:\n  {TABLE_S2_URL}")
    for dataset in PARTS:
        print(dataset)
        try:
            download_folder(dataset, "root", out / dataset)
        except urllib.error.HTTPError as error:
            print(f"  {error}. Use 'Download All' on https://data.mendeley.com/datasets/{dataset}/{VERSION} "
                  f"and unzip into {out / dataset}")
            continue
        for fid, path in subfolders(dataset).items():
            download_folder(dataset, fid, out / dataset / path)


def read_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    """
    Read one member. Some zips over 4 GB store member offsets modulo 4 GiB (e.g. made
    with macOS Archive Utility); members past 4 GB are then found at offset + k * 4 GiB.
    """
    offset = info.header_offset
    for k in range(4):
        info.header_offset = offset + k * 2**32
        try:
            return archive.read(info)
        except zipfile.BadZipFile:
            continue
    info.header_offset = offset
    return archive.read(info)


def zip_mats(archive: zipfile.ZipFile, label: str) -> Iterator[tuple[str, bytes]]:
    """(member, bytes) of every .mat in a zip, opening nested zips; unreadable members are reported and skipped."""
    with archive:
        members = [i for i in archive.infolist() if i.filename.endswith((".mat", ".zip")) and "__MACOSX" not in i.filename]
        for info in sorted(members, key=lambda i: i.filename):
            name = f"{label}/{info.filename}"
            try:
                data = read_member(archive, info)
                if info.filename.endswith(".zip"):
                    yield from zip_mats(zipfile.ZipFile(io.BytesIO(data)), name)
                else:
                    yield name, data
            except (zipfile.BadZipFile, zlib.error) as error:
                print(f"  {name}: {error}, skipped (extract the zip with unzip or 7z next to itself)")


def archive_mats(path: Path, label: str) -> Iterator[tuple[str, bytes]]:
    """(member, bytes) of every .mat in a zip, or in the folder it was extracted to (same name, next to it)."""
    extracted = path.with_suffix("")
    if not extracted.is_dir():
        try:
            yield from zip_mats(zipfile.ZipFile(path), label)
        except zipfile.BadZipFile as error:
            print(f"  {path.name}: {error}, skipped (extract it with unzip or 7z to {extracted.name}/)")
        return
    for f in sorted(extracted.rglob("*")):
        name = f"{label}/{f.relative_to(extracted)}"
        if "__MACOSX" in f.parts:
            continue
        if f.suffix == ".mat":
            yield name, f.read_bytes()
        elif f.suffix == ".zip" and not f.with_suffix("").is_dir():
            yield from archive_mats(f, name)


def discontinuous_experiment(member: str) -> str:
    """The folder or nested zip a .mat sits in, below the discontinuous-tracking archive."""
    parts = Path(member).parts
    inner = parts[max(i for i, p in enumerate(parts) if p.removesuffix(".zip") == DISCONTINUOUS) + 1:]
    return Path(inner[0] if len(inner) > 1 else parts[-1]).stem


def mat_files(root: Path) -> Iterator[tuple[str, str, bytes]]:
    """(experiment, member, bytes) of every .mat: one experiment per continuous-tracking zip, and per folder of the discontinuous one."""
    for path in sorted(root.rglob("*.zip")):
        if path.parent.name == CONTINUOUS:
            yield from ((path.stem, m, d) for m, d in archive_mats(path, path.name))
        elif path.stem == DISCONTINUOUS:
            yield from ((discontinuous_experiment(m), m, d) for m, d in archive_mats(path, path.name))


def load_fish(data: bytes) -> list[dict]:
    """boutStructure as a list of {field: array}, for both pre-7.3 and 7.3 (HDF5) .mat files."""
    fields = ("BoutInf", "FishInf", "BoutKinematicParameters")
    try:
        content = scipy.io.loadmat(io.BytesIO(data), squeeze_me=True, struct_as_record=False)
        if "boutStructure" not in content:
            print(f"    no boutStructure, skipped (variables: {[k for k in content if not k.startswith('__')]})")
            return []
        return [{k: getattr(f, k) for k in fields} for f in np.atleast_1d(content["boutStructure"])]
    except NotImplementedError:
        import mat73
        with tempfile.NamedTemporaryFile(suffix=".mat") as tmp:
            tmp.write(data)
            tmp.flush()
            s = mat73.loadmat(tmp.name)["boutStructure"]
        return [dict(zip(fields, values)) for values in zip(*(s[k] for k in fields))]


def parse_enumerator(root: Path, name: str) -> dict[int, str]:
    """{number: name} from a MATLAB enumerator class (several names for one number are joined)."""
    path = next(root.rglob(f"{name}.m"), None)
    if path is None:
        return {}
    names: dict[int, list[str]] = {}
    for key, value in re.findall(r"^\s*(\w+)\s*=\s*(\d+)", path.read_text(errors="ignore"), flags=re.M):
        names.setdefault(int(value), []).append(key)
    return {number: "/".join(keys) for number, keys in names.items()}


def collect(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(bouts, fish): one row per bout (fish, stimulus, onset, category, turn angle), one row per fish."""
    bouts, fish_rows = [], []
    for experiment, member, data in mat_files(root):
        print(f"  {member}")
        try:
            fish = load_fish(data)
        except Exception as error:  # truncated or otherwise unreadable .mat: report, keep going
            print(f"    unreadable ({type(error).__name__}: {error}), skipped")
            continue
        for i, f in enumerate(fish):
            bout_inf = np.atleast_2d(f["BoutInf"])
            if bout_inf.size == 0:
                continue
            info, key = np.ravel(f["FishInf"]), len(fish_rows)
            kinematics = np.atleast_2d(f["BoutKinematicParameters"])
            fish_rows.append({
                "fish_key": key, "experiment": experiment, "file": member, "fish": i, "fish_in_file": len(fish),
                "dataset": int(info[DATASET]), "protocol": int(info[PROTOCOL]), "age": int(info[AGE]),
            })
            bouts.append(pd.DataFrame({
                "fish_key": key, "stimulus": bout_inf[:, STIM].astype(int), "t": bout_inf[:, FRAME] / FPS,
                "category": bout_inf[:, CATEGORY].astype(int), "angle": kinematics[:, TURN_ANGLE],
            }))
    fish = pd.DataFrame(fish_rows)
    datasets, protocols = parse_enumerator(root, "EnumeratorDataSet"), parse_enumerator(root, "EnumeratorProtocol")
    fish = fish.assign(dataset_name=fish["dataset"].map(datasets), protocol_name=fish["protocol"].map(protocols))
    bouts = pd.concat(bouts, ignore_index=True).sort_values(["fish_key", "t"], ignore_index=True)
    new_run = (bouts["fish_key"].diff() != 0) | (bouts["stimulus"].diff() != 0)
    bouts["run"] = new_run.cumsum()
    bouts["t0"] = bouts.groupby("run")["t"].transform("min")
    return bouts, fish


def fish_rates(bouts: pd.DataFrame, fish: pd.DataFrame) -> pd.DataFrame:
    """Bouts, observed time and rate per fish and stimulus, from runs of consecutive bouts under one stimulus."""
    runs = bouts.groupby("run").agg(
        fish_key=("fish_key", "first"), stimulus=("stimulus", "first"), n=("t", "size"),
        span=("t", lambda t: t.max() - t.min()),
    )
    counts = runs.groupby(["fish_key", "stimulus"])["n"].sum().rename("n_raw")
    usable = runs[runs["n"] >= 2].groupby(["fish_key", "stimulus"]).agg(
        bouts=("n", lambda n: (n - 1).sum()), runs=("n", "size"), T=("span", "sum")
    )
    usable = usable[usable["T"] > 0].assign(rate=lambda d: d["bouts"] / d["T"])
    rates = pd.concat([counts, usable], axis=1).reset_index()
    return rates.merge(fish, on="fish_key").assign(label=lambda d: [
        stimulus_label(e, s) for e, s in zip(d["experiment"], d["stimulus"])
    ])


def parse_table_s2(path: Path) -> pd.DataFrame:
    """(section, condition, n_bouts) for every stimulus row of Table S2; the prey capture block is skipped."""
    sheet = pd.read_excel(path, header=None)
    rows, section, bouts_col = [], None, None
    for i, row in sheet.iterrows():
        cells = row.dropna()
        if cells.empty:
            continue
        first = str(cells.iloc[0])
        if first.startswith("Stimulus type"):
            section = str(sheet.iloc[i - 1].dropna().iloc[0]).strip()
            bouts_col = next(c for c, v in row.items() if str(v).startswith("Number of bouts"))
        elif first.startswith("Behaviour"):
            bouts_col = None
        elif bouts_col is not None and pd.notna(row[bouts_col]) and str(row[bouts_col]).isdigit():
            params = [str(v) for c, v in row.items() if 0 < c < bouts_col and pd.notna(v)]
            rows.append({"section": section, "condition": " / ".join(params), "n_bouts": int(row[bouts_col])})
    return pd.DataFrame(rows)


def label_conditions(rates: pd.DataFrame, table: pd.DataFrame, tolerance: float = 0.02) -> pd.DataFrame:
    """Match each condition to the Table S2 rows whose bout count (all ages) is within `tolerance`."""
    totals = rates.groupby(KEYS)["n_raw"].sum().reset_index()

    def match(n: int) -> str:
        close = table[(table["n_bouts"] - n).abs() <= tolerance * n]
        return "; ".join(f"{r.section}: {r.condition} ({r.n_bouts})" for r in close.itertuples())

    return totals.assign(table_s2=totals["n_raw"].map(match))


def cv_by_condition(rates: pd.DataFrame, min_fish: int) -> pd.DataFrame:
    def summarize(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "label": g["label"].iloc[0], "n_fish": len(g), "rate_hz": g["rate"].mean(), "T_s": g["T"].median(),
            "mu_bouts": g["bouts"].mean(), "cv_observed": g["rate"].std(ddof=1) / g["rate"].mean(),
            "bouts_per_run": (g["bouts"] + g["runs"]).sum() / g["runs"].sum(),
            "single_fish_arena": (g["fish_in_file"] == 1).all(),
            "dataset_name": g["dataset_name"].iloc[0], "protocol_name": g["protocol_name"].iloc[0],
        })

    rates = rates.dropna(subset=["rate"]).groupby(KEYS).filter(lambda g: len(g) >= min_fish)
    summary = rates.groupby(KEYS).apply(summarize).reset_index()
    return summary.assign(reliable=(summary["bouts_per_run"] >= MIN_BOUTS_PER_RUN) & (summary["mu_bouts"] >= MIN_BOUTS))


def turn_bias(bouts: pd.DataFrame, fish: pd.DataFrame) -> pd.DataFrame:
    """Per experiment and stimulus: mean sign of the turn angle over bouts turning more than MIN_TURN_DEG."""
    turns = bouts[bouts["angle"].abs() >= MIN_TURN_DEG].merge(fish[["fish_key", "experiment"]], on="fish_key")
    bias = turns.assign(sign=np.sign(turns["angle"])).groupby(["experiment", "stimulus"]).agg(
        turns=("sign", "size"), bias=("sign", "mean"), fish=("fish_key", "nunique")
    ).reset_index()
    return bias.assign(label=[stimulus_label(e, s) for e, s in zip(bias["experiment"], bias["stimulus"])])


def readout_per_fish(spec: Readout, bouts: pd.DataFrame, fish: pd.DataFrame) -> pd.DataFrame:
    """One row per fish: presentations, events, strength (rate or escape probability) and, if sided, direction."""
    keys = fish.loc[fish["experiment"].str.startswith(spec.experiments), ["fish_key", "experiment"]]
    b = bouts.merge(keys, on="fish_key")
    b = b[np.array([spec.stimuli(e, s) for e, s in zip(b["experiment"], b["stimulus"])], dtype=bool)]
    events = b if spec.categories is None else b[b["category"].isin([CATEGORIES[c] for c in spec.categories])]
    events = events[(events["t"] - events["t0"]).between(spec.start_s, np.inf if spec.window_s is None else spec.window_s)]
    runs = b.groupby("run").agg(fish_key=("fish_key", "first"), span=("t", lambda t: t.max() - t.min()))
    runs["observed"] = (runs["span"] - spec.start_s).clip(lower=0)
    per_fish = runs.groupby("fish_key").agg(presentations=("span", "size"), T=("observed", "sum"))
    per_fish["events"] = events.groupby("fish_key").size().reindex(per_fish.index, fill_value=0)
    if spec.escape:
        responded = events.groupby("fish_key")["run"].nunique().reindex(per_fish.index, fill_value=0)
        per_fish["strength"] = responded / per_fish["presentations"]
    elif spec.window_s is not None:
        per_fish["strength"] = per_fish["events"] / (per_fish["presentations"] * (spec.window_s - spec.start_s))
    else:
        per_fish["strength"] = per_fish["events"] / per_fish["T"].where(per_fish["T"] > 0)
    if spec.side is not None:
        events = events.assign(side=[spec.side(e, s) for e, s in zip(events["experiment"], events["stimulus"])])
        turning = events[(events["side"] != 0) & (events["angle"].abs() >= MIN_TURN_DEG)]
        toward = turning.assign(toward=np.sign(turning["angle"]) == turning["side"]).groupby("fish_key")["toward"]
        per_fish["sided_events"] = toward.size().reindex(per_fish.index, fill_value=0)
        per_fish["toward"] = toward.sum().reindex(per_fish.index, fill_value=0)
    return per_fish.reset_index().merge(fish, on="fish_key")


def readout_frames(spec: Readout, bouts: pd.DataFrame, fish: pd.DataFrame) -> list[pd.DataFrame]:
    """One per-fish table per condition when spec.conditions is set, else one over all the readout's stimuli."""
    if spec.conditions is None:
        return [readout_per_fish(spec, bouts, fish)]
    keys = fish.loc[fish["experiment"].str.startswith(spec.experiments), ["fish_key", "experiment"]]
    b = bouts.merge(keys, on="fish_key")
    values = sorted({spec.conditions(e, s) for e, s in zip(b["experiment"], b["stimulus"]) if spec.stimuli(e, s)})
    one = lambda v: replace(spec, conditions=None, stimuli=lambda e, s: spec.stimuli(e, s) and spec.conditions(e, s) == v)
    return [readout_per_fish(one(v), bouts, fish) for v in values]


def readouts(bouts: pd.DataFrame, fish: pd.DataFrame, min_fish: int, folds: tuple[float, ...], lines: int) -> pd.DataFrame:
    """
    Per readout of the screen and Marques experiment: spread of strength and direction across fish
    and the fish needed at alpha = 0.05 / lines for each fold change, as measured in Marques
    (fish_<f>x) and carried over to the screen's trials and windows (fish_screen_<f>x). For
    direction, the fold change applies to the bias above chance (direction_<f>x: the target).
    """
    alpha = 0.05 / lines
    rows = []
    for spec in READOUTS:
        frames = [f[spec.multi_fish | (f["fish_in_file"] == 1)].dropna(subset=["strength"])
                  for f in readout_frames(spec, bouts, fish)]
        for experiment in sorted({e for f in frames for e in f["experiment"].unique()}):
            parts = [g for g in (f[f["experiment"] == experiment] for f in frames) if len(g) >= min_fish]
            if not parts:
                continue
            cv = float(np.mean([g["strength"].std(ddof=1) / g["strength"].mean() for g in parts]))
            g = pd.concat(parts)
            p, k = g["strength"].mean(), g["presentations"].mean()
            if spec.escape:
                count, count_screen, scale = (1 - p) / (p * k), (1 - p) / (p * spec.screen_trials), spec.screen_trials / k
            else:
                mu, mu_screen = g["events"].mean(), p * spec.screen_window_s * spec.screen_trials
                count, count_screen, scale = 1 / mu, 1 / mu_screen, mu_screen / mu
            variance_screen = max(cv**2 - count, 0) + count_screen
            row = {
                "readout": spec.name, "experiment": experiment, "n_fish": g["fish_key"].nunique(),
                "conditions": len(parts), "presentations": k,
                "run_s": (g["T"] / g["presentations"]).median(), "events_per_fish": g["events"].mean(),
                "strength_mean": p, "cv_strength": cv,
                "screen_trials": spec.screen_trials, "cv_screen": np.sqrt(variance_screen),
                **{f"fish_{f:g}x": fish_needed(cv**2, np.log(f), alpha) for f in folds},
                **{f"fish_screen_{f:g}x": fish_needed(variance_screen, np.log(f), alpha) for f in folds},
            }
            d = g[g["sided_events"] > 0] if spec.side is not None else g.iloc[:0]
            if len(d) >= min_fish:
                logit = np.log((d["toward"] + 0.5) / (d["sided_events"] - d["toward"] + 0.5))
                q, turns = (d["toward"] / d["sided_events"]).mean(), d["sided_events"].mean()
                between = max(logit.var(ddof=1) - 1 / (turns * q * (1 - q)), 0)
                variance_screen = between + 1 / (turns * scale * q * (1 - q))
                row |= {
                    "direction_fish": len(d), "direction_mean": q,
                    "sd_logit_direction": logit.std(ddof=1), "sd_logit_screen": np.sqrt(variance_screen),
                    **{f"direction_{f:g}x": reduced_bias(q, f) for f in folds},
                    **{f"fish_direction_{f:g}x": fish_needed(logit.var(ddof=1), logit_drop(q, f), alpha) for f in folds},
                    **{f"fish_direction_screen_{f:g}x": fish_needed(variance_screen, logit_drop(q, f), alpha) for f in folds},
                }
            rows.append(row)
    return pd.DataFrame(rows)


def reduced_bias(correct: float, fold: float) -> float:
    """Fraction of correct turns once the bias above chance is `fold` times smaller."""
    return 0.5 + (correct - 0.5) / fold


def logit_drop(correct: float, fold: float) -> float:
    reduced = reduced_bias(correct, fold)
    return float(np.log(correct / (1 - correct)) - np.log(reduced / (1 - reduced)))


def fish_needed(variance: float, effect: float, alpha: float) -> float:
    """Fish per group to detect a difference `effect` in a per-fish quantity with this variance (NaN if undetectable)."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(POWER)
    return float(np.ceil(2 * z**2 * variance / effect**2)) if np.isfinite(variance) and effect > 0 else np.nan


def plot(summary: pd.DataFrame, out: Path) -> None:
    summary = summary[summary["reliable"]]
    fig, (ax_hist, ax_mu) = plt.subplots(1, 2, figsize=(10, 4))
    bins = np.linspace(0, summary["cv_observed"].max() * 1.05, 30)
    for single, label in [(True, "one fish per arena"), (False, "several fish (identity swaps)")]:
        sub = summary[summary["single_fish_arena"] == single]
        ax_hist.hist(sub["cv_observed"], bins=bins, alpha=0.6, label=f"{label} (n={len(sub)})")
        ax_mu.scatter(sub["mu_bouts"], sub["cv_observed"], s=12, alpha=0.7, label=label)
    ax_hist.set(xlabel="CV of bout rate across fish", ylabel="stimulus conditions")
    ax_mu.set(xscale="log", xlabel="mean bouts per fish", ylabel="CV across fish")
    ax_hist.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "cv_distribution.png", dpi=150)


def analyze(
    root: Path, out: Path, ages: tuple[int, ...], min_fish: int, table_s2: Path | None, folds: tuple[float, ...], lines: int
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    bouts, fish = collect(root)
    rates = fish_rates(bouts, fish)
    table_s2 = table_s2 or root / TABLE_S2
    labels = label_conditions(rates, parse_table_s2(table_s2)) if table_s2.exists() else None
    fish = fish[fish["age"].isin(ages)]
    bouts, rates = bouts[bouts["fish_key"].isin(fish["fish_key"])], rates[rates["age"].isin(ages)]
    rates.to_csv(out / "fish_rates.csv", index=False)

    summary = cv_by_condition(rates, min_fish)
    plot(summary, out)
    if labels is not None:
        summary = summary.merge(labels, on=KEYS, how="left")
        labels.to_csv(out / "table_s2_matches.csv", index=False)
    summary.to_csv(out / "cv_by_condition.csv", index=False)

    single = summary[summary["single_fish_arena"] & summary["reliable"]]
    print(f"\n{len(summary)} conditions, {len(single)} reliable with one fish per arena; ages {ages}, >= {min_fish} fish")
    by_bouts = single.groupby(pd.cut(single["mu_bouts"], [MIN_BOUTS, 50, 200, np.inf]), observed=True)["cv_observed"]
    table = by_bouts.describe(percentiles=[0.5, 0.9])[["count", "50%", "90%", "max"]]
    table.loc["all"] = single["cv_observed"].describe(percentiles=[0.5, 0.9])[["count", "50%", "90%", "max"]]
    print("\nCV of bout rate across fish, by mean bouts per fish:")
    print(table.round(2).to_string())
    print(f"\nFish per group at {POWER:.0%} power, alpha 0.05/{lines}:")
    for name, column in [("median", "50%"), ("90th percentile", "90%"), ("max", "max")]:
        cv = table.loc["all", column]
        print(f"  CV {cv:.2f} ({name}): " + ", ".join(
            f"{f:g}-fold -> {fish_needed(cv**2, np.log(f), 0.05 / lines):.0f}" for f in folds))

    bias = turn_bias(bouts, fish)
    bias.to_csv(out / "turn_bias.csv", index=False)
    directional = bias[bias["experiment"].isin(["OMRTurns", "LoomingDifferentDirections", "LoomingDifferentSpeeds", "Phototaxis"])]
    print("\nTurn bias per stimulus (mean sign of turns; check the stimulus mapping):")
    print(directional[["experiment", "stimulus", "label", "fish", "turns", "bias"]].round(2).to_string(index=False))

    result = readouts(bouts, fish, min_fish, folds, lines)
    result.to_csv(out / "readouts.csv", index=False)
    print(f"\nScreen readouts, fish per group at {POWER:.0%} power and alpha 0.05/{lines} for each fold change "
          f"(of the rate, escape probability or direction bias), in Marques and carried over to the screen's protocol:")
    print(result.round(2).to_string(index=False))



# --- Fig. 3A check of the stimulus mapping ------------------------------------------------
FIG3A_STATS = "firstOrderStatisticsOfFig3A.mat"
FIG3A_TYPES = ["AS", "S1", "S2", "SCS", "LCS", "BS", "JT", "HAT", "RT", "SAT", "O", "LLC", "SLC"]  # stats file order
LOOM_START_MM = {"directions": 40, "speeds": 40}  # as in the methods (protocol: 5 cm for directions); 2 s matches Fig. 3A better


@dataclass(frozen=True)
class Panel:
    """A Fig. 3A panel: its columns in the stats file and the Marques units (experiment, stimulus) behind them."""

    name: str
    experiments: tuple[str, ...]  # zip-name prefixes
    columns: range  # rows of firstOrderStatisticsSortedFig3A
    stimuli: Callable[[str, int], bool]
    predicted: Callable[[str, int], int | None]  # column (0-based within the panel) our mapping predicts
    window_s: Callable[[int], float | None] = lambda column: None  # per paper column: bouts this long after onset
    first_bouts: int | None = None  # or: only the first bouts of each presentation
    hypotheses: dict[str, Callable[[str, int], int]] = field(default_factory=dict)  # candidate mappings, if none is predicted


# Fig. 3A bins phototaxis by DIFFERENTIAL light intensity (its x axis, 100-1800 lux), not by the
# level of the lit side, so all 30 unequal trials contribute. The protocol's six levels give five
# distinct differences; Table S2 shows four columns, so which four are drawn is tested.
PHOTOTAXIS_DIFFERENCES = [20, 41, 61, 82, 102]


def phototaxis_difference(stimulus: int) -> int | None:
    """The trial's difference in projector level between the two sides, as one of PHOTOTAXIS_DIFFERENCES."""
    left, right = phototaxis_sides(stimulus)
    difference = abs(left - right)
    return min(PHOTOTAXIS_DIFFERENCES, key=lambda d: abs(d - difference)) if difference else None


PHOTOTAXIS_HYPOTHESES = {
    f"differences {shown} lux-equivalent": (lambda shown: lambda e, s: shown.index(d) if (d := phototaxis_difference(s)) in shown else -1)(shown)
    for shown in itertools.combinations(PHOTOTAXIS_DIFFERENCES, 4)
}


def fold_angle(stimulus: int) -> int:
    angle = (stimulus - 1) * 15
    return min(angle, 360 - angle) // 15


PANELS = [
    Panel("spontaneous (0-4700 lux)", ("DarkLightTransitions",), range(0, 9), lambda e, s: s >= 0,
          lambda e, s: 0 if s % 2 == 0 else spont_intensity(s) + 1),
    Panel("social (1000 lux, 0 lux)", ("SpontaneousMultiFish",), range(9, 11), lambda e, s: s >= 0,
          lambda e, s: 1 if "Dark" in e else 0),
    Panel("forward OMR (0-50 mm/s)", ("OMRSpeed",), range(11, 21), lambda e, s: 0 <= s <= 9, lambda e, s: s),
    Panel("directional OMR (0-180 deg)", ("OMRTurns",), range(21, 34), lambda e, s: 1 <= s <= 24,
          lambda e, s: fold_angle(s)),
    Panel("phototaxis (4 intensities)", ("Phototaxis",), range(34, 38), lambda e, s: s > 0 and s % 2 == 1,
          lambda e, s: None, first_bouts=3, hypotheses=PHOTOTAXIS_HYPOTHESES),
    Panel("expanding spot, angle (0-180 deg)", ("LoomingDifferentDirections",), range(38, 43), lambda e, s: s > 0 and s % 2 == 1,
          lambda e, s: min(loom_angle(e, s), 360 - loom_angle(e, s)) // 45, window_s=lambda c: LOOM_START_MM["directions"] / 20),
    Panel("expanding spot, speed (2.5-25 mm/s)", ("LoomingDifferentSpeeds",), range(43, 49), lambda e, s: s > 0 and s % 2 == 1,
          lambda e, s: odd_index(s) // 4, window_s=lambda c: LOOM_START_MM["speeds"] / LOOM_SPEEDS_MM_S[c]),
    Panel("acoustic (200-2000 Hz)", ("Beeps200to2000", "multiBeep"), range(49, 59),
          lambda e, s: "Flash" not in e and s % 5 == 4, beep_column, first_bouts=1),
    Panel("flash + acoustic (200-2000 Hz)", ("FlashBeeps200to2000", "multiBeepFlash"), range(59, 69),
          lambda e, s: s % 5 == 4, beep_column, first_bouts=1),
    Panel("dimming (10 conditions)", ("multiFlash3min1sManyInt",), range(69, 79),
          lambda e, s: dim_column(e, s) >= 0, dim_column, first_bouts=1),
]


def type_fractions(bouts: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Per `keys`: mean over fish of each fish's fraction of bouts per type (FIG3A_TYPES order), and the fish count."""
    codes = [CATEGORIES[t] for t in FIG3A_TYPES]
    counts = pd.crosstab([bouts[k] for k in [*keys, "fish_key"]], bouts["category"]).reindex(columns=codes, fill_value=0)
    per_fish = counts.div(counts.sum(axis=1), axis=0)
    levels = list(range(len(keys)))
    mean = per_fish.groupby(level=levels).mean()
    mean.columns = FIG3A_TYPES
    return mean.assign(n_fish=per_fish.groupby(level=levels).size())


def panel_bouts(panel: Panel, bouts: pd.DataFrame, fish: pd.DataFrame, column: int) -> pd.DataFrame:
    """The panel's bouts, filtered as the paper did for this column (first bouts, or bouts until the spot arrives)."""
    keys = fish.loc[fish["experiment"].str.startswith(panel.experiments), ["fish_key", "experiment"]]
    b = bouts.merge(keys, on="fish_key")
    b = b[np.array([panel.stimuli(e, s) for e, s in zip(b["experiment"], b["stimulus"])], dtype=bool)]
    if panel.first_bouts is not None:
        b = b[b.groupby("run").cumcount() < panel.first_bouts]
    window = panel.window_s(column)
    return b if window is None else b[b["t"] - b["t0"] <= window]


def tv_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * float(np.abs(a - b).sum())


def pooled(filtered: list[pd.DataFrame], mapping: Callable[[str, int], int]) -> pd.DataFrame:
    """Bouts pooled per paper column under `mapping`, each column's bouts filtered as for that column."""
    parts = []
    for c, b in enumerate(filtered):
        column = np.array([mapping(e, s) for e, s in zip(b["experiment"], b["stimulus"])])
        parts.append(b[column == c].assign(column=c))
    return type_fractions(pd.concat(parts), ["column"]).reindex(range(len(filtered)))


def pooled_distances(fractions: pd.DataFrame, paper: np.ndarray, panel: Panel) -> list[float]:
    shown = fractions[FIG3A_TYPES].fillna(0).to_numpy()
    return [tv_distance(shown[c], paper[col]) for c, col in enumerate(panel.columns)]


def fig3a_check(bouts: pd.DataFrame, fish: pd.DataFrame, stats_path: Path, out: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    (units, columns, hypotheses). units: per panel and Marques stimulus, the closest paper column (total
    variation distance) against the column the mapping predicts. columns: where the mapping
    predicts every stimulus, the stimuli pooled per predicted column, as the paper pooled
    them (repeats, both directions), and their distance to that column; where it does not, the
    candidate mapping (Panel.hypotheses) whose pooled columns are closest to the paper's.
    hypotheses: the mean pooled distance of every mapping tested. The figure shows the pooled
    columns, or else each stimulus at its best column.
    """
    stats = scipy.io.loadmat(stats_path)["firstOrderStatisticsSortedFig3A"]
    paper, n_fish = stats[:, :, 0], stats[:, 0, 5]
    units, columns, hypotheses = [], [], []
    fig, axes = plt.subplots(2, len(PANELS), figsize=(3.2 * len(PANELS), 6), squeeze=False)
    for i, panel in enumerate(PANELS):
        n = len(panel.columns)
        same_window = len({panel.window_s(c) for c in range(n)}) == 1
        filtered = [panel_bouts(panel, bouts, fish, 0)] * n if same_window else [panel_bouts(panel, bouts, fish, c) for c in range(n)]
        if filtered[0].empty:
            axes[0, i].axis("off")
            axes[1, i].axis("off")
            continue
        by_column = [type_fractions(b, ["experiment", "stimulus"]) for b in filtered]
        index = by_column[0].index
        aligned = [f.reindex(index)[FIG3A_TYPES].fillna(0).to_numpy() for f in by_column]
        best_rows = []
        for k, unit in enumerate(index):
            distance = [tv_distance(aligned[c][k], paper[col]) for c, col in enumerate(panel.columns)]
            best, predicted = int(np.argmin(distance)), panel.predicted(*unit)
            best_rows.append(aligned[best][k])
            units.append({
                "panel": panel.name, "experiment": unit[0], "stimulus": unit[1], "label": stimulus_label(*unit),
                "n_fish": int(by_column[0].loc[unit, "n_fish"]), "best_column": best, "best_distance": distance[best],
                "predicted_column": predicted,
                "predicted_distance": distance[predicted] if predicted is not None else np.nan,
                "match": (best == predicted) if predicted is not None else None,
            })
        mappings = {"mapping": panel.predicted} if None not in [panel.predicted(*u) for u in index] else panel.hypotheses
        if mappings:
            scored = {name: pooled(filtered, m) for name, m in mappings.items()}
            mean = {name: float(np.mean(pooled_distances(f, paper, panel))) for name, f in scored.items()}
            chosen = min(mean, key=mean.get)
            hypotheses += [{"panel": panel.name, "hypothesis": name, "mean_distance": d, "chosen": name == chosen}
                           for name, d in mean.items()]
            fractions = scored[chosen]
            shown = fractions[FIG3A_TYPES].fillna(0).to_numpy()
            columns += [{"panel": panel.name, "mapping": chosen, "column": c, "n_fish": fractions.loc[c, "n_fish"],
                         "paper_n_fish": int(n_fish[col]), "distance": d}
                        for c, (col, d) in enumerate(zip(panel.columns, pooled_distances(fractions, paper, panel)))]
            label = f"pooled: {chosen}"
        else:
            order = np.argsort([u["best_column"] for u in units[-len(index):]], kind="stable")
            shown = np.array(best_rows)[order]
            label = "each stimulus, at its best column"
        axes[0, i].imshow(paper[panel.columns].T, aspect="auto", cmap="hot", vmin=0, vmax=0.8)
        axes[1, i].imshow(shown.T, aspect="auto", cmap="hot", vmin=0, vmax=0.8)
        axes[0, i].set_title(panel.name, fontsize=8)
        axes[1, i].set_xlabel(label, fontsize=7)
        for ax in axes[:, i]:
            ax.set_yticks(range(13), FIG3A_TYPES if i == 0 else [""] * 13, fontsize=6)
            ax.set_xticks([])
    axes[0, 0].set_ylabel("paper (Fig. 3A)")
    axes[1, 0].set_ylabel("Marques data")
    fig.tight_layout()
    fig.savefig(out / "fig3a_check.png", dpi=150)
    return pd.DataFrame(units), pd.DataFrame(columns), pd.DataFrame(hypotheses)


def fig3a(root: Path, out: Path, ages: tuple[int, ...], stats: Path | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    stats = stats or next(root.rglob(FIG3A_STATS))
    bouts, fish = collect(root)
    fish = fish[fish["age"].isin(ages)]
    bouts = bouts[bouts["fish_key"].isin(fish["fish_key"])]
    result, columns, hypotheses = fig3a_check(bouts, fish, stats, out)
    result.to_csv(out / "fig3a_check.csv", index=False)
    columns.to_csv(out / "fig3a_pooled.csv", index=False)
    hypotheses.to_csv(out / "fig3a_hypotheses.csv", index=False)
    print("\nMappings tested, best 5 per panel (mean distance of the pooled columns to the paper's):")
    top = hypotheses.sort_values("mean_distance").groupby("panel", sort=False).head(5) if len(hypotheses) else hypotheses
    print(top.round(3).to_string(index=False))
    print("\nStimuli pooled per paper column under the chosen mapping (distance 0 = identical, 1 = disjoint):")
    print(columns.round(3).to_string(index=False))
    for panel, g in result.groupby("panel", sort=False):
        checked = g.dropna(subset=["predicted_column"])
        verdict = f"{int(checked['match'].sum())}/{len(checked)} stimuli at their predicted column" if len(checked) else "no prediction"
        print(f"\n{panel}: {verdict}")
        print(g[["experiment", "stimulus", "label", "n_fish", "best_column", "best_distance",
                 "predicted_column", "predicted_distance"]].round(2).to_string(index=False))



def inventory(root: Path, out: Path) -> None:
    """Fish and bouts per experiment, protocol and stimulus number: what is in the data, to map new stimuli."""
    out.mkdir(parents=True, exist_ok=True)
    bouts, fish = collect(root)
    per_fish = bouts.groupby(["fish_key", "stimulus"]).size().rename("bouts").reset_index().merge(fish, on="fish_key")
    table = per_fish.groupby(["experiment", "dataset_name", "protocol_name", "stimulus"], dropna=False).agg(
        n_fish=("fish_key", "nunique"), bouts=("bouts", "sum"), fish_per_file=("fish_in_file", "median"),
        ages=("age", lambda a: " ".join(map(str, sorted(set(a))))),
    ).reset_index()
    table.to_csv(out / "inventory.csv", index=False)
    print(table.groupby(["experiment", "dataset_name", "protocol_name"], dropna=False).agg(
        stimuli=("stimulus", "nunique"), max_fish=("n_fish", "max"), bouts=("bouts", "sum"), fish_per_file=("fish_per_file", "max"),
    ).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download").add_argument("out", type=Path)
    a = sub.add_parser("analyze")
    a.add_argument("root", type=Path)
    a.add_argument("out", type=Path)
    a.add_argument("--ages", type=int, nargs="+", default=[6, 7])
    a.add_argument("--min-fish", type=int, default=8)
    a.add_argument("--table-s2", type=Path, help="Table S2 (default: ROOT/mmc2.xlsx), to label conditions by bout count")
    a.add_argument("--fold", type=float, nargs="+", default=[1.5, 2.0, 3.0], help="fold changes to detect (default 1.5 2 3)")
    a.add_argument("--lines", type=int, default=60, help="tests to correct alpha for (default 60 lines)")
    f = sub.add_parser("fig3a", help="check the stimulus mapping against Fig. 3A of the paper")
    f.add_argument("root", type=Path)
    f.add_argument("out", type=Path)
    f.add_argument("--ages", type=int, nargs="+", default=[6, 7])
    f.add_argument("--stats", type=Path, help=f"{FIG3A_STATS} (default: found under ROOT)")
    i = sub.add_parser("inventory", help="list experiments, protocols and stimulus numbers in the data")
    i.add_argument("root", type=Path)
    i.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.command == "inventory":
        inventory(args.root, args.out)
    elif args.command == "download":
        download(args.out)
    elif args.command == "fig3a":
        fig3a(args.root, args.out, tuple(args.ages), args.stats)
    else:
        analyze(args.root, args.out, tuple(args.ages), args.min_fish, args.table_s2, tuple(args.fold), args.lines)
