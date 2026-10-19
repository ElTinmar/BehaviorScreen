"""
Between-fish variability of bout rates and screen readouts in the Marques et al. 2018 tracking data.

    python marques_cv.py download ./marques_data
    python marques_cv.py analyze ./marques_data ./marques_cv
    python marques_cv.py fig3a ./marques_data ./marques_fig3a

`download` fetches the five Mendeley parts and Table S2 (mmc2.xlsx). Run it again to
resume interrupted downloads: only files missing or shorter than listed are fetched. If the API refuses, use "Download All" on each dataset page (links in PARTS).

`analyze` reads the .mat files straight from the zips in each "Experiments with
Continuous Tracking" folder (discontinuous tracking is skipped). Conditions are
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
to detect a FOLD change of a rate at 80% power, two-sided, at alpha = 0.05 and
alpha = 0.05 / LINES (log-normal approximation, variance ln(1 + CV^2) per fish):

    n = 2 (z_alpha + z_0.8)^2 ln(1 + CV^2) / ln(FOLD)^2

Stimulus numbers are labelled from a cross-reference with Table S1/S2 and the methods
(STIMULUS_LABELS); turn_bias.csv gives, per experiment and stimulus, the mean sign of
the turn angle (BoutKinematicParameters column 10), to check the directional mapping.

Readouts of the screen (READOUTS, named and pooled as in model_config) are measured per
fish with the same bout categories (BoutInf column 11) and windows, from the first bout
of each presentation (stimulus onsets are not in the files):
    strength   rate of the readout's bouts in the window, or for escapes the fraction
               of presentations with one; its CV across fish gives the fish needed
    direction  for sided readouts, fraction of turning events toward the side the
               population prefers for that stimulus (stimuli with |bias| < MIN_SIDE_BIAS
               are frontal/rear and left out); the SD of its log-odds across fish gives
               the fish needed for a FOLD change of the odds
Only arenas with one fish are used. dark_flash has no protocol in these data.

`fig3a` checks the stimulus mapping against Fig. 3A, using the paper's numbers
(firstOrderStatisticsOfFig3A.mat, in part 5): per stimulus number, the bout-type
distribution (mean over fish of each fish's fractions, with the paper's filters
approximated from the first bout of each presentation) is matched to the closest paper
column of its panel and compared with the column the mapping predicts.
"""
import argparse
import io
import json
import re
import tempfile
import zipfile
import zlib
import urllib.error
import urllib.request
from pathlib import Path
from dataclasses import dataclass
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
KEYS = ["experiment", "dataset", "protocol", "stimulus"]

# BoutInf / FishInf columns (0-based), from Tracking_Data_Information
STIM, FRAME, CATEGORY = 2, 3, 10
DATASET, PROTOCOL, AGE = 2, 3, 10
TURN_ANGLE = 9  # BoutKinematicParameters: angle turned during tail motion (deg, clockwise negative)
CATEGORIES = {
    "SCS": 1, "LCS": 2, "BS": 3, "O": 4, "JT": 5, "SLC": 6, "S1": 7, "RT": 8, "S2": 9, "LLC": 10, "AS": 11, "SAT": 12, "HAT": 13,
}
MIN_TURN_DEG = 10.0  # bouts turning less are left out of turn bias and direction
MIN_SIDE_BIAS = 0.2  # stimuli with a weaker population bias count as frontal/rear: no direction

# Stimulus numbers, cross-referenced with Table S1/S2 and the methods of Marques et al. 2018
# (see README in the docstring). -1 is the recording before or between stimuli.
OMR_SPEEDS_MM_S = [0, 2.5, 5, 7.5, 10, 15, 20, 30, 40, 50]
STIMULUS_LABELS: dict[str, Callable[[int], str]] = {
    "OMRSpeed": lambda s: f"forward grating {OMR_SPEEDS_MM_S[s]} mm/s",
    "OMRTurns": lambda s: "static grating / pre-stimulus (?)" if s == 0 else f"grating {(s - 1) * 15} deg",
    "LoomingDifferentDirections": lambda s: "loom (one of 8 angles)" if s % 2 else "interval",
    "LoomingDifferentSpeeds": lambda s: "loom (6 speeds x 4 directions)" if s % 2 else "interval",
    "Phototaxis": lambda s: "phototaxis" if s % 2 else "1 s pre-flash",
    "3minLightDark": lambda s: ("dark", "light")[s],
    "DarkLightTransitions": lambda s: "dark" if s == 0 else "light epoch (intensity unknown)",
    "AcousticLightDark": lambda s: "tone" if s % 3 == 1 else "background",
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


@dataclass(frozen=True)
class Readout:
    """A readout of the screen (model_config names), measured in the Marques experiments that have it."""

    name: str
    experiments: tuple[str, ...]  # zip-name prefixes
    stimuli: Callable[[str, int], bool]
    categories: tuple[str, ...] | None  # None: all bouts
    window_s: float | None  # from the first bout of each presentation; None: whole run, as a rate
    escape: bool = False  # strength = fraction of presentations with >= 1 event, instead of a rate
    sided: bool = False  # also: fraction of turning events toward the stimulus side's preferred direction


READOUTS = [
    Readout("omr_forward", ("OMRSpeed",), lambda e, s: 1 <= s <= 9, ("S1", "S2", "BS"), 9.0),
    Readout("omr_lateral", ("OMRTurns",), lambda e, s: 2 <= s <= 24 and s != 13, ("RT", "HAT"), 9.0, sided=True),
    Readout("phototaxis", ("Phototaxis",), lambda e, s: s % 2 == 1, ("RT",), 24.0, sided=True),
    Readout("looming", ("LoomingDifferent",), lambda e, s: s % 2 == 1, ("SLC", "LLC"), 9.0, escape=True, sided=True),
    Readout("spont_dark", ("3minLightDark", "DarkLightTransitions"), lambda e, s: s == 0, None, None),
    Readout("spont_bright", ("3minLightDark",), lambda e, s: s == 1, None, None),
    Readout("prey_capture", ("PreyCapture",), lambda e, s: True, ("JT",), None),
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


def mat_files(root: Path) -> Iterator[tuple[str, str, bytes]]:
    """
    (experiment, member, bytes) of every .mat in the continuous-tracking zips. A zip
    extracted next to itself (folder named after the zip) is read from the folder instead;
    members Python's zipfile cannot read are reported and skipped.
    """
    for archive_path in sorted(root.rglob("*.zip")):
        if archive_path.parent.name != CONTINUOUS:
            continue
        extracted = archive_path.with_suffix("")
        if extracted.is_dir():
            for mat in sorted(extracted.rglob("*.mat")):
                if "__MACOSX" not in mat.parts:
                    yield archive_path.stem, str(mat.relative_to(root)), mat.read_bytes()
            continue
        try:
            archive = zipfile.ZipFile(archive_path)
        except zipfile.BadZipFile as error:
            print(f"  {archive_path.name}: {error}, skipped (extract it with unzip or 7z to {extracted.name}/)")
            continue
        with archive:
            members = [i for i in archive.infolist() if i.filename.endswith(".mat") and "__MACOSX" not in i.filename]
            for info in sorted(members, key=lambda i: i.filename):
                try:
                    data = read_member(archive, info)
                except (zipfile.BadZipFile, zlib.error) as error:
                    print(f"  {archive_path.name}/{info.filename}: {error}, skipped (extract the zip to {extracted.name}/)")
                    continue
                yield archive_path.stem, f"{archive_path.name}/{info.filename}", data


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


def readout_per_fish(spec: Readout, bouts: pd.DataFrame, fish: pd.DataFrame, bias: pd.DataFrame) -> pd.DataFrame:
    """One row per fish: presentations, events, strength (rate or escape probability) and, if sided, direction."""
    keys = fish.loc[fish["experiment"].str.startswith(spec.experiments), ["fish_key", "experiment"]]
    b = bouts.merge(keys, on="fish_key")
    b = b[np.array([spec.stimuli(e, s) for e, s in zip(b["experiment"], b["stimulus"])], dtype=bool)]
    events = b if spec.categories is None else b[b["category"].isin([CATEGORIES[c] for c in spec.categories])]
    if spec.window_s is not None:
        events = events[events["t"] - events["t0"] <= spec.window_s]
    runs = b.groupby("run").agg(fish_key=("fish_key", "first"), span=("t", lambda t: t.max() - t.min()))
    per_fish = runs.groupby("fish_key").agg(presentations=("span", "size"), T=("span", "sum"))
    per_fish["events"] = events.groupby("fish_key").size().reindex(per_fish.index, fill_value=0)
    if spec.escape:
        responded = events.groupby("fish_key")["run"].nunique().reindex(per_fish.index, fill_value=0)
        per_fish["strength"] = responded / per_fish["presentations"]
    elif spec.window_s is not None:
        per_fish["strength"] = per_fish["events"] / (per_fish["presentations"] * spec.window_s)
    else:
        per_fish["strength"] = per_fish["events"] / per_fish["T"].where(per_fish["T"] > 0)
    if spec.sided:
        preferred = bias.set_index(["experiment", "stimulus"])["bias"]
        side = preferred.reindex(pd.MultiIndex.from_arrays([events["experiment"], events["stimulus"]])).to_numpy()
        events = events.assign(side=side)
        turning = events[(events["side"].abs() >= MIN_SIDE_BIAS) & (events["angle"].abs() >= MIN_TURN_DEG)]
        toward = turning.assign(toward=np.sign(turning["angle"]) == np.sign(turning["side"])).groupby("fish_key")["toward"]
        per_fish["sided_events"] = toward.size().reindex(per_fish.index, fill_value=0)
        per_fish["toward"] = toward.sum().reindex(per_fish.index, fill_value=0)
    return per_fish.reset_index().merge(fish, on="fish_key")


def readouts(bouts: pd.DataFrame, fish: pd.DataFrame, bias: pd.DataFrame, min_fish: int, fold: float, lines: int) -> pd.DataFrame:
    """Per readout of the screen and Marques experiment: CV of strength across fish, spread of direction, fish needed."""
    rows = []
    for spec in READOUTS:
        per_fish = readout_per_fish(spec, bouts, fish, bias)
        per_fish = per_fish[per_fish["fish_in_file"] == 1].dropna(subset=["strength"])
        for experiment, g in per_fish.groupby("experiment"):
            if len(g) < min_fish:
                continue
            cv = g["strength"].std(ddof=1) / g["strength"].mean()
            row = {
                "readout": spec.name, "experiment": experiment, "n_fish": len(g),
                "presentations": g["presentations"].median(), "events_per_fish": g["events"].mean(),
                "strength_mean": g["strength"].mean(), "cv_strength": cv,
                "fish_strength": fish_needed(np.log1p(cv**2), fold, 0.05),
                f"fish_strength_alpha/{lines}": fish_needed(np.log1p(cv**2), fold, 0.05 / lines),
            }
            d = g[g["sided_events"] > 0] if spec.sided else g.iloc[:0]
            if len(d) >= min_fish:
                logit = np.log((d["toward"] + 0.5) / (d["sided_events"] - d["toward"] + 0.5))
                row |= {
                    "direction_fish": len(d), "direction_mean": (d["toward"] / d["sided_events"]).mean(),
                    "sd_logit_direction": logit.std(ddof=1),
                    "fish_direction": fish_needed(logit.var(ddof=1), fold, 0.05),
                    f"fish_direction_alpha/{lines}": fish_needed(logit.var(ddof=1), fold, 0.05 / lines),
                }
            rows.append(row)
    return pd.DataFrame(rows)


def fish_needed(log_variance: float, fold: float, alpha: float) -> float:
    """Fish per group to detect a FOLD change of a log-normal quantity with this per-fish log variance (NaN if unknown)."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(POWER)
    return float(np.ceil(2 * z**2 * log_variance / np.log(fold) ** 2)) if np.isfinite(log_variance) else np.nan


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
    root: Path, out: Path, ages: tuple[int, ...], min_fish: int, table_s2: Path | None, fold: float, lines: int
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
    print(f"\nFish per group for a {fold:g}-fold change, {POWER:.0%} power:")
    for name, column in [("median", "50%"), ("90th percentile", "90%"), ("max", "max")]:
        cv = table.loc["all", column]
        print(f"  CV {cv:.2f} ({name}): alpha 0.05 -> {fish_needed(np.log1p(cv**2), fold, 0.05):.0f}, "
              f"alpha 0.05/{lines} -> {fish_needed(np.log1p(cv**2), fold, 0.05 / lines):.0f}")

    bias = turn_bias(bouts, fish)
    bias.to_csv(out / "turn_bias.csv", index=False)
    directional = bias[bias["experiment"].isin(["OMRTurns", "LoomingDifferentDirections", "LoomingDifferentSpeeds", "Phototaxis"])]
    print("\nTurn bias per stimulus (mean sign of turns; check the stimulus mapping):")
    print(directional[["experiment", "stimulus", "label", "fish", "turns", "bias"]].round(2).to_string(index=False))

    result = readouts(bouts, fish, bias, min_fish, fold, lines)
    result.to_csv(out / "readouts.csv", index=False)
    print(f"\nScreen readouts (one fish per arena), fish per group for a {fold:g}-fold change, {POWER:.0%} power:")
    print(result.round(2).to_string(index=False))
    print("dark_flash: no dark-flash protocol in the continuous data; not estimated.")



# --- Fig. 3A check of the stimulus mapping ------------------------------------------------
FIG3A_STATS = "firstOrderStatisticsOfFig3A.mat"
FIG3A_TYPES = ["AS", "S1", "S2", "SCS", "LCS", "BS", "JT", "HAT", "RT", "SAT", "O", "LLC", "SLC"]  # stats file order
LOOM_START_MM = 40  # the spot starts 4 cm from the fish; bouts count until it reaches the fish
LOOM_SPEEDS_MM_S = [2.5, 5, 10, 15, 20, 25]


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


def fold_angle(stimulus: int) -> int:
    angle = (stimulus - 1) * 15
    return min(angle, 360 - angle) // 15


PANELS = [
    Panel("spontaneous (0-4700 lux)", ("DarkLightTransitions",), range(0, 9), lambda e, s: s >= 0,
          lambda e, s: 0 if s == 0 else None),
    Panel("social (1000 lux, 0 lux)", ("SpontaneousMultiFish",), range(9, 11), lambda e, s: s >= 0,
          lambda e, s: 1 if "Dark" in e else 0),
    Panel("forward OMR (0-50 mm/s)", ("OMRSpeed",), range(11, 21), lambda e, s: 0 <= s <= 9, lambda e, s: s),
    Panel("directional OMR (0-180 deg)", ("OMRTurns",), range(21, 34), lambda e, s: 1 <= s <= 24,
          lambda e, s: fold_angle(s)),
    Panel("phototaxis (4 intensities)", ("Phototaxis",), range(34, 38), lambda e, s: s % 2 == 1,
          lambda e, s: None, first_bouts=3),
    Panel("expanding spot, angle (0-180 deg)", ("LoomingDifferentDirections",), range(38, 43), lambda e, s: s % 2 == 1,
          lambda e, s: None, window_s=lambda c: LOOM_START_MM / 20),
    Panel("expanding spot, speed (2.5-25 mm/s)", ("LoomingDifferentSpeeds",), range(43, 49), lambda e, s: s % 2 == 1,
          lambda e, s: None, window_s=lambda c: LOOM_START_MM / LOOM_SPEEDS_MM_S[c]),
]


def type_fractions(bouts: pd.DataFrame) -> pd.DataFrame:
    """Mean over fish of each fish's fraction of bouts per type, per (experiment, stimulus), in FIG3A_TYPES order."""
    codes = [CATEGORIES[t] for t in FIG3A_TYPES]
    counts = pd.crosstab([bouts["experiment"], bouts["stimulus"], bouts["fish_key"]], bouts["category"])
    counts = counts.reindex(columns=codes, fill_value=0)
    per_fish = counts.div(counts.sum(axis=1), axis=0)
    mean = per_fish.groupby(level=[0, 1]).mean()
    mean.columns = FIG3A_TYPES
    return mean.assign(n_fish=per_fish.groupby(level=[0, 1]).size())


def panel_fractions(panel: Panel, bouts: pd.DataFrame, fish: pd.DataFrame, column: int) -> pd.DataFrame:
    keys = fish.loc[fish["experiment"].str.startswith(panel.experiments), ["fish_key", "experiment"]]
    b = bouts.merge(keys, on="fish_key")
    b = b[np.array([panel.stimuli(e, s) for e, s in zip(b["experiment"], b["stimulus"])], dtype=bool)]
    if panel.first_bouts is not None:
        b = b[b.groupby("run").cumcount() < panel.first_bouts]
    window = panel.window_s(column)
    if window is not None:
        b = b[b["t"] - b["t0"] <= window]
    return type_fractions(b)


def fig3a_check(bouts: pd.DataFrame, fish: pd.DataFrame, stats_path: Path, out: Path) -> pd.DataFrame:
    """
    For each panel and Marques unit (experiment, stimulus): the paper column whose bout-type
    distribution is closest (total variation distance), against the column our mapping predicts.
    """
    paper = scipy.io.loadmat(stats_path)["firstOrderStatisticsSortedFig3A"][:, :, 0]
    rows = []
    fig, axes = plt.subplots(2, len(PANELS), figsize=(3.2 * len(PANELS), 6), squeeze=False)
    for i, panel in enumerate(PANELS):
        windows = {panel.window_s(c) for c in range(len(panel.columns))}
        by_column = {c: panel_fractions(panel, bouts, fish, c) for c in range(len(panel.columns))} if len(windows) > 1 \
            else dict.fromkeys(range(len(panel.columns)), panel_fractions(panel, bouts, fish, 0))
        ours = by_column[0]
        aligned = {c: f.reindex(ours.index)[FIG3A_TYPES].fillna(0).to_numpy() for c, f in by_column.items()}
        for k, unit in enumerate(ours.index):
            distance = [0.5 * np.abs(aligned[c][k] - paper[col]).sum() for c, col in enumerate(panel.columns)]
            best, predicted = int(np.argmin(distance)), panel.predicted(*unit)
            rows.append({
                "panel": panel.name, "experiment": unit[0], "stimulus": unit[1], "label": stimulus_label(*unit),
                "n_fish": int(ours.loc[unit, "n_fish"]), "best_column": best, "best_distance": distance[best],
                "predicted_column": predicted,
                "predicted_distance": distance[predicted] if predicted is not None else np.nan,
                "match": (best == predicted) if predicted is not None else None,
            })
        best = {(r["experiment"], r["stimulus"]): r["best_column"] for r in rows if r["panel"] == panel.name}
        order = sorted(ours.index, key=best.get)
        axes[0, i].imshow(paper[panel.columns].T, aspect="auto", cmap="hot", vmin=0, vmax=0.8)
        axes[1, i].imshow(ours.loc[order, FIG3A_TYPES].to_numpy().T, aspect="auto", cmap="hot", vmin=0, vmax=0.8)
        axes[0, i].set_title(panel.name, fontsize=8)
        axes[1, i].set_xlabel("our stimuli, by best column", fontsize=7)
        for ax in axes[:, i]:
            ax.set_yticks(range(13), FIG3A_TYPES if i == 0 else [""] * 13, fontsize=6)
            ax.set_xticks([])
    axes[0, 0].set_ylabel("paper (Fig. 3A)")
    axes[1, 0].set_ylabel("this mapping")
    fig.tight_layout()
    fig.savefig(out / "fig3a_check.png", dpi=150)
    return pd.DataFrame(rows)


def fig3a(root: Path, out: Path, ages: tuple[int, ...], stats: Path | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    stats = stats or next(root.rglob(FIG3A_STATS))
    bouts, fish = collect(root)
    fish = fish[fish["age"].isin(ages)]
    bouts = bouts[bouts["fish_key"].isin(fish["fish_key"])]
    result = fig3a_check(bouts, fish, stats, out)
    result.to_csv(out / "fig3a_check.csv", index=False)
    for panel, g in result.groupby("panel", sort=False):
        checked = g.dropna(subset=["predicted_column"])
        verdict = f"{int(checked['match'].sum())}/{len(checked)} stimuli at their predicted column" if len(checked) else "no prediction"
        print(f"\n{panel}: {verdict}")
        print(g[["experiment", "stimulus", "label", "n_fish", "best_column", "best_distance",
                 "predicted_column", "predicted_distance"]].round(2).to_string(index=False))


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
    a.add_argument("--fold", type=float, default=2.0, help="fold change to detect (default 2)")
    a.add_argument("--lines", type=int, default=60, help="tests to correct alpha for (default 60 lines)")
    f = sub.add_parser("fig3a", help="check the stimulus mapping against Fig. 3A of the paper")
    f.add_argument("root", type=Path)
    f.add_argument("out", type=Path)
    f.add_argument("--ages", type=int, nargs="+", default=[6, 7])
    f.add_argument("--stats", type=Path, help=f"{FIG3A_STATS} (default: found under ROOT)")
    args = parser.parse_args()
    if args.command == "download":
        download(args.out)
    elif args.command == "fig3a":
        fig3a(args.root, args.out, tuple(args.ages), args.stats)
    else:
        analyze(args.root, args.out, tuple(args.ages), args.min_fish, args.table_s2, args.fold, args.lines)
