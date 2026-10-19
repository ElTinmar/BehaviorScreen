"""
Between-fish variability of bout rates in the Marques et al. 2018 tracking data.

    python marques_cv.py download ./marques_data
    python marques_cv.py analyze ./marques_data ./marques_cv

`download` fetches the five Mendeley parts and Table S2 (mmc2.xlsx). Run it again to
re-download only the files that are missing, of the wrong size, or zips failing their
CRC check. If the API refuses, use "Download All" on each dataset page (links in PARTS).

`analyze` reads the .mat files straight from the zips in each "Experiments with
Continuous Tracking" folder (discontinuous tracking is skipped). Conditions are
labelled by experiment (zip name), dataset and protocol names (EnumeratorDataSet.m and
EnumeratorProtocol.m, if found under ROOT) and Table S2 rows matched by bout count.

`analyze` estimates, per fish and stimulus, the bout rate from runs of consecutive
bouts under the same stimulus number: (bouts - 1) / (last onset - first onset) per
run, pooled over runs. Interbout data are not in the files, so the time before the
first and after the last bout of a run is unknown; fish without two bouts in a run
are missing, which biases the CV down for low-rate conditions.

Per (experiment, dataset, protocol, stimulus) with enough fish, the observed CV of rates is split
into Poisson noise and between-fish heterogeneity (method of moments):

    Var(rate) = c^2 m^2 + mean(rate_i / T_i)   ->   c^2 = (Var - mean(rate_i / T_i)) / m^2

c is the frailty CV used in the power analysis; mu is the mean number of bouts per fish.
The run-based rate is only trustworthy when runs hold many bouts: conditions with
fewer than MIN_BOUTS_PER_RUN bouts per run (rare-event stimuli) are reported but left
out of the summary, as their c is biased towards zero.
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
from typing import Iterator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io

TABLE_S2_URL = "https://www.cell.com/cms/10.1016/j.cub.2017.12.002/attachment/d751b425-5672-4343-b3d3-f86449108bf9/mmc2.xlsx"
TABLE_S2 = "mmc2.xlsx"
PARTS = ["r9vn7x287r", "xf2zv4yhv8", "bzh35dd2td", "fgpbbv29rw", "6hsdh9gj89"]
API = "https://data.mendeley.com/public-api/datasets"
VERSION = 1
FPS = 700.0
MIN_BOUTS_PER_RUN = 5
CONTINUOUS = "Experiments with Continuous Tracking"
KEYS = ["experiment", "dataset", "protocol", "stimulus"]

# BoutInf / FishInf columns (0-based), from Tracking_Data_Information
STIM, FRAME = 2, 3
DATASET, PROTOCOL, AGE = 2, 3, 10


BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


def open_url(url: str, referer: str = "https://data.mendeley.com/"):
    return urllib.request.urlopen(urllib.request.Request(url, headers={**BROWSER_HEADERS, "Referer": referer}), timeout=60)


def save(url: str, target: Path, referer: str = "https://data.mendeley.com/") -> None:
    """Download to a .part file renamed on completion, so an interrupted download never looks complete."""
    partial = target.with_name(target.name + ".part")
    with open_url(url, referer) as response, partial.open("wb") as f:
        while chunk := response.read(1 << 20):
            f.write(chunk)
    partial.replace(target)


def intact(path: Path, size: int) -> bool:
    """Expected size, and for a zip every member passing its CRC check."""
    if not path.exists() or path.stat().st_size != size:
        return False
    if path.suffix != ".zip":
        return True
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None
    except (zipfile.BadZipFile, zlib.error, OSError):
        return False


def get_json(url: str) -> list:
    with open_url(url) as response:
        return json.load(response)


def download_folder(dataset: str, folder_id: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for f in get_json(f"{API}/{dataset}/files?folder_id={folder_id}&version={VERSION}"):
        target = out / f["filename"]
        if intact(target, f["size"]):
            continue
        action = "broken, downloading again" if target.exists() else "downloading"
        print(f"  {target} ({f['size'] / 1e6:.0f} MB): {action}")
        save(f["content_details"]["download_url"], target)
        if not intact(target, f["size"]):
            print(f"  {target}: still fails the integrity check after download")


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
            print(f"  {archive_path.name}: {error}, skipped (incomplete download?)")
            continue
        with archive:
            for name in sorted(n for n in archive.namelist() if n.endswith(".mat") and "__MACOSX" not in n):
                try:
                    data = archive.read(name)
                except (zipfile.BadZipFile, zlib.error) as error:
                    print(f"  {archive_path.name}/{name}: {error}, skipped (extract the zip to {extracted.name}/)")
                    continue
                yield archive_path.stem, f"{archive_path.name}/{name}", data


def load_fish(data: bytes) -> list[dict]:
    """boutStructure as a list of {field: array}, for both pre-7.3 and 7.3 (HDF5) .mat files."""
    try:
        content = scipy.io.loadmat(io.BytesIO(data), squeeze_me=True, struct_as_record=False)
        if "boutStructure" not in content:
            print(f"    no boutStructure, skipped (variables: {[k for k in content if not k.startswith('__')]})")
            return []
        s = np.atleast_1d(content["boutStructure"])
        return [{"BoutInf": np.atleast_2d(f.BoutInf), "FishInf": np.ravel(f.FishInf)} for f in s]
    except NotImplementedError:
        import mat73
        with tempfile.NamedTemporaryFile(suffix=".mat") as tmp:
            tmp.write(data)
            tmp.flush()
            s = mat73.loadmat(tmp.name)["boutStructure"]
        return [{"BoutInf": np.atleast_2d(b), "FishInf": np.ravel(i)} for b, i in zip(s["BoutInf"], s["FishInf"])]


def parse_enumerator(root: Path, name: str) -> dict[int, str]:
    """{number: name} from a MATLAB enumerator class (several names for one number are joined)."""
    path = next(root.rglob(f"{name}.m"), None)
    if path is None:
        return {}
    names: dict[int, list[str]] = {}
    for key, value in re.findall(r"^\s*(\w+)\s*=\s*(\d+)", path.read_text(errors="ignore"), flags=re.M):
        names.setdefault(int(value), []).append(key)
    return {number: "/".join(keys) for number, keys in names.items()}


def fish_rates(bout_inf: np.ndarray) -> pd.DataFrame:
    """Bouts, observed time and rate per stimulus, from runs of consecutive bouts under one stimulus."""
    bouts = bout_inf[np.argsort(bout_inf[:, FRAME])] if bout_inf.size else bout_inf
    t, stim = bouts[:, FRAME] / FPS, bouts[:, STIM]
    run = np.concatenate([[0], np.cumsum(stim[1:] != stim[:-1])])
    runs = pd.DataFrame({"stimulus": stim, "t": t, "run": run}).groupby("run").agg(
        stimulus=("stimulus", "first"), n=("t", "size"), span=("t", lambda x: x.max() - x.min())
    )
    runs = runs[runs["n"] >= 2]
    per_stim = runs.groupby("stimulus").agg(
        bouts=("n", lambda n: (n - 1).sum()), runs=("n", "size"), T=("span", "sum")
    ).reset_index()
    per_stim = per_stim[per_stim["T"] > 0].assign(rate=lambda d: d["bouts"] / d["T"])
    counts = pd.Series(stim).value_counts().rename_axis("stimulus").rename("n_raw").reset_index()
    return counts.merge(per_stim, on="stimulus", how="left")


def collect(root: Path) -> pd.DataFrame:
    rows = []
    for experiment, member, data in mat_files(root):
        print(f"  {member}")
        fish = load_fish(data)
        for i, f in enumerate(fish):
            if f["BoutInf"].size == 0:
                continue
            info = f["FishInf"]
            rows.append(fish_rates(f["BoutInf"]).assign(
                experiment=experiment, file=member, fish=i, fish_in_file=len(fish),
                dataset=int(info[DATASET]), protocol=int(info[PROTOCOL]), age=int(info[AGE]),
            ))
    rates = pd.concat(rows, ignore_index=True)
    datasets, protocols = parse_enumerator(root, "EnumeratorDataSet"), parse_enumerator(root, "EnumeratorProtocol")
    return rates.assign(dataset_name=rates["dataset"].map(datasets), protocol_name=rates["protocol"].map(protocols))


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
    """Match each condition to the Table S2 rows whose bout count is within `tolerance`."""
    totals = rates.groupby(KEYS)["n_raw"].sum().reset_index()

    def match(n: int) -> str:
        close = table[(table["n_bouts"] - n).abs() <= tolerance * n]
        return "; ".join(f"{r.section}: {r.condition} ({r.n_bouts})" for r in close.itertuples())

    return totals.assign(table_s2=totals["n_raw"].map(match))


def heterogeneity(rates: pd.DataFrame, min_fish: int) -> pd.DataFrame:
    def summarize(g: pd.DataFrame) -> pd.Series:
        m, var = g["rate"].mean(), g["rate"].var(ddof=1)
        poisson = (g["rate"] / g["T"]).mean()
        return pd.Series({
            "n_fish": len(g), "rate_hz": m, "T_s": g["T"].median(), "mu_bouts": g["bouts"].mean(),
            "cv_observed": np.sqrt(var) / m, "c_frailty": np.sqrt(max(var - poisson, 0) / m**2),
            "bouts_per_run": (g["bouts"] + g["runs"]).sum() / g["runs"].sum(),
            "single_fish_arena": (g["fish_in_file"] == 1).all(),
            "dataset_name": g["dataset_name"].iloc[0], "protocol_name": g["protocol_name"].iloc[0],
        })

    rates = rates.dropna(subset=["rate"])
    rates = rates.groupby(KEYS).filter(lambda g: len(g) >= min_fish)
    return rates.groupby(KEYS).apply(summarize).reset_index()


def plot(summary: pd.DataFrame, out: Path) -> None:
    summary = summary[summary["bouts_per_run"] >= MIN_BOUTS_PER_RUN]
    fig, (ax_hist, ax_mu) = plt.subplots(1, 2, figsize=(10, 4))
    bins = np.linspace(0, max(2.0, summary["c_frailty"].max()), 30)
    for single, label in [(True, "one fish per arena"), (False, "several fish (identity swaps)")]:
        sub = summary[summary["single_fish_arena"] == single]
        ax_hist.hist(sub["c_frailty"], bins=bins, alpha=0.6, label=f"{label} (n={len(sub)})")
        ax_mu.scatter(sub["mu_bouts"], sub["c_frailty"], s=12, alpha=0.7, label=label)
    ax_hist.set(xlabel="between-fish CV of bout rate (c)", ylabel="stimulus conditions")
    ax_mu.set(xscale="log", xlabel="mean bouts per fish (mu)", ylabel="c")
    ax_hist.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "cv_distribution.png", dpi=150)


def analyze(root: Path, out: Path, ages: tuple[int, ...], min_fish: int, table_s2: Path | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rates = collect(root)
    table_s2 = table_s2 or root / TABLE_S2
    labels = label_conditions(rates, parse_table_s2(table_s2)) if table_s2.exists() else None
    rates = rates[rates["age"].isin(ages)]
    rates.to_csv(out / "fish_rates.csv", index=False)
    summary = heterogeneity(rates, min_fish)
    plot(summary, out)
    summary["reliable"] = summary["bouts_per_run"] >= MIN_BOUTS_PER_RUN
    if labels is not None:
        summary = summary.merge(labels, on=KEYS, how="left")
        labels.to_csv(out / "table_s2_matches.csv", index=False)
    summary.to_csv(out / "cv_by_condition.csv", index=False)
    single = summary[summary["single_fish_arena"] & summary["reliable"]]
    print(f"{len(summary)} conditions, {len(single)} reliable with one fish per arena; ages {ages}, >= {min_fish} fish")
    columns = ["rate_hz", "mu_bouts", "cv_observed", "c_frailty"]
    print(single[columns].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(2))
    shown = ["experiment", "protocol_name", "stimulus", "n_fish", *columns] + (["table_s2"] if labels is not None else [])
    print(single[shown].round(2).to_string(index=False))


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
    args = parser.parse_args()
    if args.command == "download":
        download(args.out)
    else:
        analyze(args.root, args.out, tuple(args.ages), args.min_fish, args.table_s2)
