"""
Between-fish variability of bout rates in the Marques et al. 2018 tracking data.

    python marques_cv.py download ./marques_data
    python marques_cv.py analyze ./marques_data ./marques_cv

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
STIM, FRAME = 2, 3
DATASET, PROTOCOL, AGE = 2, 3, 10


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
        try:
            fish = load_fish(data)
        except Exception as error:  # truncated or otherwise unreadable .mat: report, keep going
            print(f"    unreadable ({type(error).__name__}: {error}), skipped")
            continue
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


def cv_by_condition(rates: pd.DataFrame, min_fish: int) -> pd.DataFrame:
    def summarize(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "n_fish": len(g), "rate_hz": g["rate"].mean(), "T_s": g["T"].median(), "mu_bouts": g["bouts"].mean(),
            "cv_observed": g["rate"].std(ddof=1) / g["rate"].mean(),
            "bouts_per_run": (g["bouts"] + g["runs"]).sum() / g["runs"].sum(),
            "single_fish_arena": (g["fish_in_file"] == 1).all(),
            "dataset_name": g["dataset_name"].iloc[0], "protocol_name": g["protocol_name"].iloc[0],
        })

    rates = rates.dropna(subset=["rate"]).groupby(KEYS).filter(lambda g: len(g) >= min_fish)
    summary = rates.groupby(KEYS).apply(summarize).reset_index()
    return summary.assign(reliable=(summary["bouts_per_run"] >= MIN_BOUTS_PER_RUN) & (summary["mu_bouts"] >= MIN_BOUTS))


def fish_needed(cv: float, fold: float, alpha: float) -> int:
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(POWER)
    return int(np.ceil(2 * z**2 * np.log(1 + cv**2) / np.log(fold) ** 2))


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
    rates = collect(root)
    table_s2 = table_s2 or root / TABLE_S2
    labels = label_conditions(rates, parse_table_s2(table_s2)) if table_s2.exists() else None
    rates = rates[rates["age"].isin(ages)]
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
    quantiles = {"median": table.loc["all", "50%"], "90th percentile": table.loc["all", "90%"], "max": table.loc["all", "max"]}
    print(f"\nFish per group for a {fold:g}-fold change, {POWER:.0%} power:")
    for name, cv in quantiles.items():
        print(f"  CV {cv:.2f} ({name}): alpha 0.05 -> {fish_needed(cv, fold, 0.05)}, "
              f"alpha 0.05/{lines} -> {fish_needed(cv, fold, 0.05 / lines)}")


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
    args = parser.parse_args()
    if args.command == "download":
        download(args.out)
    else:
        analyze(args.root, args.out, tuple(args.ages), args.min_fish, args.table_s2, args.fold, args.lines)
