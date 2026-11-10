"""
Where the variance of the fish gain sits, in the controls of each experiment (default: the Tier 1 columns): time of
day, recording day, plate, rig, position within the plate and genotype (line); and a summary across experiments.

Each fish's posterior gain E[g | data] under the experiment's best model (its fish-level residual once the stimulus,
history and trial gains are accounted for) and its rates of selected events and other bouts, with its covariates:
line, day and time of day from the bout table (day, cos/sin_daytime; the time of day is the circular mean over the
fish's bouts, the middle of its recording), plate (the file name less _fish_N), position in the plate (N) and rig
(calibration.pix_per_mm in the recording's .metadata, split at RIG_SPLIT_PX_PER_MM).

  time_of_day      Within-day slopes per hour of log gain, selected-event rate, other-bout rate and their ratio,
                   on plate means (fish of a plate share their time), with the rig means removed as a check.
  components       Variance of log gain between days, between plates within a day and between fish within a
                   plate (nested ANOVA, method of moments), 95% intervals from resampling days; as recorded, with
                   the time trend removed, with line within day as the upper level, and both.
  genotype         Day and line are nearly the same comparison in the controls; two subsets tell them apart (time
                   trend removed): the same line on different days (line, then day within line: clutch and day
                   conditions without genotype) and different lines on the same day (day, then line within day:
                   genotype and clutch without the day's conditions); 95% intervals from resampling lines or days.
                   And the mean log gain of each WT day.
  rig              Rig means, and rig differences within the days that used both (log gain, rates); components
                   with the rig means removed.
  position         Log gain less its plate's mean, by position and rig; one-way ANOVA within plates.

Posterior gains are shrunk toward 1, more for fish with few events (survival experiments: one escape per trial at
most): effects come out smaller than they are, never larger. These show where the variance sits, not the size of a
model term. Sided models: the fish gain only, not the side preference. Without other bouts (saccades), no other-bout
rate.

Run from the repo root with run_analysis's data options (writes <output>/fish.csv), or again on that table:
    python fish_gain_variance.py -d <data root> --bouts bouts_control.csv [--exps prey_capture_ipsi spont_dark] [--fits-root figures]
    python fish_gain_variance.py --from-csv fish_gain_variance/fish.csv [--exps ...] [-o fish_gain_variance] [--n-boot 500]
Outputs (<output>/): fish.csv (every experiment), summary.csv and summary.png (one row per experiment), and per
experiment (<output>/<experiment>/): one CSV per table, report.txt (every table) and fish_gain_variance.png.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RIG_SPLIT_PX_PER_MM = 39.0  # between the two setups (about 38 and 40)
DAY_FORMAT = "%d.%b.%Y"
N_BOOT = 500  # --n-boot


def covariates(args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, Path]]:
    """Line, day and mean cos/sin_daytime of every file of the bout table, and every .metadata file under the data root."""
    bouts = pd.read_csv(args.data_root / args.bouts, usecols=["file", "line", "day", "cos_daytime", "sin_daytime"], dtype={"line": str, "day": str})
    cov = bouts.groupby("file").agg(line=("line", "first"), day=("day", "first"), cos=("cos_daytime", "mean"), sin=("sin_daytime", "mean"))
    return cov, {m.stem: m for m in args.data_root.rglob("*.metadata")}


def fish_table(args: argparse.Namespace, experiment: str, cov: pd.DataFrame, metadata: Dict[str, Path]) -> pd.DataFrame:
    """Posterior gain, rates and covariates of each fish of the experiment's controls."""
    from BehaviorScreen.point_process.analysis.model_config import get_model_config
    from BehaviorScreen.point_process.analysis.run_analysis import make_loaders
    from BehaviorScreen.point_process.analysis.tasks import load_best_model

    config = get_model_config()[experiment]
    model = load_best_model(config, args.fits_root / experiment / "models")
    dataset = make_loaders(args)(config["events"]).prepare_dataset(**config["dataset"])
    try:
        gains = model.estimate_fish_gains(dataset)
    except (AttributeError, NotImplementedError):
        raise ValueError(f"its best model has no fish gain: {model.name}") from None
    idx = gains.fish_idx.to_numpy()
    ids = dataset.fish_ids[idx]
    missing = (set(ids) - set(cov.index)) | (set(ids) - set(metadata))
    if missing:
        raise ValueError(f"{len(missing)} fish without bout-table rows or a .metadata file, e.g. {sorted(missing)[0]}")
    c = cov.loc[ids]

    seconds = dataset.fish_trial_mask.sum(axis=1)[idx] * dataset.duration_s
    selected = np.bincount(dataset.event_fish_idx, minlength=dataset.num_fish)[idx]
    history = dataset.history
    if history is None:
        other = np.full(len(idx), np.nan)
    else:  # the history also holds each trial's last earlier bout
        other = np.bincount(history.event_fish_idx[history.event_times >= 0], minlength=dataset.num_fish)[idx] - selected
        other = np.where((other == 0).all(), np.nan, other)  # the experiment selects every bout: no other bouts
    px = np.array([json.loads(metadata[i].read_text())["calibration"]["pix_per_mm"] for i in ids], dtype=float)
    fish = pd.DataFrame({
        "experiment": experiment, "model": model.name,
        "fish": ids, "line": c.line.values, "day": c.day.values,
        "recording": pd.Series(ids).str.replace(r"_fish_\d+$", "", regex=True).values,
        "position": pd.Series(ids).str.extract(r"_fish_(\d+)$")[0].astype(int).values,
        "hour": np.mod(np.arctan2(c.sin, c.cos), 2 * np.pi).values * 24 / (2 * np.pi),
        "px_per_mm": px, "rig": np.where(px > RIG_SPLIT_PX_PER_MM, "40 px/mm", "38 px/mm"),
        "n_events": gains.n_events.values, "gain": gains.estimated_gain.values,
        "selected_hz": selected / seconds, "other_hz": other / seconds,
    })
    return fish.assign(log_gain=np.log(fish.gain))


def nested(y: pd.Series, upper: pd.Series, middle: pd.Series) -> pd.Series:
    """
    Variance between upper groups, between middle groups within an upper group and within middle groups:
    nested ANOVA with unbalanced groups by the method of moments (Sokal and Rohlf). Middle labels must be unique
    across upper groups.
    """
    N = len(y)
    n_m, n_u = middle.value_counts(), upper.value_counts()
    upper_of = upper.groupby(middle).first()
    U, M = len(n_u), len(n_m)
    if U < 3 or M <= U or N <= M:
        return pd.Series([np.nan] * 3)
    mean_m, mean_u = y.groupby(middle).transform("mean"), y.groupby(upper).transform("mean")
    ms_within = ((y - mean_m) ** 2).sum() / (N - M)
    ms_middle = ((mean_m - mean_u) ** 2).sum() / (M - U)
    ms_upper = ((mean_u - y.mean()) ** 2).sum() / (U - 1)
    within_upper = (n_m**2 / n_u[upper_of[n_m.index]].values).sum()
    c_middle = (N - within_upper) / (M - U)
    c_middle_upper = (within_upper - (n_m**2).sum() / N) / (U - 1)
    c_upper = (N - (n_u**2).sum() / N) / (U - 1)
    v_middle = max((ms_middle - ms_within) / c_middle, 0.0)
    v_upper = max((ms_upper - ms_within - c_middle_upper * v_middle) / c_upper, 0.0)
    return pd.Series([v_upper, v_middle, ms_within])


def components(fish: pd.DataFrame, upper: str, middle: str, names: tuple, rng: np.random.Generator, resample: Optional[str] = None) -> pd.DataFrame:
    """nested() of log gain with share of the sum and 95% intervals from resampling `resample` groups (default: upper)."""
    resample = resample or upper
    estimate = lambda d: nested(d.log_gain, d[upper].astype(str), d[upper].astype(str) + " | " + d[middle].astype(str))
    groups = {g: d for g, d in fish.groupby(resample)}
    keys = list(groups)
    draw = lambda: pd.concat([groups[keys[k]].assign(**{c: f"{i}#" + groups[keys[k]][c].astype(str) for c in {upper, middle, resample}})
                              for i, k in enumerate(rng.choice(len(keys), len(keys)))])  # a group drawn twice: two groups
    point = estimate(fish)
    boot = pd.DataFrame([estimate(draw()) for _ in range(N_BOOT)] if point.notna().all() else [point])
    table = pd.DataFrame({"variance": point, "ci 2.5%": boot.quantile(0.025), "ci 97.5%": boot.quantile(0.975),
                          "share": point / point.sum(), "sd": np.sqrt(point)})
    return table.set_axis(list(names))


def within_day_slope(plates: pd.DataFrame, y: str) -> Dict[str, float]:
    """Slope per hour of y on time of day within days, on plate means; SE from the residuals (days as fixed effects)."""
    plates = plates[np.isfinite(plates[y])]
    plates = plates[plates.groupby("day").hour.transform("size") > 1]
    dof = len(plates) - plates.day.nunique() - 1
    if dof < 2:
        return {"slope per h": np.nan, "se": np.nan, "t": np.nan, "fold over 6 h": np.nan}
    centred = lambda c: plates[c] - plates.groupby("day")[c].transform("mean")
    x, v = centred("hour"), centred(y)
    slope = (x * v).sum() / (x**2).sum()
    se = np.sqrt(((v - slope * x) ** 2).sum() / dof / (x**2).sum())
    return {"slope per h": slope, "se": se, "t": slope / se, "fold over 6 h": np.exp(6 * slope)}


def plate_means(fish: pd.DataFrame) -> pd.DataFrame:
    """Per plate: day, rig, time, mean log gain and the log of its mean rates (robust to fish without events)."""
    plates = fish.groupby("recording").agg(day=("day", "first"), rig=("rig", "first"), hour=("hour", "mean"), log_gain=("log_gain", "mean"),
                                           selected=("selected_hz", "mean"), other=("other_hz", lambda x: x.mean(skipna=False)))
    with np.errstate(divide="ignore"):
        return plates.assign(log_selected=np.log(plates.selected), log_other=np.log(plates.other),
                             log_selected_per_other=np.log(plates.selected / plates.other))


def time_of_day(fish: pd.DataFrame) -> pd.DataFrame:
    plates = plate_means(fish)
    rows = {label: within_day_slope(plates, c) for c, label in (
        ("log_gain", "log gain"), ("log_selected", "log selected-event rate"), ("log_other", "log other-bout rate"),
        ("log_selected_per_other", "log selected events per other bout"))}
    no_rig = plates.assign(log_gain=plates.log_gain - plates.groupby("rig").log_gain.transform("mean"))
    rows["log gain, rig means removed"] = within_day_slope(no_rig, "log_gain")
    within = plates[plates.groupby("day").hour.transform("size") > 1]  # days with one plate say nothing within days
    return pd.DataFrame(rows).T.assign(plates=len(within), days=within.day.nunique())


def detrended(fish: pd.DataFrame) -> pd.DataFrame:
    """log gain less the within-day time trend (the plate-level slope)."""
    slope = within_day_slope(plate_means(fish), "log_gain")["slope per h"]
    return fish.assign(log_gain=fish.log_gain - slope * (fish.hour - fish.groupby("day").hour.transform("mean")))


def all_components(fish: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    names = ("day (or day x line)", "plate within it", "fish within plate")
    flat, lined = detrended(fish), lambda d: d.assign(day_line=d.day + " " + d.line)
    variants = {
        "as recorded": components(fish, "day", "recording", names, rng),
        "time trend removed": components(flat, "day", "recording", names, rng),
        "day x line as upper level": components(lined(fish), "day_line", "recording", names, rng, resample="day"),
        "both": components(lined(flat), "day_line", "recording", names, rng, resample="day"),
    }
    return pd.concat(variants, names=["variant", "level"])


def genotype(fish: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    flat = detrended(fish).assign(day_line=lambda d: d.day + " " + d.line)
    same_line = flat[flat.groupby("line").day.transform("nunique") > 1]
    same_day = flat[flat.groupby("day").line.transform("nunique") > 1]
    return pd.concat({
        f"same line, different days ({same_line.line.nunique()} lines, {same_line.day_line.nunique()} line-days)":
            components(same_line, "line", "day_line", ("line: genotype", "day within line: clutch + day conditions", "plates + fish"), rng),
        f"same day, different lines ({same_day.day.nunique()} days, {same_day.day_line.nunique()} line-days)":
            components(same_day, "day", "day_line", ("day: shared conditions", "line within day: genotype + clutch", "plates + fish"), rng),
    }, names=["subset", "level"])  # NaN where a subset has too few groups


def wt_days(fish: pd.DataFrame) -> pd.DataFrame:
    wt = detrended(fish)[fish.line == "WT"]
    return wt.groupby("day").agg(fish=("log_gain", "size"), plates=("recording", "nunique"), mean_log_gain=("log_gain", "mean"), sem=("log_gain", "sem"))


def rig(fish: pd.DataFrame, rng: np.random.Generator) -> Dict[str, pd.DataFrame]:
    by_rig = fish.groupby("rig").agg(fish=("gain", "size"), plates=("recording", "nunique"), days=("day", "nunique"),
                                     mean_log_gain=("log_gain", "mean"), selected_hz=("selected_hz", "median"), other_hz=("other_hz", "median"))
    calibrations = fish.groupby("recording").px_per_mm.first().round(2).value_counts().sort_index().rename("plates").to_frame()
    tables = {"rig_means": by_rig, "rig_calibrations": calibrations}
    per_day = fish.groupby(["day", "rig"])[["log_gain", "selected_hz", "other_hz"]].mean().unstack("rig")
    per_day = per_day[per_day.log_gain.notna().all(axis=1)]  # days with both rigs
    if fish.rig.nunique() == 2 and len(per_day) > 1:
        a, b = sorted(fish.rig.unique())
        diff = pd.DataFrame({"log gain": per_day.log_gain[b] - per_day.log_gain[a],
                             "log selected-event rate": np.log(per_day.selected_hz[b] / per_day.selected_hz[a]),
                             "log other-bout rate": np.log(per_day.other_hz[b] / per_day.other_hz[a])})
        se = diff.std() / np.sqrt(len(diff))
        tables["rig_within_days"] = pd.DataFrame({"mean": diff.mean(), "se": se, "t": diff.mean() / se, "days": len(diff)}).rename_axis(f"{b} less {a}")
    no_rig = fish.assign(log_gain=fish.log_gain - fish.groupby("rig").log_gain.transform("mean"))
    tables["rig_removed_components"] = components(no_rig, "day", "recording", ("day", "plate within day", "fish within plate"), rng)
    return tables


def position(fish: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    within = fish.assign(within=fish.log_gain - fish.groupby("recording").log_gain.transform("mean"))
    by = within.groupby(["position", "rig"]).within.agg(["mean", "sem", "size"]).unstack("rig")
    test = lambda d: stats.f_oneway(*[g.within for _, g in d.groupby("position")])
    complete = within[within.groupby("recording").position.transform("nunique") == within.position.nunique()]
    anova = pd.DataFrame([{"fish": "all", **dict(zip(("F", "p"), test(within)))},
                          {"fish": "plates with every position", **dict(zip(("F", "p"), test(complete)))},
                          *[{"fish": r, **dict(zip(("F", "p"), test(d)))} for r, d in within.groupby("rig")]]).set_index("fish")
    ss = lambda x: ((x - x.mean()) ** 2).sum()
    anova["share of within-plate variance"] = [1 - sum(ss(g.within) for _, g in d.groupby("position")) / ss(d.within)
                                               for d in (within, complete, *[d for _, d in within.groupby("rig")])]
    return {"position_by_rig": by, "position_anova": anova}


def plot(fish: pd.DataFrame, slopes: pd.DataFrame, comps: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    centred = fish.assign(log_gain=fish.log_gain - fish.groupby("day").log_gain.transform("mean"),
                          hour=fish.hour - fish.groupby("day").hour.transform("mean"))
    for ax, d, title, xlabel in ((axes[0, 0], fish, "across days", "time of day (h)"),
                                 (axes[0, 1], centred, "within days (day means removed)", "time of day less its day's mean (h)")):
        ax.scatter(d.hour, d.log_gain, s=5, alpha=0.35, color="k", lw=0)
        means = d.groupby(pd.cut(d.hour, 8), observed=True).log_gain.agg(["mean", "sem"])
        ax.errorbar([b.mid for b in means.index], means["mean"], 2 * means["sem"], color="#ea580c", marker="o", ms=3, lw=1)
        ax.set(title=title, xlabel=xlabel, ylabel="log posterior fish gain")
    s = slopes.iloc[:4]
    axes[0, 2].barh(range(len(s)), s["slope per h"], xerr=2 * s["se"], color="#2563eb", height=0.6)
    axes[0, 2].set_yticks(range(len(s)), s.index, fontsize=8)
    axes[0, 2].invert_yaxis()
    axes[0, 2].axvline(0, color="0.4", lw=0.8)
    axes[0, 2].set(title=f"within-day slope, {int(s.plates.iloc[0])} plate means (±2 SE)", xlabel="change per hour (log)")
    order = sorted(fish.day.unique(), key=lambda d: pd.to_datetime(d, format=DAY_FORMAT))
    axes[1, 0].boxplot([fish.log_gain[fish.day == d] for d in order], showfliers=False, widths=0.6)
    axes[1, 0].set_xticks(range(1, len(order) + 1), order, rotation=90, fontsize=5)
    axes[1, 0].set(title="by recording day", ylabel="log posterior fish gain")
    shares = comps.share.unstack("level")[list(comps.index.get_level_values("level").unique())]
    left = np.zeros(len(shares))
    for (level, values), color in zip(shares.items(), ("#ea580c", "#2563eb", "#cbd5e1")):
        axes[1, 1].barh(range(len(shares)), values, left=left, color=color, label=level, height=0.6)
        left += values.values
    axes[1, 1].set_yticks(range(len(shares)), shares.index, fontsize=8)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set(title="variance of log gain", xlabel="share")
    axes[1, 1].legend(frameon=False, fontsize=7, loc="lower right")
    within = fish.log_gain - fish.groupby("recording").log_gain.transform("mean")
    for (r, d), dx, color in zip(within.groupby(fish.rig), (-0.12, 0.12), ("#0f766e", "#9333ea")):
        m = d.groupby(fish.position[d.index]).agg(["mean", "sem"])
        axes[1, 2].errorbar(m.index + dx, m["mean"], 2 * m["sem"], fmt="o", color=color, label=r, ms=4)
    axes[1, 2].set(title="position in the plate (±2 SE)", xlabel="position", ylabel="log gain less its plate's mean", xticks=sorted(fish.position.unique()))
    axes[1, 2].legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        if ax not in (axes[0, 2], axes[1, 1]):
            ax.axhline(0, color="0.6", lw=0.8, ls="--")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def analyse(fish: pd.DataFrame, rng: np.random.Generator, output: Path) -> Dict[str, pd.DataFrame]:
    """Every table of one experiment, its report and figure in output/."""
    slopes, comps = time_of_day(fish), all_components(fish, rng)
    tables = {"time_of_day": slopes, "components": comps, "genotype": genotype(fish, rng), "wt_days": wt_days(fish),
              **rig(fish, rng), **position(fish)}
    header = (f"{fish.experiment.iloc[0]}: {fish.model.iloc[0]}\n{len(fish)} fish (median {fish.n_events.median():.0f} events), "
              f"{fish.recording.nunique()} plates, {fish.day.nunique()} days, {fish.line.nunique()} lines; time of day "
              f"{fish.hour.min():.1f} to {fish.hour.max():.1f} h; total variance of log gain {fish.log_gain.var():.3f}")
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        report = header + "".join(f"\n\n== {name}\n{table.to_string()}" for name, table in tables.items())
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.txt").write_text(report + "\n")
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv")
    fig = plot(fish, slopes, comps)
    fig.savefig(output / "fish_gain_variance.png", dpi=200)
    plt.close(fig)
    return tables


def summary_row(fish: pd.DataFrame, tables: Dict[str, pd.DataFrame]) -> dict:
    t, c, g = tables["time_of_day"], tables["components"], tables["genotype"]
    share = lambda variant, i: c.loc[variant].share.iloc[i]
    gv = lambda i: g.variance.iloc[i] if len(g) > i else np.nan
    within = tables.get("rig_within_days")
    return {
        "experiment": fish.experiment.iloc[0], "model": fish.model.iloc[0], "fish": len(fish), "median events": fish.n_events.median(),
        "slope log gain /h": t.loc["log gain", "slope per h"], "se": t.loc["log gain", "se"], "t": t.loc["log gain", "t"],
        "slope selected /h": t.loc["log selected-event rate", "slope per h"], "slope other bouts /h": t.loc["log other-bout rate", "slope per h"],
        "slope selected per other /h": t.loc["log selected events per other bout", "slope per h"],
        "share day": share("as recorded", 0), "share plate": share("as recorded", 1),
        "share day x line (adjusted)": share("both", 0), "share plate (adjusted)": share("both", 1),
        "var genotype": gv(0), "var day within line": gv(1),
        "rig diff log gain": within.loc["log gain", "mean"] if within is not None else np.nan,
        "rig diff se": within.loc["log gain", "se"] if within is not None else np.nan,
        "position p": tables["position_anova"].loc["all", "p"],
    }


def plot_summary(summary: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.45 * len(summary) + 2), sharey=True)
    y = np.arange(len(summary))
    labels = [f"{e} ({n:.0f} ev./fish)" for e, n in zip(summary.experiment, summary["median events"])]
    axes[0].errorbar(summary["slope log gain /h"], y, xerr=2 * summary.se, fmt="o", color="k", label="log gain (±2 SE)")
    axes[0].scatter(summary["slope selected /h"], y + 0.2, marker="^", color="#ea580c", label="selected events", zorder=3)
    axes[0].scatter(summary["slope other bouts /h"], y - 0.2, marker="s", color="#2563eb", label="other bouts", zorder=3)
    axes[0].axvline(0, color="0.5", lw=0.8)
    axes[0].set(title="within-day slope on plate means", xlabel="change per hour (log)", yticks=y, yticklabels=labels)
    axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12 - 1.2 / (0.45 * len(summary) + 2)))
    left = np.zeros(len(summary))
    for column, color, label in (("share day x line (adjusted)", "#ea580c", "day x line"), ("share plate (adjusted)", "#2563eb", "plate within it")):
        values = summary[column].fillna(0).values
        axes[1].barh(y, values, left=left, color=color, height=0.6, label=label)
        left += values
    axes[1].set(title="variance of log gain, time trend removed", xlabel="share (rest: fish within plate)", xlim=(0, max(0.5, left.max() * 1.1)))
    axes[1].legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12 - 1.2 / (0.45 * len(summary) + 2)))
    axes[0].invert_yaxis()
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-csv", type=Path, help="A fish.csv of an earlier run: no data or model needed.")
    parser.add_argument("--exps", nargs="+", help="Experiments (default: the Tier 1 columns, or every experiment of --from-csv).")
    parser.add_argument("--fits-root", type=Path, default=Path("figures"))
    parser.add_argument("-o", "--output", type=Path, default=Path("fish_gain_variance"))
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=0)
    try:
        from BehaviorScreen.point_process.analysis.run_analysis import add_data_arguments
        add_data_arguments(parser, required=False)
    except ImportError:  # --from-csv outside the repo
        pass
    args = parser.parse_args()
    if args.from_csv is None and getattr(args, "data_root", None) is None:
        parser.error("give -d/--data-root (with the data options) or --from-csv")
    globals()["N_BOOT"] = args.n_boot
    args.output.mkdir(parents=True, exist_ok=True)

    if args.from_csv:
        fish = pd.read_csv(args.from_csv, dtype={"line": str, "day": str})
        fish = fish if "experiment" in fish else fish.assign(experiment="experiment", model="")
        fish = fish[fish.experiment.isin(args.exps)] if args.exps else fish
    else:
        from BehaviorScreen.ablation_screen.tier1_config import MAP_COLUMNS
        cov, metadata = covariates(args)
        tables = []
        for experiment in args.exps or list(MAP_COLUMNS):
            try:
                tables.append(fish_table(args, experiment, cov, metadata))
                print(f"{experiment}: {len(tables[-1])} fish, {tables[-1].model.iloc[0]}")
            except Exception as e:  # one experiment's missing fit or odd model does not stop the others
                print(f"{experiment}: skipped ({type(e).__name__}: {e})")
        fish = pd.concat(tables, ignore_index=True)
        fish.to_csv(args.output / "fish.csv", index=False)

    rows = []
    for experiment, f in fish.groupby("experiment", sort=False):
        tables = analyse(f.reset_index(drop=True), np.random.default_rng(args.seed), args.output / experiment)
        rows.append(summary_row(f, tables))
        print(f"{experiment}: done")
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output / "summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print("\n" + summary.drop(columns="model").to_string(index=False))
    fig = plot_summary(summary)
    fig.savefig(args.output / "summary.png", dpi=200)
    print(f"\nSaved {args.output}/: fish.csv, summary.csv, summary.png and a folder per experiment")


if __name__ == "__main__":
    main()
