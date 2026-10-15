"""
Absolute values of the Tier 1 free parameters in every arm, with fish
bootstrap intervals, drawn as one dumbbell figure per behavior: each test arm
next to its own control, rows in map order, the spread of the control arms
behind them.

    python -m BehaviorScreen.ablation_screen.arm_absolute [--n-boot 200] [--plot-only]

Run after run_screen (hits come from tier1_results.csv). The bootstrap is
resumable: arms already in arm_bootstrap.csv are skipped.
"""
import argparse
import zlib
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm

from BehaviorScreen.point_process.point_process.dataset import PointProcessDataset
from BehaviorScreen.point_process.point_process.io import save_fig
from BehaviorScreen.point_process.point_process.parallel import parallel_map
from BehaviorScreen.point_process.point_process.point_process import PointProcess

from .dataset_utils import Arms, Contrast
from .tier1_config import MAP_COLUMNS
from .tier1_omnibus import PARAMETER_LABELS, map_rows

ARM_KEYS = ["line", "condition", "behavior"]
CONTROL, UP, DOWN, NEUTRAL, TEST = "#4d4d4d", "#b2182b", "#2166ac", "#bdbdbd", "#8c8c8c"
BAND, MEDIAN = "#efefef", "#a6a6a6"
MIN_SUCCESS = 0.8  # below this fraction of successful refits the interval is drawn dashed


# ---------------------------------------------------------------------------
# Fish bootstrap
# ---------------------------------------------------------------------------

def _refit(seed: np.random.SeedSequence, factory: Callable[[], PointProcess], dataset: PointProcessDataset,
           start: np.ndarray) -> Optional[np.ndarray]:
    """Free parameters fitted to the arm's fish resampled with replacement (None if the fit fails)."""
    model = factory()
    model.initial_guesses = list(start)
    try:
        model.fit(dataset.resample(np.random.default_rng(seed)))
    except Exception:
        return None
    return model.params_


def bootstrap_arm(factory: Callable[[], PointProcess], model: PointProcess, dataset: PointProcessDataset,
                  n_boot: int, seed: int, n_jobs: int = -1, desc: str = "") -> pd.DataFrame:
    """Per free parameter: fitted value, 95% percentile interval of the refits, refits that succeeded, value on a bound."""
    draws = parallel_map(partial(_refit, factory=factory, dataset=dataset, start=model.params_),
                         np.random.SeedSequence(seed).spawn(n_boot), n_jobs=n_jobs, desc=desc)
    draws = np.array([d for d in draws if d is not None]).reshape(-1, len(model.param_names))
    records = []
    for i, (name, value, bounds) in enumerate(zip(model.param_names, model.params_, model.bounds)):
        lo, hi = np.percentile(draws[:, i], [2.5, 97.5]) if len(draws) else (np.nan, np.nan)
        on_bound = any(b is not None and np.isclose(value, b, rtol=1e-3, atol=1e-9) for b in bounds)
        records.append({"parameter": name, "value": value, "ci_lo": lo, "ci_hi": hi,
                        "n_boot": n_boot, "n_success": len(draws), "at_bound": on_bound, "model": model.name})
    return pd.DataFrame(records)


def bootstrap_arms(arms: Arms, tier1_df: pd.DataFrame, path: Path, n_boot: int = 200, n_jobs: int = -1) -> pd.DataFrame:
    """Every arm of the map's cells (status ok), each bootstrapped once and appended to `path` as it finishes."""
    done = set(map(tuple, read_bootstrap(path)[ARM_KEYS].values)) if path.exists() else set()
    keys = []
    for _, cell in tier1_df[tier1_df["status"] == "ok"].iterrows():
        contrast = arms.contrasts[cell["line"]]
        keys += [(*arm, cell["behavior"]) for arm in (contrast.reference, contrast.test)]
    todo = [k for k in dict.fromkeys(keys) if k not in done]
    for key in tqdm(todo, desc="Bootstrapping arms"):
        try:
            model, dataset = arms.model(*key), arms.dataset(*key)
        except Exception as e:
            tqdm.write(f"[{'/'.join(key)}] fit failed, skipped: {e}")
            continue
        seed = zlib.crc32("/".join(key).encode())
        result = bootstrap_arm(arms.process_factories[key[2]], model, dataset, n_boot, seed, n_jobs, desc="/".join(key))
        result.insert(0, "line", key[0]); result.insert(1, "condition", key[1]); result.insert(2, "behavior", key[2])
        result.to_csv(path, mode="a", header=not path.exists(), index=False)
    return read_bootstrap(path)


def read_bootstrap(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"line": str, "condition": str})


def save_contrasts(contrasts: List[Contrast], path: Path) -> None:
    pd.DataFrame([{"name": c.name, "family": c.family, "reference_line": c.reference[0], "reference_condition": c.reference[1],
                   "test_line": c.test[0], "test_condition": c.test[1]} for c in contrasts]).to_csv(path, index=False)


def load_contrasts(path: Path) -> List[Contrast]:
    return [Contrast(r.name, r.family, (r.reference_line, r.reference_condition), (r.test_line, r.test_condition))
            for r in pd.read_csv(path, dtype=str).itertuples()]


# ---------------------------------------------------------------------------
# Dumbbell figure
# ---------------------------------------------------------------------------

def axis_label(parameter: str, model_name: str) -> str:
    if parameter == "D":
        return "D: away / towards odds (log)" if "SidedSurvival" in model_name else "D: ipsi / contra baseline (log)"
    unit = {"B": "Hz", "H": "1/s"}.get(parameter, "")
    return f"{parameter} ({unit}, log)" if unit else f"{parameter} (log)"


def _limits(values: np.ndarray, intervals: np.ndarray) -> Tuple[float, float]:
    """Log-axis limits around the points, stretched by the intervals up to twice the points' own range."""
    lo, hi = np.log(np.nanmin(values)), np.log(np.nanmax(values))
    span = max(hi - lo, np.log(1.5))
    lo = max(min(lo, np.log(np.nanmin(intervals))), lo - 2 * span) if np.isfinite(intervals).any() else lo
    hi = min(max(hi, np.log(np.nanmax(intervals))), hi + 2 * span) if np.isfinite(intervals).any() else hi
    pad = 0.06 * (hi - lo)
    return float(np.exp(lo - pad)), float(np.exp(hi + pad))


def _whisker(ax: plt.Axes, arm: dict, y: float, color: str, limits: Tuple[float, float]) -> None:
    """95% interval clipped to the axis, with an arrow where it leaves it; none on a bound."""
    if arm["at_bound"] or not np.isfinite(arm["ci_lo"]):
        return
    lo, hi = max(arm["ci_lo"], limits[0]), min(arm["ci_hi"], limits[1])
    style = "-" if arm["n_success"] >= MIN_SUCCESS * arm["n_boot"] else (0, (2, 1.5))
    ax.plot([lo, hi], [y, y], color=color, lw=1, ls=style, zorder=2)
    for clipped, x, marker in ((arm["ci_lo"] < limits[0], lo, "<"), (arm["ci_hi"] > limits[1], hi, ">")):
        if clipped:
            ax.plot(x, y, marker=marker, color=color, ms=4, zorder=2)


def _point(ax: plt.Axes, arm: dict, y: float, control: bool, color: str) -> None:
    marker = "D" if arm["at_bound"] else "o"
    if control or arm["at_bound"]:
        ax.scatter([arm["value"]], [y], s=28, marker=marker, color="white", edgecolor=color, lw=1.3, zorder=3)
    else:
        ax.scatter([arm["value"]], [y], s=30, marker=marker, color=color, edgecolor="white", lw=0.8, zorder=3)


def _sections(rows: List[str], contrasts: Dict[str, Contrast]) -> List[Tuple[str, List[Tuple[float, str]]]]:
    """(family, [(y, line)]) top to bottom, a gap between families."""
    out, y = [], 0.0
    for family in dict.fromkeys(contrasts[r].family for r in rows):
        lines = [r for r in rows if contrasts[r].family == family]
        out.append((family, [(y - i, line) for i, line in enumerate(lines)]))
        y -= len(lines) + 0.8
    return out


def plot_absolute(behavior: str, boot_df: pd.DataFrame, contrasts: List[Contrast], tier1_df: pd.DataFrame,
                  rows: List[str], title: Optional[str] = None) -> Optional[plt.Figure]:
    """One panel per free parameter of `behavior`; rows are lines, each test arm joined to its own control."""
    sub = boot_df[boot_df["behavior"] == behavior]
    if sub.empty:
        return None
    by_contrast = {c.name: c for c in contrasts}
    arm = {(r["line"], r["condition"], r["parameter"]): r for r in sub.to_dict("records")}
    hits = set(tier1_df[(tier1_df["behavior"] == behavior) & tier1_df["significant"].fillna(False).astype(bool)]["line"])
    parameters = list(dict.fromkeys(sub["parameter"]))
    sections = _sections([r for r in rows if r in by_contrast], by_contrast)
    n_rows = sum(len(lines) for _, lines in sections)

    fig, axes = plt.subplots(1, len(parameters), figsize=(1.6 + 4.6 * len(parameters), 1.6 + 0.34 * n_rows),
                             sharey=True, squeeze=False, layout="constrained")
    for ax, parameter in zip(axes[0], parameters):
        get = lambda line, condition: arm.get((line, condition, parameter))
        present = sub[sub["parameter"] == parameter]
        positive = present[present["value"] > 0]
        limits = _limits(positive["value"].to_numpy(), positive[["ci_lo", "ci_hi"]].to_numpy().ravel())
        model_name = present["model"].iloc[0]

        for family, lines in sections:
            controls = {c.reference for c in (by_contrast[l] for _, l in lines)}
            control_values = [r["value"] for r in (get(*a) for a in controls) if r is not None and r["value"] > 0]
            if len(control_values) >= 3:
                q1, median, q3 = np.percentile(control_values, [25, 50, 75])
                top, bottom = lines[0][0] + 0.6, lines[-1][0] - 0.4
                ax.fill_betweenx([bottom, top], q1, q3, color=BAND, lw=0, zorder=0)
                ax.plot([median, median], [bottom, top], color=MEDIAN, lw=1, zorder=0)
            for y, line in lines:
                c = by_contrast[line]
                ref, test = get(*c.reference), get(*c.test)
                if ref is None or test is None or min(ref["value"], test["value"]) <= 0:
                    continue
                hit = line in hits
                color = (UP if test["value"] > ref["value"] else DOWN) if hit else NEUTRAL
                ax.plot([ref["value"], test["value"]], [y, y], color=color, lw=2.2 if hit else 1.2, zorder=1,
                        solid_capstyle="round")
                _whisker(ax, ref, y + 0.18, CONTROL, limits)
                _whisker(ax, test, y - 0.18, color if hit else TEST, limits)
                _point(ax, ref, y + 0.18, True, CONTROL)
                _point(ax, test, y - 0.18, False, color if hit else TEST)
        for (_, lines), (_, next_lines) in zip(sections, sections[1:]):
            ax.axhline((lines[-1][0] + next_lines[0][0]) / 2, color="#e0e0e0", lw=0.8)
        if parameter == "D":
            ax.axvline(1.0, color="#9e9e9e", lw=0.8, ls=":", zorder=0)

        ax.set_xscale("log")
        ax.set_xlim(*limits)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if f"{v:.0e}"[0] in "25" else ""))
        ax.tick_params(axis="x", which="both", labelsize=8)
        ax.set_xlabel(axis_label(parameter, model_name), fontsize=9)
        ax.set_title(f"{PARAMETER_LABELS.get(parameter, parameter)} ({parameter})", fontsize=10, loc="left", fontweight="bold")
        ax.grid(axis="x", which="both", color="#f0f0f0", lw=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0)

    ticks = [(y, line) for _, lines in sections for y, line in lines]
    axes[0][0].set_yticks([y for y, _ in ticks], [line for _, line in ticks], fontsize=8)
    axes[0][0].set_ylim(ticks[-1][0] - 0.8, 0.8)
    fig.suptitle(f"{title or behavior}: absolute values per arm (NTR: vehicle → ronidazole; mutants: WT background → mutant)",
                 fontsize=11, x=0.01, ha="left")
    handles = [
        Line2D([], [], marker="o", ls="", mfc="white", mec=CONTROL, label="control arm"),
        Line2D([], [], marker="o", ls="", color=TEST, label="test arm"),
        Line2D([], [], color=CONTROL, lw=1, label="95% fish-bootstrap interval"),
        Line2D([], [], color=CONTROL, lw=1, ls=(0, (2, 1.5)), label=f"interval, < {MIN_SUCCESS:.0%} refits succeeded"),
        Line2D([], [], marker="D", ls="", mfc="white", mec=CONTROL, label="on a parameter bound (no interval)"),
        Line2D([], [], color=UP, lw=2.2, label="hit, increase"),
        Line2D([], [], color=DOWN, lw=2.2, label="hit, decrease"),
        Line2D([], [], color=MEDIAN, lw=1, label="median of control arms"),
        Patch(color=BAND, label="middle 50% of control arms"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, fontsize=8, frameon=False)
    return fig


def plot_all(boot_df: pd.DataFrame, contrasts: List[Contrast], tier1_df: pd.DataFrame, output_dir: Path) -> None:
    rows = map_rows(tier1_df, contrasts)
    for behavior in [b for b in MAP_COLUMNS if b in set(boot_df["behavior"])]:
        fig = plot_absolute(behavior, boot_df, contrasts, tier1_df, rows, MAP_COLUMNS[behavior])
        save_fig(fig, output_dir, behavior)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Absolute Tier 1 parameters per arm with fish-bootstrap intervals.")
    parser.add_argument("--results", type=Path, default=Path("./tier1_results"), help="run_screen output folder")
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--plot-only", action="store_true", help="redraw from arm_bootstrap.csv without refitting")
    args = parser.parse_args()

    tier1_df = pd.read_csv(args.results / "tier1_results.csv", dtype={"line": str})
    if not args.plot_only:
        from .screen_setup import arms  # loads the event tables
        save_contrasts(list(arms.contrasts.values()), args.results / "contrasts.csv")
        bootstrap_arms(arms, tier1_df, args.results / "arm_bootstrap.csv", args.n_boot, args.n_jobs)
    plot_all(read_bootstrap(args.results / "arm_bootstrap.csv"), load_contrasts(args.results / "contrasts.csv"),
             tier1_df, args.results / "tier1_absolute")


if __name__ == "__main__":
    main()
