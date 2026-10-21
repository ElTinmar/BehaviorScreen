"""
Power of Tier 1, from the model fits only: for each behavior and free
parameter, the power the screen has now to detect a fold change, and the fish
per arm it needs to reach nominal power (80%).

A Tier 1 effect is the log2 change of a free parameter between two arms of n
fish. Its variance:

    2 sigma_m^2 / n + 2 v [+ 2 tau^2]

sigma_m  per-fish noise of the fitted parameter: the median over NTR vehicle
         arms of its log2 SE x sqrt(fish). The SEs come from the fish bootstrap
         of arm_absolute when it has run (arm_bootstrap.csv), from the Hessian
         otherwise; the outputs say which.
v        arm variance: sibling arms differing beyond their SEs (day, plate,
         drug handling): the v making the NTR deltas over sqrt(delta_se^2 + 2 v)
         have a robust SD (MAD) of 1, with a 95% interval from resampling the
         lines. A floor more fish cannot lower.
tau      between-clutch SD (clutch_variability.csv): mutant design only, as a
         mutant and its reference are two clutches. A floor too.

Two-sided test at alpha = 0.05 / lines (NTR screen) or 0.05 / mutants (mutant
screen), about what the strongest hit of a column needs to survive BH:

    power(f) = Phi(d - z_a) + Phi(-d - z_a),  d = |log2 f| / SE(n fish now)
    n(f)     = 2 (z_a + z_0.8)^2 sigma_m^2 / (log2(f)^2 - 2 (z_a + z_0.8)^2 (v [+ tau^2]))

n is inf when the floors alone exceed the effect. A normal approximation of the
permutation test, one parameter at a time (Tier 1 tests a cell's free parameters
together): a guide to the design, not the exact power of the test.

Output (power/):
    power_summary.csv  one row per behavior and parameter: power now and fish per arm
                       for 80% power at each fold change and design, fold detectable now
    power_detail.csv   the same in long form, with the fish range over the 95% interval of v
    power_now.png      power against fold change with the fish per arm now, one panel per column
    sensitivity.png    per column, the fold change detectable now (80% power, interval over v) and the
                       variance budget of a contrast: fish noise, arm floor, clutch floor (sensitivity.csv)
    fish_needed.png    fish per arm for 80% power against fold change, one panel per column

Run after run_screen, arm_absolute (optional: bootstrap SEs) and clutch_variability
(optional: mutant design):
    python -m BehaviorScreen.ablation_screen.power ./tier1_results [fold changes, default 1.5 2 3]
"""
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats

from BehaviorScreen.point_process.point_process.io import save_fig

from .arm_absolute import with_bootstrap_se
from .tier1_omnibus import MIN_FOLD, PARAMETER_LABELS

KEYS = ["behavior", "parameter"]
FOLD_CHANGES = (1.5, 2.0, 3.0)
POWER = 0.8
MUTED = "#8c8c8c"
MAX_FISH = 1000
DESIGNS = {"NTR screen": ("#2a78d6", "-"), "mutant screen": ("#eb6834", "--")}
BUDGET = {"fish": ("#1baf7a", "fish noise, 2 σm² / n"), "arm": ("#eda100", "arm floor, 2 v"),
          "clutch": ("#6250d6", "clutch floor, 2 τ² (mutant screen)")}
MAX_FOLD = 8.0


def per_fish_sd(param_df: pd.DataFrame, tier1_df: pd.DataFrame) -> pd.DataFrame:
    """sigma per (behavior, parameter): MAD across NTR lines of the per-fish-scaled log2 effects (all noise, lumped)."""
    df = param_df[(param_df["family"] == "ntr") & param_df["parameter"].isin(PARAMETER_LABELS)].merge(
        tier1_df[["line", "behavior", "n_fish_ref", "n_fish_test"]], on=["line", "behavior"]
    )
    df["scaled"] = df["delta"] / np.sqrt(1 / df["n_fish_ref"] + 1 / df["n_fish_test"])
    df = df[np.isfinite(df["scaled"])]
    return df.groupby(KEYS).agg(
        sigma=("scaled", lambda x: stats.median_abs_deviation(x, scale="normal")), n_lines=("line", "nunique")
    ).reset_index()


def fish_needed(
    sigma: np.ndarray, effect_log2: float, alpha: np.ndarray, power: float = 0.8, tau: np.ndarray = 0.0, whole: bool = True
) -> np.ndarray:
    """Fish per arm to detect `effect_log2` (rounded up if `whole`); inf when the per-arm floor tau alone is larger than the effect."""
    z = stats.norm.ppf(1 - np.asarray(alpha) / 2) + stats.norm.ppf(power)
    room = (effect_log2 / z) ** 2 - 2 * np.asarray(tau) ** 2
    n = 2 * np.asarray(sigma) ** 2 / np.where(room > 0, room, 1)
    return np.where(room > 0, np.ceil(n) if whole else n, np.inf)


def _arm_variance(delta: np.ndarray, delta_se: np.ndarray) -> float:
    """v >= 0 such that delta / sqrt(delta_se^2 + 2 v) has a MAD-based SD of 1."""
    spread = lambda v: stats.median_abs_deviation(delta / np.sqrt(delta_se**2 + 2 * v), scale="normal") - 1
    if spread(0.0) <= 0:
        return 0.0
    return optimize.brentq(spread, 0.0, 2 * stats.median_abs_deviation(delta, scale="normal") ** 2)


def model_noise(param_df: pd.DataFrame, tier1_df: pd.DataFrame, min_lines: int = 5, n_boot: int = 200) -> pd.DataFrame:
    """
    Per (behavior, parameter): sigma_m, arm variance v with its 95% interval
    over resampled lines, current fish per arm, NTR lines and mutants tested.
    """
    rng = np.random.default_rng(0)
    ntr = param_df[(param_df["family"] == "ntr") & param_df["parameter"].isin(PARAMETER_LABELS)].merge(
        tier1_df[["line", "behavior", "n_fish_ref", "n_fish_test"]], on=["line", "behavior"]
    )
    ntr["sigma"] = ntr["se_reference"] / ntr["value_reference"] / np.log(2) * np.sqrt(ntr["n_fish_ref"])
    mutants = tier1_df[(tier1_df["family"] != "ntr") & (tier1_df["status"] == "ok")].groupby("behavior")["line"].nunique()
    records = []
    for (behavior, parameter), g in ntr.groupby(KEYS):
        sigma = g["sigma"][np.isfinite(g["sigma"]) & (g["sigma"] > 0)]
        pairs = g[np.isfinite(g["delta"]) & np.isfinite(g["delta_se"])]
        if min(len(sigma), len(pairs)) < min_lines:
            continue
        delta, delta_se = pairs["delta"].to_numpy(), pairs["delta_se"].to_numpy()
        boot = [_arm_variance(delta[i], delta_se[i]) for i in rng.integers(0, len(delta), (n_boot, len(delta)))]
        records.append({
            "behavior": behavior, "parameter": parameter, "sigma_m": sigma.median(),
            "arm_var": _arm_variance(delta, delta_se), "arm_var_low": np.percentile(boot, 2.5), "arm_var_high": np.percentile(boot, 97.5),
            "fish_now": float(np.median(np.minimum(g["n_fish_ref"], g["n_fish_test"]))),
            "lines": g["line"].nunique(), "mutants": mutants.get(behavior, np.nan),
        })
    return pd.DataFrame(records)


def with_clutch(noise: pd.DataFrame, clutch: Optional[pd.DataFrame]) -> pd.DataFrame:
    return noise.merge(clutch[[*KEYS, "tau"]], on=KEYS, how="left") if clutch is not None else noise.assign(tau=np.nan)


def design_terms(c: pd.Series, design: str) -> Tuple[float, float]:
    """(alpha, floor variance per arm) of a design for one column; NaN when the design does not apply."""
    if design == "NTR screen":
        return 0.05 / c["lines"], c["arm_var"]
    return 0.05 / c["mutants"], c["arm_var"] + c["tau"] ** 2


def power_at(sigma_m: float, n: float, effect_log2: np.ndarray, alpha: float, floor: float) -> np.ndarray:
    """Two-sided power to detect `effect_log2` with n fish per arm."""
    d = np.abs(effect_log2) / np.sqrt(2 * sigma_m**2 / n + 2 * floor)
    z = stats.norm.ppf(1 - alpha / 2)
    return stats.norm.cdf(d - z) + stats.norm.cdf(-d - z)


def detectable_now(sigma_m: float, n: float, alpha: float, floor: float, power: float = POWER) -> float:
    """Smallest fold change detected with `power` with n fish per arm."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return 2 ** (z * np.sqrt(2 * sigma_m**2 / n + 2 * floor))


def power_detail(noise: pd.DataFrame, fold_changes: Sequence[float] = FOLD_CHANGES, power: float = POWER) -> pd.DataFrame:
    """Per column, design and fold change: power now, fish per arm for `power` (and its range over the interval of v)."""
    records = []
    for _, c in noise.iterrows():
        for design in DESIGNS:
            alpha, floor = design_terms(c, design)
            if not (np.isfinite(alpha) and np.isfinite(floor)):
                continue
            extra = floor - c["arm_var"]
            for fold in fold_changes:
                needed = lambda v: float(fish_needed(c["sigma_m"], np.log2(fold), alpha, power, np.sqrt(v + extra)))
                records.append({
                    **c[[*KEYS, "fish_now", "sigma_m", "arm_var", "tau", "se_source"]].to_dict(), "design": design,
                    "fold_change": fold, "power_now": float(power_at(c["sigma_m"], c["fish_now"], np.log2(fold), alpha, floor)),
                    "fish_per_arm": needed(c["arm_var"]), "fish_low": needed(c["arm_var_low"]), "fish_high": needed(c["arm_var_high"]),
                    "detectable_now": detectable_now(c["sigma_m"], c["fish_now"], alpha, floor, power),
                })
    return pd.DataFrame(records)


def summary_table(detail: pd.DataFrame, order: List[str], labels: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """One row per (behavior, parameter) in map order: power now and fish for 80% power per fold change and design."""
    index = [*KEYS, "fish_now", "se_source"]
    power = detail.pivot_table(index=index, columns=["design", "fold_change"], values="power_now")
    power.columns = [f"power ×{fold:g} ({design})" for design, fold in power.columns]
    fish = detail.pivot_table(index=index, columns=["design", "fold_change"], values="fish_per_arm")
    fish.columns = [f"fish ×{fold:g} ({design})" for design, fold in fish.columns]
    now = detail.drop_duplicates([*KEYS, "design"]).pivot_table(index=KEYS, columns="design", values="detectable_now")
    now.columns = [f"detectable now ({design})" for design in now.columns]
    table = power.join(fish).reset_index().merge(now.reset_index(), on=KEYS)
    table.insert(0, "column", table["behavior"].map(lambda b: (labels or {}).get(b, b)) + " · " + table["parameter"].map(PARAMETER_LABELS))
    return _in_map_order(table, order)


def _in_map_order(df: pd.DataFrame, order: List[str]) -> pd.DataFrame:
    rank = {b: i for i, b in enumerate(order)}
    keys = np.lexsort((df["parameter"].map(list(PARAMETER_LABELS).index), df["behavior"].map(rank).fillna(len(order))))
    return df.iloc[keys].reset_index(drop=True)


def _panels(noise: pd.DataFrame, order: List[str], labels: Optional[Dict[str, str]], ncols: int,
            draw: Callable[[plt.Axes, pd.Series, np.ndarray], None]) -> plt.Figure:
    """One panel per (behavior, parameter) in map order, fold change on a shared x axis; `draw` fills each."""
    rows = _in_map_order(noise, order)
    folds = np.geomspace(1.2, 4, 80)
    nrows = -(-len(rows) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), sharex=True, sharey=True, squeeze=False,
                             layout="constrained")
    for ax, (_, c) in zip(axes.flat, rows.iterrows()):
        draw(ax, c, folds)
        ax.tick_params(labelbottom=True, labelsize=7)
        ax.set_title(f"{(labels or {}).get(c['behavior'], c['behavior'])} · {PARAMETER_LABELS[c['parameter']]} ({c['parameter']})",
                     fontsize=8, loc="left")
        ax.grid(which="major", color="#f0f0f0", lw=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    for ax in axes.flat[len(rows):]:
        ax.set_visible(False)
    for i in range(min(ncols, len(rows))):  # the lowest panel of each column
        axes.flat[i + ncols * ((len(rows) - 1 - i) // ncols)].set_xlabel("fold change to detect (either way)", fontsize=8)
    ax = axes.flat[0]
    ax.set_xscale("log", base=2)
    ax.set_xlim(1.2, 4)
    ax.set_xticks([1.25, 1.5, 2, 3, 4], ["×1.25", "×1.5", "×2", "×3", "×4"])
    ax.minorticks_off()
    return fig


def _legend(fig: plt.Figure, extra: plt.Line2D) -> None:
    handles = [plt.Line2D([], [], color=color, ls=style, lw=1.5, label=design) for design, (color, style) in DESIGNS.items()]
    fig.legend(handles=[*handles, extra], loc="outside lower center", ncol=3, fontsize=8, frameon=False)


def plot_power_now(noise: pd.DataFrame, order: List[str], labels: Optional[Dict[str, str]] = None, power: float = POWER,
                   ncols: int = 4) -> plt.Figure:
    """Power with the fish per arm now against the fold change, per design."""
    def draw(ax: plt.Axes, c: pd.Series, folds: np.ndarray) -> None:
        for design, (color, style) in DESIGNS.items():
            alpha, floor = design_terms(c, design)
            if np.isfinite(alpha) and np.isfinite(floor):
                ax.plot(folds, power_at(c["sigma_m"], c["fish_now"], np.log2(folds), alpha, floor), color=color, ls=style, lw=1.5)
        ax.axhline(power, color=MUTED, ls=":", lw=1)
        ax.text(3.9, 0.03, f"{c['fish_now']:.0f} fish per arm now", fontsize=7, color=MUTED, ha="right", va="bottom")

    fig = _panels(noise, order, labels, ncols, draw)
    fig.axes[0].set_ylim(0, 1)
    fig.supylabel("power with the fish per arm now", fontsize=9)
    _legend(fig, plt.Line2D([], [], color=MUTED, ls=":", label=f"nominal power {power:.0%}"))
    return fig


def plot_fish_needed(noise: pd.DataFrame, order: List[str], labels: Optional[Dict[str, str]] = None, power: float = POWER,
                     ncols: int = 4) -> plt.Figure:
    """Fish per arm for `power` against the fold change, per design; inf goes off the top."""
    def draw(ax: plt.Axes, c: pd.Series, folds: np.ndarray) -> None:
        for design, (color, style) in DESIGNS.items():
            alpha, floor = design_terms(c, design)
            if np.isfinite(alpha) and np.isfinite(floor):
                needed = fish_needed(c["sigma_m"], np.log2(folds), alpha, power, np.sqrt(floor), whole=False)
                ax.plot(folds, np.minimum(needed, 100 * MAX_FISH), color=color, ls=style, lw=1.5)
        ax.axhline(c["fish_now"], color=MUTED, ls=":", lw=1)
        ax.text(4, c["fish_now"], f"now {c['fish_now']:.0f} ", fontsize=7, color=MUTED, ha="right", va="bottom")

    fig = _panels(noise, order, labels, ncols, draw)
    ax = fig.axes[0]
    ax.set_yscale("log")
    ax.set_ylim(2, MAX_FISH)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    fig.supylabel(f"fish per arm for {power:.0%} power", fontsize=9)
    _legend(fig, plt.Line2D([], [], color=MUTED, ls=":", label="fish per arm now"))
    return fig


def sensitivity_table(noise: pd.DataFrame, order: List[str], labels: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Per column in map order: fold change detectable now per design (with its range over v) and the variance budget."""
    rows = _in_map_order(noise, order)
    table = rows[[*KEYS, "fish_now"]].copy()
    table.insert(0, "column", [f"{(labels or {}).get(b, b)} · {PARAMETER_LABELS[p]}" for b, p in zip(rows["behavior"], rows["parameter"])])
    for design in DESIGNS:
        terms = [design_terms(c, design) for _, c in rows.iterrows()]
        alpha, floor = np.array(terms).T
        extra = floor - rows["arm_var"].to_numpy()
        for suffix, v in (("", rows["arm_var"]), (" low", rows["arm_var_low"]), (" high", rows["arm_var_high"])):
            table[f"detectable{suffix} ({design})"] = detectable_now(rows["sigma_m"].to_numpy(), rows["fish_now"].to_numpy(), alpha, v.to_numpy() + extra)
    table["var fish"] = 2 * rows["sigma_m"].to_numpy() ** 2 / rows["fish_now"].to_numpy()
    table["var arm"] = 2 * rows["arm_var"].to_numpy()
    table["var clutch"] = 2 * rows["tau"].to_numpy() ** 2
    return table


def plot_sensitivity(table: pd.DataFrame, min_fold: float = MIN_FOLD, power: float = POWER) -> plt.Figure:
    """A: fold change detectable now per design, interval over v, threshold of a hit. B: variance budget of a contrast."""
    y = np.arange(len(table))[::-1]
    fig, (ax_a, ax_b) = plt.subplots(1, 2, sharey=True, figsize=(10, 0.32 * len(table) + 1.6), layout="constrained",
                                     gridspec_kw={"width_ratios": [1.5, 1]})
    for offset, (design, (color, _)) in zip((0.15, -0.15), DESIGNS.items()):
        x, low, high = (np.minimum(table[f"detectable{s} ({design})"], MAX_FOLD) for s in ("", " low", " high"))
        shown = np.isfinite(table[f"detectable ({design})"])
        ax_a.hlines(y[shown] + offset, low[shown], high[shown], color=color, lw=2)
        off = shown & (table[f"detectable ({design})"] >= MAX_FOLD)
        ax_a.plot(x[shown & ~off], y[shown & ~off] + offset, "o", ms=6, color=color, mec="white", mew=1, label=design)
        ax_a.plot(x[off], y[off] + offset, ">", ms=7, color=color, mec="white", mew=1)
    ax_a.axvline(min_fold, color=MUTED, ls="--", lw=1)
    ax_a.text(min_fold, y[0] + 0.8, f" hit threshold ×{min_fold:g}", fontsize=7, color=MUTED, va="bottom")
    ax_a.set_xscale("log", base=2)
    ax_a.set_xlim(1.2, MAX_FOLD * 1.1)
    ax_a.set_xticks([1.25, 1.5, 2, 3, 4, 8], ["×1.25", "×1.5", "×2", "×3", "×4", f"≥×{MAX_FOLD:g}"])
    ax_a.minorticks_off()
    ax_a.set_xlabel(f"fold change detectable with {power:.0%} power, fish per arm now (bar: 95% interval of v)", fontsize=8)
    ax_a.set_yticks(y, [f"{c}  (n={n:.0f})" for c, n in zip(table["column"], table["fish_now"])], fontsize=8)
    ax_a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=7, frameon=False)

    left = np.zeros(len(table))
    for key, (color, label) in BUDGET.items():
        width = np.nan_to_num(table[f"var {key}"].to_numpy())
        ax_b.barh(y, width, left=left, height=0.6, color=color, edgecolor="white", linewidth=1, label=label)
        left += width
    ax_b.set_xlabel("variance of a log2 contrast between two arms", fontsize=8)
    ax_b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=1, fontsize=7, frameon=False)
    for ax, letter in ((ax_a, "A"), (ax_b, "B")):
        ax.text(0, 1.01, letter, transform=ax.transAxes, fontweight="bold", va="bottom")
        ax.grid(axis="x", color="#f0f0f0", lw=0.6)
        ax.tick_params(labelsize=7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    return fig


def main(results_dir: Path, fold_changes: Sequence[float] = FOLD_CHANGES) -> None:
    from .tier1_config import MAP_COLUMNS

    tier1_df = pd.read_csv(results_dir / "tier1_results.csv", dtype={"line": str})
    param_df, se_source = with_bootstrap_se(pd.read_csv(results_dir / "tier1_parameters.csv", dtype={"line": str}), results_dir)
    clutch_path = results_dir / "clutch_variability" / "clutch_variability.csv"
    clutch = pd.read_csv(clutch_path) if clutch_path.exists() else None
    out = results_dir / "power"
    out.mkdir(exist_ok=True)

    noise = with_clutch(model_noise(param_df, tier1_df), clutch).assign(se_source=se_source)
    order = list(MAP_COLUMNS) + [b for b in dict.fromkeys(tier1_df["behavior"]) if b not in MAP_COLUMNS]
    detail = power_detail(noise, fold_changes)
    summary = summary_table(detail, order, MAP_COLUMNS)
    detail.to_csv(out / "power_detail.csv", index=False)
    summary.to_csv(out / "power_summary.csv", index=False)
    save_fig(plot_power_now(noise, order, MAP_COLUMNS), out, "power_now")
    save_fig(plot_fish_needed(noise, order, MAP_COLUMNS), out, "fish_needed")
    sensitivity = sensitivity_table(noise, order, MAP_COLUMNS)
    sensitivity.to_csv(out / "sensitivity.csv", index=False)
    save_fig(plot_sensitivity(sensitivity), out, "sensitivity")
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(f"Tier 1 power (SEs: {se_source}; mutant screen {'with' if clutch is not None else 'skipped: no'} clutch variability;"
              " inf: the floors alone exceed the effect)")
        print(summary.drop(columns=[*KEYS, "se_source"]).round(2).to_string(index=False))


if __name__ == "__main__":
    args = sys.argv[1:]
    main(Path(args[0] if args else "./tier1_results"), tuple(float(f) for f in args[1:]) or FOLD_CHANGES)
