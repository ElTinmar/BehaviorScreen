"""
How many fish Tier 1 needs per arm, and where the noise comes from.

A Tier 1 effect is the log2 change of a free amplitude between two arms of n
fish. Its variance splits into

    counting  2 sigma_c^2 share / n         few events per trial: more trials lower it
    fish      2 sigma_c^2 (1 - share) / n   real fish differences: only more fish lower it
    model     2 (sigma_m^2 - sigma_c^2) / n the fitted amplitude noisier than plain counts
    arm       2 v                           sibling arms differing beyond their SEs (day, plate,
                                            drug; or SEs too optimistic): a floor
    clutch    2 tau^2                       mutant design only (two clutches): a floor

sigma_c and its counting share: per-fish count noise in the vehicle arms at the
current trials (variance_budget.csv: CV^2 = a / trials + b, log2 by CV / ln 2).
sigma_m: per-fish noise of the fitted amplitude, the median over NTR vehicle
arms of its Hessian SE x sqrt(fish); when below sigma_c, the model does better
than counts and there is no model term. v: the arm variance making the NTR
sibling deltas over sqrt(delta_se^2 + 2 v) have a robust SD (MAD) of 1,
with a 95% interval from resampling the lines: about 25 lines pin it down
loosely, and at the screen alpha a small v already caps what is detectable.
tau: clutch_variability.csv.

Fish per arm to detect a fold change f either way (80% power, Wald):

    n = 2 z^2 sigma_m^2 / (log2(f)^2 - 2 z^2 (v [+ tau^2]))

inf when the floors alone exceed the effect. alpha: 0.05 / lines (NTR screen)
or 0.05 / mutants (mutant screen), the strongest hit surviving BH.

Run after run_screen, clutch_variability and variance_budget:
    python -m BehaviorScreen.ablation_screen.power ./tier1_results [fold changes, default 1.5 2 3]
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats

from BehaviorScreen.point_process.point_process.io import save_fig

from .tier1_omnibus import PARAMETER_LABELS

KEYS = ["behavior", "parameter"]
FOLD_CHANGES = (1.5, 2.0, 3.0)
COMPONENTS = {"counting": "#2a78d6", "fish": "#1baf7a", "model": "#eb6834", "arm": "#6250d6", "clutch": "#eda100"}
SHADES = {"now": "#9ec5f4", "3x": "#e3eefb"}
INK, MUTED = "#333333", "#8c8c8c"
MAX_FISH = 1000


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


def variance_components(noise: pd.DataFrame, budget: pd.DataFrame, clutch: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Each component's variance of a log2 effect between two arms at the current fish and trials."""
    df = noise.merge(budget[[*KEYS, "counting slope", "fish floor", "trials now"]], on=KEYS)
    df = df.merge(clutch[[*KEYS, "tau"]], on=KEYS, how="left") if clutch is not None else df.assign(tau=np.nan)
    counting = df["counting slope"].clip(lower=0) / df["trials now"]
    fish = df["fish floor"].clip(lower=0)
    sampled = np.minimum(df["sigma_m"] ** 2, (counting + fish) / np.log(2) ** 2)
    share = counting / (counting + fish)
    n = df["fish_now"]
    return df.assign(
        counting=2 * sampled * share / n, fish=2 * sampled * (1 - share) / n,
        model=2 * (df["sigma_m"] ** 2 - sampled) / n, arm=2 * df["arm_var"], clutch=2 * df["tau"].fillna(0) ** 2,
    )


def fish_table(components: pd.DataFrame, fold_changes: Sequence[float] = FOLD_CHANGES, power: float = 0.8) -> pd.DataFrame:
    """
    Fish per arm per (behavior, parameter), design and fold change, also at the
    ends of the arm variance interval, with the smallest fold detectable now.
    """
    c = components
    designs = {"NTR screen": (0.05 / c["lines"], 0.0), "mutant screen": (0.05 / c["mutants"], c["tau"] ** 2)}
    rows = []
    for design, (alpha, clutch) in designs.items():
        z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
        now = 2 ** (z * np.sqrt(2 * c["sigma_m"] ** 2 / c["fish_now"] + 2 * (c["arm_var"] + clutch)))
        needed = lambda fold, v: fish_needed(c["sigma_m"], np.log2(fold), alpha, power, np.sqrt(v + clutch))
        for fold in fold_changes:
            rows.append(c[[*KEYS, "fish_now"]].assign(
                design=design, fold_change=fold, detectable_now=now, fish_per_arm=needed(fold, c["arm_var"]),
                fish_low=needed(fold, c["arm_var_low"]), fish_high=needed(fold, c["arm_var_high"]),
            ))
    return pd.concat(rows, ignore_index=True)


def _fish_text(n: float) -> str:
    return "∞" if not np.isfinite(n) else f">{MAX_FISH}" if n > MAX_FISH else f"{n:.0f}"


def plot_power_panel(
    components: pd.DataFrame, fish: pd.DataFrame, order: List[str], labels: Optional[Dict[str, str]] = None
) -> plt.Figure:
    """
    Left: share of each noise source (NTR screen above, mutant screen below).
    Right: a table of fish per arm per design and fold change, shaded by
    whether the current fish number reaches it.
    """
    rank = {b: i for i, b in enumerate(order)}
    rows = components.assign(r=components["behavior"].map(rank).fillna(len(order)),
                             p=components["parameter"].map(["A", "A_peak", "H", "B"].index))
    rows = rows.sort_values(["r", "p"]).reset_index(drop=True)
    y = np.arange(len(rows))[::-1].astype(float)
    folds = sorted(fish["fold_change"].unique())
    designs = ("NTR screen", "mutant screen")
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 0.42 * len(rows) + 1.8), sharey=True,
                                      gridspec_kw={"width_ratios": [1.1, 1]})
    for offset, design, parts in ((0.2, designs[0], list(COMPONENTS)[:-1]), (-0.2, designs[1], list(COMPONENTS))):
        total = rows[parts].sum(axis=1)
        start = np.zeros(len(rows))
        for part in parts:
            width = (rows[part] / total).to_numpy()
            left.barh(y + offset, width, left=start, height=0.36, color=COMPONENTS[part], edgecolor="white", linewidth=1,
                      hatch="//" if part == "clutch" else None, label=part if offset < 0 else None)
            start += width
    cells = fish.merge(rows[KEYS].assign(y=y), on=KEYS)
    for (design, offset), g in cells.drop_duplicates([*KEYS, "design"]).groupby(["design", cells["design"].map({designs[0]: 0.2, designs[1]: -0.2})]):
        for _, r in g.iterrows():
            text = f"×{r['detectable_now']:.1f}" if r["detectable_now"] < 100 else ">×100"
            left.text(1.01, r["y"] + offset, text, fontsize=6, va="center", color=MUTED)

    columns = [(d, f) for d in designs for f in folds]
    x = {c: i + 1.3 * (c[0] == designs[1]) + 1 for i, c in enumerate(columns)}
    for _, r in rows.assign(y=y).iterrows():
        right.text(0, r["y"], f"{r['fish_now']:.0f}", ha="center", va="center", fontsize=7, color=MUTED)
    for _, r in cells.iterrows():
        n, now, cx = r["fish_per_arm"], r["fish_now"], x[(r["design"], r["fold_change"])]
        fill = SHADES["now"] if n <= now else SHADES["3x"] if n <= 3 * now else "white"
        right.add_patch(plt.Rectangle((cx - 0.47, r["y"] - 0.45), 0.94, 0.9, facecolor=fill, edgecolor="#e6e6e6", linewidth=0.6))
        right.text(cx, r["y"] + 0.12, _fish_text(n), ha="center", va="center", fontsize=7, color=INK if np.isfinite(n) else MUTED)
        span = f"{_fish_text(r['fish_low'])}–{_fish_text(r['fish_high'])}"
        if r["fish_low"] != r["fish_high"]:
            right.text(cx, r["y"] - 0.24, span, ha="center", va="center", fontsize=5, color=MUTED)
    top = y.max() + 0.6
    right.text(0, top, "fish\nnow", ha="center", va="bottom", fontsize=7, color=MUTED)
    for c, cx in x.items():
        right.text(cx, top, f"×{c[1]:g}", ha="center", va="bottom", fontsize=7, color=INK)
    for design in designs:
        centre = np.mean([cx for c, cx in x.items() if c[0] == design])
        right.text(centre, top + 0.8, design, ha="center", va="bottom", fontsize=8, color=INK, weight="bold")
    right.set_xlim(-0.6, max(x.values()) + 0.6)
    right.set_ylim(y.min() - 0.7, top + 1.6)
    right.axis("off")
    right.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=SHADES["now"]), plt.Rectangle((0, 0), 1, 1, facecolor=SHADES["3x"])],
                 labels=["reachable with the fish now", "within 3× the fish now"], fontsize=7, frameon=False,
                 ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.0))
    right.text(0.5, -0.035, "fish per arm to detect the fold change, either way (80% power)\n"
               "small: range over the 95% interval of the arm variance; ∞: the floors alone exceed the effect",
               transform=right.transAxes, ha="center", va="top", fontsize=6, color=MUTED)

    left.set_xlim(0, 1)
    left.set_xlabel("share of the variance of a log2 effect at current fish and trials\n"
                    "(upper bar: NTR screen; lower: mutant screen; grey: fold detectable now)", fontsize=7)
    names = labels or {}
    left.set_yticks(y, [f"{names.get(b, b)} · {PARAMETER_LABELS[p]} ({p})" for b, p in zip(rows["behavior"], rows["parameter"])], fontsize=7)
    left.legend(fontsize=7, frameon=False, ncol=5, loc="lower left", bbox_to_anchor=(0, 1.0))
    left.tick_params(labelsize=7)
    for spine in ("top", "right"):
        left.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def main(results_dir: Path, fold_changes: Sequence[float] = FOLD_CHANGES) -> None:
    from .tier1_config import MAP_COLUMNS

    tier1_df = pd.read_csv(results_dir / "tier1_results.csv")
    param_df = pd.read_csv(results_dir / "tier1_parameters.csv")
    budget = pd.read_csv(results_dir / "variance_budget" / "variance_budget.csv")
    clutch_path = results_dir / "clutch_variability" / "clutch_variability.csv"
    clutch = pd.read_csv(clutch_path) if clutch_path.exists() else None
    out = results_dir / "power"
    out.mkdir(exist_ok=True)
    components = variance_components(model_noise(param_df, tier1_df), budget, clutch)
    fish = fish_table(components, fold_changes)
    components.to_csv(out / "variance_components.csv", index=False)
    fish.to_csv(out / "fish_needed.csv", index=False)
    order = list(MAP_COLUMNS) + [b for b in dict.fromkeys(tier1_df["behavior"]) if b not in MAP_COLUMNS]
    save_fig(plot_power_panel(components, fish, order, MAP_COLUMNS), out, "power_panel")
    parts = list(COMPONENTS)
    shares = components.set_index(KEYS)[parts[:-1]]
    with pd.option_context("display.width", 250):
        print("Share of the NTR-screen variance at current fish and trials:")
        print(shares.div(shares.sum(axis=1), axis=0).round(2).to_string())
        table = fish.pivot_table(index=[*KEYS, "design"], columns="fold_change", values="fish_per_arm")
        print("\nFish per arm to detect each fold change, either way (80% power):")
        print(table.rename(columns=lambda f: f"x{f:g}").to_string())


if __name__ == "__main__":
    args = sys.argv[1:]
    main(Path(args[0] if args else "./tier1_results"), tuple(float(f) for f in args[1:]) or FOLD_CHANGES)
