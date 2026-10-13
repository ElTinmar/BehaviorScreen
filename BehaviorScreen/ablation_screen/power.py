"""
Power of the screen, both ways round: the smallest amplitude change each
behavior can detect with n fish per arm (sensitivity), and the fish per arm
needed to detect a given fold change (sample size).

Per-fish SD sigma of each (behavior, parameter): the robust spread (MAD)
across NTR lines of the log2 effects, each scaled to one fish per arm by
sqrt(1/n_ref + 1/n_test). Most lines have no effect in a given column, so
this is the noise of a typical cell as the screen sees it (fish, misfit,
clutch-to-clutch variation of the drug effect), insensitive to the few hits.
Then, two arms of n fish (Wald approximation of the test):

    MDE(n) = (z_{1 - alpha/2} + z_power) * sqrt(2 sigma^2 / n  [+ 2 tau^2 for mutants])

and, solved for n, n = 2 sigma^2 / ((effect / z)^2 [- 2 tau^2]).

alpha 0.05: one planned test (a follow-up). alpha 0.05 / m: the strongest
hit of a behavior surviving BH among its m tests (the screen; m counts the
NTR lines, or the mutants for the mutant screen). Mutants add
the between-clutch variance of clutch_variability.csv, a floor that more
fish cannot lower: below it no number of fish is enough (inf).

    python -m BehaviorScreen.ablation_screen.power ./tier1_results [fold changes, default 1.25 1.5 2 3]
"""
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from BehaviorScreen.point_process.point_process.io import save_fig

from .tier1_omnibus import PARAMETER_LABELS

KEYS = ["behavior", "parameter"]
FISH = np.array([6, 10, 15, 20, 30, 50])
FOLD_CHANGES = (1.25, 1.5, 2.0, 3.0)


def per_fish_sd(param_df: pd.DataFrame, tier1_df: pd.DataFrame) -> pd.DataFrame:
    """sigma per (behavior, parameter): MAD across NTR lines of the per-fish-scaled log2 effects."""
    # Sibling contrasts only: mutant cells carry clutch variance on top of fish noise.
    df = param_df[(param_df["family"] == "ntr") & param_df["parameter"].isin(PARAMETER_LABELS)].merge(
        tier1_df[["line", "behavior", "n_fish_ref", "n_fish_test"]], on=["line", "behavior"]
    )
    df["scaled"] = df["delta"] / np.sqrt(1 / df["n_fish_ref"] + 1 / df["n_fish_test"])
    df = df[np.isfinite(df["scaled"])]
    sigma = df.groupby(KEYS).agg(
        sigma=("scaled", lambda x: stats.median_abs_deviation(x, scale="normal")), n_lines=("line", "nunique")
    ).reset_index()
    mutants = tier1_df[(tier1_df["family"] != "ntr") & (tier1_df["status"] == "ok")].groupby("behavior")["line"].nunique()
    return sigma.assign(n_mutants=sigma["behavior"].map(mutants))


def mde(sigma: np.ndarray, n: np.ndarray, alpha: float, power: float = 0.8, tau: np.ndarray = 0.0) -> np.ndarray:
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return z * np.sqrt(2 * sigma**2 / n + 2 * tau**2)


def _designs(sigma_df: pd.DataFrame, clutch_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """One row per (behavior, parameter, design) with its alpha and clutch SD."""
    df = sigma_df.merge(clutch_df[[*KEYS, "tau"]], on=KEYS, how="left") if clutch_df is not None else sigma_df.assign(tau=np.nan)
    screen_alpha = 0.05 / df["n_lines"]
    return pd.concat([
        df.assign(design="planned", alpha=0.05, tau=0.0),
        df.assign(design="screen", alpha=screen_alpha, tau=0.0),
        df.assign(design="mutant screen", alpha=0.05 / df["n_mutants"]).dropna(subset=["tau", "alpha"]),
    ], ignore_index=True)


def power_table(
    sigma_df: pd.DataFrame, clutch_df: Optional[pd.DataFrame] = None, fish: np.ndarray = FISH, power: float = 0.8
) -> pd.DataFrame:
    """MDE (log2) per behavior, parameter, design (planned / screen / mutant screen) and fish per arm."""
    df = _designs(sigma_df, clutch_df).merge(pd.DataFrame({"fish_per_arm": fish}), how="cross")
    return df.assign(mde_log2=mde(df["sigma"], df["fish_per_arm"], df["alpha"], power, df["tau"]))


def fish_needed(
    sigma: np.ndarray, effect_log2: float, alpha: np.ndarray, power: float = 0.8, tau: np.ndarray = 0.0, whole: bool = True
) -> np.ndarray:
    """Fish per arm to detect `effect_log2` (rounded up if `whole`); inf when clutch variance alone is larger than the effect."""
    z = stats.norm.ppf(1 - np.asarray(alpha) / 2) + stats.norm.ppf(power)
    room = (effect_log2 / z) ** 2 - 2 * np.asarray(tau) ** 2
    n = 2 * np.asarray(sigma) ** 2 / np.where(room > 0, room, 1)
    return np.where(room > 0, np.ceil(n) if whole else n, np.inf)


def sample_size_table(
    sigma_df: pd.DataFrame, clutch_df: Optional[pd.DataFrame] = None, fold_changes=FOLD_CHANGES, power: float = 0.8
) -> pd.DataFrame:
    """Fish per arm per behavior, parameter, design and fold change (either direction)."""
    df = _designs(sigma_df, clutch_df).merge(pd.DataFrame({"fold_change": fold_changes}), how="cross")
    return df.assign(fish_per_arm=fish_needed(df["sigma"], np.log2(df["fold_change"]), df["alpha"], power, df["tau"]))


def plot_fish_needed(sizes: pd.DataFrame, behaviors: List[str], current_fish: float, design: str = "planned") -> plt.Figure:
    """Fish per arm against the fold change to detect, one line per behavior and parameter."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sub = sizes[sizes["design"] == design]
    columns = [(b, p) for b in behaviors for p in PARAMETER_LABELS if ((sub["behavior"] == b) & (sub["parameter"] == p)).any()]
    folds = np.geomspace(1.15, 4, 60)
    for (b, p), color in zip(columns, plt.cm.viridis(np.linspace(0, 0.9, len(columns)))):
        row = sub[(sub["behavior"] == b) & (sub["parameter"] == p)].iloc[0]
        ax.plot(folds, fish_needed(row["sigma"], np.log2(folds), row["alpha"], tau=row["tau"], whole=False), color=color, linewidth=1,
                label=f"{b} {PARAMETER_LABELS[p]}")
    ax.axhline(current_fish, color="#bdbdbd", linestyle="--", linewidth=0.8)
    ax.text(4, current_fish, "median now ", fontsize=7, ha="right", va="bottom", color="#8c8c8c")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylim(3, 1000)
    ax.set_xticks([1.25, 1.5, 2, 3, 4], ["×1.25", "×1.5", "×2", "×3", "×4"])
    ax.set_xlabel("fold change to detect (either direction)")
    ax.set_ylabel("fish per arm (80% power)")
    ax.set_title(f"Fish needed: {design} design", fontsize=9)
    ax.legend(fontsize=5, frameon=False, ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def plot_mde(table: pd.DataFrame, behaviors: List[str], current_fish: float) -> plt.Figure:
    """MDE against fish per arm for the screen design, one line per behavior and parameter."""
    screen = table[table["design"] == "screen"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    columns = [(b, p) for b in behaviors for p in PARAMETER_LABELS if ((screen["behavior"] == b) & (screen["parameter"] == p)).any()]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(columns)))
    for (b, p), color in zip(columns, colors):
        cell = screen[(screen["behavior"] == b) & (screen["parameter"] == p)]
        ax.plot(cell["fish_per_arm"], 2 ** cell["mde_log2"], color=color, linewidth=1, label=f"{b} {PARAMETER_LABELS[p]}")
    ax.axvline(current_fish, color="#bdbdbd", linestyle="--", linewidth=0.8)
    ax.text(current_fish, ax.get_ylim()[1], " median now", fontsize=7, va="top", color="#8c8c8c")
    ax.set_yscale("log", base=2)
    ax.set_xlabel("fish per arm")
    ax.set_ylabel("smallest detectable fold change (80% power)")
    ax.set_title("Screen design: strongest hit surviving FDR", fontsize=9)
    ax.legend(fontsize=5, frameon=False, ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def main(results_dir: Path, fold_changes=FOLD_CHANGES) -> None:
    tier1_df = pd.read_csv(results_dir / "tier1_results.csv")
    param_df = pd.read_csv(results_dir / "tier1_parameters.csv")
    clutch_path = results_dir / "clutch_variability" / "clutch_variability.csv"
    clutch_df = pd.read_csv(clutch_path) if clutch_path.exists() else None
    out = results_dir / "power"
    out.mkdir(exist_ok=True)
    sigma_df = per_fish_sd(param_df, tier1_df)
    ok = tier1_df[(tier1_df["status"] == "ok") & (tier1_df["family"] == "ntr")]
    current = float(np.median(np.minimum(ok["n_fish_ref"], ok["n_fish_test"])))
    table = power_table(sigma_df, clutch_df, np.unique(np.append(FISH, current)))
    table.to_csv(out / "power_table.csv", index=False)
    sizes = sample_size_table(sigma_df, clutch_df, fold_changes)
    sizes.to_csv(out / "fish_needed.csv", index=False)
    behaviors = list(dict.fromkeys(tier1_df["behavior"]))
    save_fig(plot_mde(table, behaviors, current), out, "mde_vs_fish")
    for design in ("planned", "screen"):
        save_fig(plot_fish_needed(sizes, behaviors, current, design), out, f"fish_needed_{design.replace(' ', '_')}")
    now = table[table["fish_per_arm"] == current].pivot_table(index=KEYS, columns="design", values="mde_log2")
    print(f"Smallest detectable fold change with {current:g} fish per arm (80% power):")
    print((2 ** now).round(2).to_string())
    for design in ("planned", "screen", "mutant screen"):
        table_n = sizes[sizes["design"] == design].pivot_table(index=KEYS, columns="fold_change", values="fish_per_arm")
        print(f"\nFish per arm to detect each fold change, {design} design (80% power):")
        print(table_n.rename(columns=lambda f: f"x{f:g}").to_string())


if __name__ == "__main__":
    args = sys.argv[1:]
    main(Path(args[0] if args else "./tier1_results"), tuple(float(f) for f in args[1:]) or FOLD_CHANGES)
