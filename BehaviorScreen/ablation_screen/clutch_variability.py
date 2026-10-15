"""
Between-clutch variability of the amplitudes against each line's effect.

The vehicle arm of every NTR line is one clutch recorded on one day, so the
spread of their log2 amplitudes (beyond their standard errors) estimates the
between-clutch SD tau of each (behavior, parameter); random effects,
DerSimonian-Laird. It also holds any effect of background or transgene on the
untreated fish, so tau is an upper bound. The standard errors come from the
fish bootstrap of arm_absolute when it has run, from the Hessian otherwise:
too small SEs inflate tau.

Each line's effect is then expressed in clutch SDs. NTR contrasts compare
siblings (one clutch), so their within-clutch test stands; a mutant and its
reference are two clutches, so 'z_clutch' adds 2 tau^2 to its variance.

The vehicle map shows each NTR line's untreated fish against the mean of all
vehicle arms, framing atypical controls: a parameter offset beyond
OUTLIER_Z x sqrt(tau^2 + SE^2), the spread expected of one more clutch.

    python -m BehaviorScreen.ablation_screen.clutch_variability ./tier1_results
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from BehaviorScreen.point_process.point_process.io import save_fig

from .arm_absolute import with_bootstrap_se
from .fdr import add_fdr
from .tier1_omnibus import MAP_COLORS, PARAMETER_LABELS, plot_tier1_map

KEYS = ["behavior", "parameter"]
ABOLISHED_LOG2 = 4.0  # effects below -4 log2 (1/16): response abolished, its size not meaningful
OUTLIER_Z = 3.0  # about 1 false frame per 400 cells


def _log2_amplitudes(param_df: pd.DataFrame) -> pd.DataFrame:
    """Vehicle arms of the NTR lines: log2 amplitude and its SE, one row per clutch and column."""
    df = param_df[(param_df["family"] == "ntr") & (param_df["value_reference"] > 0) & (param_df["se_reference"] > 0)]
    return df.assign(
        theta=np.log2(df["value_reference"]), s=df["se_reference"] / df["value_reference"] / np.log(2)
    )[["line", *KEYS, "theta", "s"]]


def _random_effects(theta: np.ndarray, s: np.ndarray) -> dict:
    w = 1 / s**2
    q = np.sum(w * (theta - np.sum(w * theta) / w.sum()) ** 2)
    k = len(theta)
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - np.sum(w**2) / w.sum()))
    w_re = 1 / (s**2 + tau2)
    return {
        "n_clutches": k, "mean_log2": np.sum(w_re * theta) / w_re.sum(), "tau": np.sqrt(tau2),
        "median_within_se": np.median(s), "i2": max(0.0, 1 - (k - 1) / q) if q > 0 else 0.0,
        "q_p_value": stats.chi2.sf(q, k - 1),
    }


def clutch_variability(param_df: pd.DataFrame, min_clutches: int = 5) -> pd.DataFrame:
    """
    Per (behavior, parameter): tau (log2 SD between clutches), the median
    within-clutch SE, I^2 (share of the spread due to clutches) and the
    p-value of Cochran's Q (no clutch effect).
    """
    records = [
        {"behavior": b, "parameter": p, **_random_effects(g["theta"].to_numpy(), g["s"].to_numpy())}
        for (b, p), g in _log2_amplitudes(param_df).groupby(KEYS) if len(g) >= min_clutches
    ]
    return pd.DataFrame(records)


def line_report(param_df: pd.DataFrame, clutch_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Per (line, behavior, parameter): the effect (delta, log2) in clutch SDs,
    how atypical the line's reference clutch is ('reference_offset_sd', NTR
    vehicles only) and, for mutants, the test with clutch variance
    (z_clutch, p_clutch, q_clutch per behavior). 'abolished': the test arm
    lost the response (delta < -ABOLISHED_LOG2).
    """
    df = param_df[param_df["parameter"].isin(PARAMETER_LABELS)].merge(clutch_df[[*KEYS, "mean_log2", "tau"]], on=KEYS)
    tau = df["tau"].where(df["tau"] > 0)  # no clutch spread detected: effects in SDs undefined
    df["effect_sd"] = df["delta"] / tau
    df["abolished"] = df["delta"] < -ABOLISHED_LOG2
    ntr = df["family"] == "ntr"
    df["reference_offset"] = np.where(ntr, np.log2(df["value_reference"]) - df["mean_log2"], np.nan)
    df["reference_offset_sd"] = df["reference_offset"] / tau
    df["z_clutch"] = df["delta"] / np.sqrt(df["delta_se"] ** 2 + np.where(ntr, 0, 2) * df["tau"] ** 2)
    df["p_clutch"] = np.where(ntr, np.nan, 2 * stats.norm.sf(np.abs(df["z_clutch"])))
    mutants = [add_fdr(g, "p_clutch", alpha, prefix="clutch_") for _, g in df[~ntr].groupby(KEYS)]
    return pd.concat([df[ntr], *mutants], ignore_index=True).rename(
        columns={"clutch_q_value": "q_clutch", "clutch_significant": "significant_clutch"}
    )


def line_summary(report: pd.DataFrame, tier1_df: pd.DataFrame) -> pd.DataFrame:
    """One row per line: typical effect size and atypical vehicle clutch, both in clutch SDs."""
    by_line = report.groupby("line")
    out = pd.DataFrame({
        "family": by_line["family"].first(),
        "columns": by_line.size(),
        "median_abs_effect_sd": by_line["effect_sd"].apply(lambda e: e.abs().median()),
        "columns_beyond_2sd": by_line["effect_sd"].apply(lambda e: int((e.abs() > 2).sum())),
        "columns_abolished": by_line["abolished"].sum(),
        "median_abs_reference_offset_sd": by_line["reference_offset_sd"].apply(lambda e: e.abs().median()),
        "tier1_hits": tier1_df[tier1_df["significant"]].groupby("line").size(),
        "hits_with_clutch_variance": report[report["significant_clutch"] == True].groupby("line")["behavior"].nunique(),
    })
    out.loc[out["family"] != "ntr", "hits_with_clutch_variance"] = out["hits_with_clutch_variance"].fillna(0)
    return out.fillna({"tier1_hits": 0}).sort_values(["family", "median_abs_effect_sd"], ascending=[False, False])


def plot_clutch_vs_effects(report: pd.DataFrame, clutch_df: pd.DataFrame, behaviors: List[str], ylim: float = ABOLISHED_LOG2) -> plt.Figure:
    """
    Per column: the NTR vehicle clutches around their mean (grey) and the
    effects of the NTR lines and mutants (log2), with the +-2 tau band.
    Points beyond +-ylim are drawn as triangles on the edge.
    """
    columns = [(b, p) for b in behaviors for p in PARAMETER_LABELS if ((clutch_df["behavior"] == b) & (clutch_df["parameter"] == p)).any()]
    tau = clutch_df.set_index(KEYS)["tau"]
    fig, ax = plt.subplots(figsize=(max(6.5, 0.45 * len(columns) + 1.5), 4.2))
    rng = np.random.default_rng(0)
    jitter = lambda n: rng.uniform(-0.12, 0.12, n)

    def dots(x: float, values: pd.Series, color: str, zorder: int) -> None:
        xs, ys = x + jitter(len(values)), values.to_numpy()
        inside = np.abs(ys) <= ylim
        ax.scatter(xs[inside], ys[inside], s=9, color=color, zorder=zorder)
        for sign, marker in ((1, "^"), (-1, "v")):
            out = sign * ys > ylim
            ax.scatter(xs[out], np.full(out.sum(), sign * ylim), s=14, marker=marker, color=color, zorder=zorder)

    for x, (b, p) in enumerate(columns):
        cell = report[(report["behavior"] == b) & (report["parameter"] == p)]
        ntr = cell[cell["family"] == "ntr"]
        ax.add_patch(plt.Rectangle((x - 0.4, -2 * tau[(b, p)]), 0.8, 4 * tau[(b, p)], color="#eeeeee", zorder=0))
        dots(x - 0.22, ntr["reference_offset"], "#8c8c8c", 2)
        for family, dx, color in (("ntr", 0.0, "#2166ac"), ("mutant", 0.22, "#d6604d")):
            dots(x + dx, cell.loc[cell["family"] == family, "delta"], color, 3)
    ax.axhline(0, color="#bdbdbd", linewidth=0.6)
    ax.set_xticks(range(len(columns)), [f"{b} {PARAMETER_LABELS[p]}" for b, p in columns], rotation=90, fontsize=7)
    ax.set_xlim(-0.6, len(columns) - 0.4)
    ax.set_ylim(-ylim - 0.3, ylim + 0.3)
    ax.set_ylabel(f"log2 (triangles: beyond ±{ylim:g})")
    ax.legend(handles=[
        plt.Line2D([], [], marker="o", color=c, linestyle="", markersize=4, label=l) for c, l in (
            ("#8c8c8c", "vehicle clutch vs clutch mean"), ("#2166ac", "NTR effect (drug vs vehicle)"), ("#d6604d", "mutant vs reference"))
    ] + [plt.Rectangle((0, 0), 1, 1, color="#eeeeee", label="±2 between-clutch SD")], fontsize=7, frameon=False, ncol=4,
        loc="upper center", bbox_to_anchor=(0.5, 1.12))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def plot_effect_sd_map(report: pd.DataFrame, tier1_df: pd.DataFrame, behaviors: List[str], rows: List[str]) -> Tuple[plt.Figure, plt.Axes]:
    """The Tier 1 map in clutch SDs; mutant frames come from the test with clutch variance."""
    effects = report.dropna(subset=["effect_sd"])
    effects = effects.assign(delta=effects["effect_sd"], delta_se=effects["delta_se"] / effects["tau"])
    clutch_hits = set(map(tuple, report.loc[report["significant_clutch"] == True, ["line", "behavior"]].to_numpy()))
    status = tier1_df.assign(significant=[
        sig if family == "ntr" else (line, b) in clutch_hits
        for line, b, family, sig in zip(tier1_df["line"], tier1_df["behavior"], tier1_df["family"], tier1_df["significant"])
    ])
    fig, ax = plot_tier1_map(effects, status, behaviors, rows, vmax=3.0, colorbar_label="effect / between-clutch SD")
    fig.suptitle("Effects in between-clutch SDs (mutant frames: q < 0.05 with clutch variance)", fontsize=9, y=1.0)
    return fig, ax


def plot_vehicle_map(
    report: pd.DataFrame, tier1_df: pd.DataFrame, behaviors: List[str], column_labels: Optional[Dict[str, str]] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Each NTR line's vehicle arm against the mean vehicle arm (log2); frame: an atypical control in that behavior."""
    v = report[report["family"] == "ntr"].dropna(subset=["reference_offset"])
    v = v.assign(delta=v["reference_offset"], delta_se=v["se_reference"] / v["value_reference"] / np.log(2))
    z = v["delta"] / np.sqrt(v["tau"] ** 2 + v["delta_se"].fillna(np.inf) ** 2)  # arms at a bound (no SE): never framed
    atypical = set(map(tuple, v.loc[np.abs(z) > OUTLIER_Z, ["line", "behavior"]].to_numpy()))
    status = tier1_df.loc[tier1_df["family"] == "ntr", ["line", "behavior", "family", "status"]].drop_duplicates(["line", "behavior"])
    status["significant"] = [(l, b) in atypical for l, b in zip(status["line"], status["behavior"])]
    rows = ["WT"] * ("WT" in set(v["line"])) + sorted(set(v["line"]) - {"WT"})
    fig, ax = plot_tier1_map(v, status, behaviors, rows, colorbar_label="log2(vehicle / mean vehicle)", column_labels=column_labels,
                             frame_label=f"atypical control (beyond {OUTLIER_Z:g} clutch SDs)")
    fig.suptitle("Untreated fish of each line against the mean vehicle arm", fontsize=9, y=1.0)
    return fig, ax


def main(results_dir: Path) -> None:
    from .tier1_config import MAP_COLUMNS

    tier1_df = pd.read_csv(results_dir / "tier1_results.csv", dtype={"line": str})
    param_df, se_source = with_bootstrap_se(pd.read_csv(results_dir / "tier1_parameters.csv", dtype={"line": str}), results_dir)
    print(f"SEs: {se_source}")
    out = results_dir / "clutch_variability"
    out.mkdir(exist_ok=True)
    behaviors = list(dict.fromkeys(param_df["behavior"]))
    clutch_df = clutch_variability(param_df).assign(se_source=se_source)
    report = line_report(param_df, clutch_df)
    clutch_df.to_csv(out / "clutch_variability.csv", index=False)
    report.to_csv(out / "line_report.csv", index=False)
    line_summary(report, tier1_df).to_csv(out / "line_summary.csv")
    report[report["abolished"]].to_csv(out / "abolished.csv", index=False)
    save_fig(plot_clutch_vs_effects(report, clutch_df, behaviors), out, "clutch_vs_effects")
    families = tier1_df.drop_duplicates("line").set_index("line")["family"]
    rows = [l for l in families.index if families[l] == "ntr"] + [l for l in families.index if families[l] != "ntr"]
    fig, _ = plot_effect_sd_map(report, tier1_df, behaviors, [r for r in rows if r in set(report["line"])])
    save_fig(fig, out, "effect_sd_map")
    order = [b for b in MAP_COLUMNS if b in behaviors] + [b for b in behaviors if b not in MAP_COLUMNS]
    fig, _ = plot_vehicle_map(report, tier1_df, order, MAP_COLUMNS)
    save_fig(fig, out, "vehicle_map")
    print(clutch_df.round(3).to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "./tier1_results"))
