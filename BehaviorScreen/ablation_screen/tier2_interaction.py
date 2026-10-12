"""
Tier 2: is a line's drug effect on the amplitudes different from the drug's
effect on WT? Four arms (WT reference, WT drug, line vehicle, line drug),
amplitudes on the log scale:

    log amplitude = baseline + line + drug + line x drug

Null: no interaction; alternative: with it (saturated). Both are joint
fits (MultiGroupProcess, log link), so they share one parameter space. The deviance is
calibrated by a parametric bootstrap from the null fit: fish permutation
would break the line and drug main effects the null keeps.
"""
from functools import partial
from typing import Callable, Dict, List, Sequence, Tuple, Union

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from tqdm import tqdm

from BehaviorScreen.point_process.point_process.dataset import PointProcessDataset
from BehaviorScreen.point_process.point_process.parallel import parallel_map, report_failures
from BehaviorScreen.point_process.point_process.point_process import PointProcess

from .dataset_utils import Arms
from .multi_group_process import GroupSpec, MultiGroupProcess
from .param_links import LogLink
from .tier1_omnibus import plot_tier1_map

DESIGN = {
    "reference": {},
    "reference_drug": {"drug": 1.0},
    "vehicle": {"line": 1.0},
    "drug": {"line": 1.0, "drug": 1.0, "line_x_drug": 1.0},
}


def fit_design(
    factory: Callable[[], PointProcess], datasets: Sequence[PointProcessDataset], covariates: List[str]
) -> MultiGroupProcess:
    """Joint fit of the four arms (DESIGN order), amplitudes on the log scale."""
    base = factory()
    groups = [GroupSpec(name, ds, design) for (name, design), ds in zip(DESIGN.items(), datasets)]
    return MultiGroupProcess(base, groups, covariates, links=[LogLink()] * len(base.param_names)).fit()


def interaction_deviance(
    factory: Callable[[], PointProcess], datasets: Sequence[PointProcessDataset]
) -> Tuple[float, MultiGroupProcess]:
    """2 x (with - without the line x drug term), and the fit without it (the null)."""
    null = fit_design(factory, datasets, ["line", "drug"])
    full = fit_design(factory, datasets, ["line", "drug", "line_x_drug"])
    return max(0.0, 2.0 * (full.log_likelihood - null.log_likelihood)), null


def _replicate(
    seed_seq: np.random.SeedSequence,
    factory: Callable[[], PointProcess],
    datasets: Sequence[PointProcessDataset],
    null_params: List[np.ndarray],
) -> Union[float, str]:
    rng = np.random.default_rng(seed_seq)
    try:
        simulated = []
        for ds, params in zip(datasets, null_params):
            model = factory()
            model.set_params(params)
            simulated.append(model.simulate_dataset(ds, rng))
        return interaction_deviance(factory, simulated)[0]
    except Exception as exc:
        return repr(exc)


def tier2_interaction_test(
    factory: Callable[[], PointProcess],
    datasets: Sequence[PointProcessDataset],
    n_boot: int = 200,
    seed: int = 0,
    n_jobs: int = -1,
) -> Dict:
    deviance, null = interaction_deviance(factory, datasets)
    null_params = [np.array([null.group_params(g.name)[p] for p in null.base_process.param_names]) for g in null.groups]
    results = parallel_map(
        partial(_replicate, factory=factory, datasets=datasets, null_params=null_params),
        np.random.SeedSequence(seed).spawn(n_boot), n_jobs=n_jobs, desc="Interaction bootstrap",
    )
    report_failures("interaction bootstrap", [r for r in results if isinstance(r, str)], n_boot)
    null_dist = np.array([r for r in results if not isinstance(r, str)])
    p = (np.sum(null_dist >= deviance) + 1) / (len(null_dist) + 1) if len(null_dist) >= 0.3 * n_boot else np.nan
    return {"interaction_deviance": deviance, "p_value": float(p), "n_boot_success": len(null_dist)}


def run_tier2_screen(
    tier1_df: pd.DataFrame, arms: Arms, reference: str = "WT", n_boot: int = 200, n_jobs: int = -1
) -> pd.DataFrame:
    """
    Interaction test for every significant Tier 1 cell of an NTR line;
    `reference` is the contrast whose drug effect is the baseline (WT).
    """
    hits = tier1_df[(tier1_df["family"] == "ntr") & tier1_df["significant"] & (tier1_df["line"] != reference)]
    records = []
    for _, row in tqdm(hits.iterrows(), total=len(hits), desc="Tier 2 cells"):
        line, behavior = row["line"], row["behavior"]
        record = {"line": line, "behavior": behavior, "reference": reference}
        try:
            datasets = [*arms.datasets(reference, behavior), *arms.datasets(line, behavior)]
            if any(ds is None for ds in datasets):
                records.append({**record, "status": "skipped_reference_missing", "p_value": np.nan})
                continue
            result = tier2_interaction_test(arms.process_factories[behavior], datasets, n_boot=n_boot, n_jobs=n_jobs)
            records.append({**record, "status": "ok", **result})
        except Exception as exc:
            records.append({**record, "status": f"fit_failed: {type(exc).__name__}: {exc}", "p_value": np.nan})
    return pd.DataFrame(records, columns=["line", "behavior", "reference", "status", "interaction_deviance", "p_value", "n_boot_success"])


def interaction_effects(param_df: pd.DataFrame, reference: str = "WT") -> pd.DataFrame:
    """
    log2FC(line) - log2FC(reference) per (line, behavior, parameter) of the
    Tier 1 parameter table (compute_parameter_deltas), with its standard error.
    """
    ref = param_df[param_df["line"] == reference].set_index(["behavior", "parameter"])[["delta", "delta_se"]]
    lines = param_df[(param_df["family"] == "ntr") & (param_df["line"] != reference)]
    out = lines.join(ref, on=["behavior", "parameter"], rsuffix="_reference")
    out["interaction"] = out["delta"] - out["delta_reference"]
    out["interaction_se"] = np.hypot(out["delta_se"], out["delta_se_reference"])
    return out


def plot_tier2_map(
    param_df: pd.DataFrame, tier2_df: pd.DataFrame, behaviors: List[str], reference: str = "WT"
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Lines tested in Tier 2, coloured by log2FC(line) - log2FC(reference)
    (grey: same drug effect as the reference). Frames: Tier 2 significant;
    untested cells are coloured but never framed. The reference's own drug
    effect is its row of the Tier 1 map.
    """
    tested = tier2_df[tier2_df["status"] == "ok"]
    effects = interaction_effects(param_df, reference)
    effects = effects[effects["line"].isin(tested["line"])].assign(delta=lambda d: d["interaction"], delta_se=lambda d: d["interaction_se"])
    significant = set(zip(tested.loc[tested["significant"], "line"], tested.loc[tested["significant"], "behavior"]))
    status = effects.drop_duplicates(["line", "behavior"])[["line", "behavior", "family"]].assign(
        status="ok", reference=reference, significant=lambda d: [(l, b) in significant for l, b in zip(d["line"], d["behavior"])]
    )
    fig, ax = plot_tier1_map(
        effects, status, behaviors, list(dict.fromkeys(effects["line"])),
        colorbar_label=f"log2 fold change: line minus {reference}",
    )
    fig.suptitle(f"Drug effect beyond {reference} (reference: {reference} vehicle -> ronidazole)", fontsize=9, y=1.0)
    return fig, ax
