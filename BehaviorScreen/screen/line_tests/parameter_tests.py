"""
Post hoc of a Tier 1 hit: which of its free amplitudes differ between the arms. Per parameter, the joint
fit of both arms with that parameter in common (the others each arm's own) against the one with none in
common (Tier 1's two fits, as one MultiGroupProcess with log links). The deviance is calibrated by datasets
simulated from the fit with the parameter in common: fish permutation would also erase the differences in
the other parameters, which this null keeps. p-values are Holm-corrected over the cell's parameters.
A cell with one free parameter has nothing to tell apart: Tier 1 is its test.
"""
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from BehaviorScreen.point_process.dataset import PointProcessDataset
from BehaviorScreen.point_process.parallel import parallel_map, report_failures
from BehaviorScreen.point_process.point_process import PointProcess
from BehaviorScreen.point_process.wrappers import PartiallyFixedProcess

from .dataset_utils import Arms
from .fdr import ALPHA, holm
from .gain_structure import GainStructure, fish_groups
from .multi_group_process import GroupSpec, MultiGroupProcess
from .param_links import LogLink

DESIGN = {"reference": {}, "test": {"test": 1.0}}
COLUMNS = ["line", "behavior", "parameter", "deviance", "p_value", "p_holm", "significant", "n_boot_success"]


def fit_arms(
    factory: Callable[[], PointProcess], datasets: Sequence[PointProcessDataset], shared: Optional[str] = None
) -> PartiallyFixedProcess:
    """Joint fit of the two arms (DESIGN order), amplitudes on the log scale; shared: a parameter they have in common."""
    base = factory()
    groups = [GroupSpec(name, ds, design) for (name, design), ds in zip(DESIGN.items(), datasets)]
    arms = MultiGroupProcess(base, groups, ["test"], links=[LogLink()] * len(base.param_names))
    return PartiallyFixedProcess(arms, {} if shared is None else {f"{shared}[test]": 0.0}).fit(None)


def shared_deviance(
    factory: Callable[[], PointProcess], datasets: Sequence[PointProcessDataset], parameter: str
) -> Tuple[float, PartiallyFixedProcess]:
    """2 x (none in common - parameter in common), and the fit with it in common (the null)."""
    null = fit_arms(factory, datasets, parameter)
    return max(0.0, 2.0 * (fit_arms(factory, datasets).log_likelihood - null.log_likelihood)), null


def _replicate(
    seed_seq: np.random.SeedSequence, factory: Callable[[], PointProcess], datasets: Sequence[PointProcessDataset],
    null_params: List[np.ndarray], parameter: str, groups: pd.DataFrame, structure: Optional[GainStructure],
) -> Union[float, str]:
    rng = np.random.default_rng(seed_seq)
    try:
        latent = [None, None] if structure is None else np.split(structure.latent(groups, rng), [datasets[0].num_fish])
        simulated = []
        for ds, params, z in zip(datasets, null_params, latent):
            model = factory()
            model.set_params(params)
            simulated.append(model.simulate_dataset(ds, rng, z))
        return shared_deviance(factory, simulated, parameter)[0]
    except Exception as exc:
        return repr(exc)


def parameter_tests(
    factory: Callable[[], PointProcess], datasets: Sequence[PointProcessDataset], structure: Optional[GainStructure] = None,
    *, n_boot: int, seed: int = 0, n_jobs: int = -1,
) -> pd.DataFrame:
    """
    One row per free parameter of factory's model: its deviance and p-value, and the p-value Holm-corrected over
    them. datasets: (reference, test). structure: the dependence between the simulated fish of a day and of a plate.
    """
    groups = fish_groups(np.concatenate([ds.fish_ids for ds in datasets]), "")
    rows = []
    for parameter in factory().param_names:
        deviance, null = shared_deviance(factory, datasets, parameter)
        arms = null.base_process
        null_params = [np.array([arms.group_params(g.name)[p] for p in arms.base_process.param_names]) for g in arms.groups]
        results = parallel_map(
            partial(_replicate, factory=factory, datasets=datasets, null_params=null_params, parameter=parameter, groups=groups, structure=structure),
            np.random.SeedSequence(seed).spawn(n_boot), n_jobs=n_jobs, desc=f"{parameter} in common",
        )
        report_failures(f"{parameter} in common", [r for r in results if isinstance(r, str)], n_boot)
        null_dist = np.array([r for r in results if not isinstance(r, str)])
        p = (np.sum(null_dist >= deviance) + 1) / (len(null_dist) + 1) if len(null_dist) >= 0.3 * n_boot else np.nan
        rows.append({"parameter": parameter, "deviance": deviance, "p_value": float(p), "n_boot_success": len(null_dist)})
    table = pd.DataFrame(rows)
    return table.assign(p_holm=holm(table["p_value"].to_numpy()), significant=lambda t: t["p_holm"] < ALPHA)


def run_parameter_tests(
    tier1_df: pd.DataFrame, arms: Arms, n_boot: int, structure: Optional[GainStructure] = None, n_jobs: int = -1
) -> pd.DataFrame:
    """parameter_tests of every significant Tier 1 cell of an NTR line whose model has several free parameters."""
    hits = tier1_df[(tier1_df["family"] == "ntr") & tier1_df["significant"]]
    tables = []
    for _, row in tqdm(hits.iterrows(), total=len(hits), desc="Parameter tests"):
        factory = arms.process_factory(row["line"], row["behavior"])
        if len(factory().param_names) > 1:
            table = parameter_tests(factory, arms.datasets(row["line"], row["behavior"]), structure, n_boot=n_boot, n_jobs=n_jobs)
            tables.append(table.assign(line=row["line"], behavior=row["behavior"]))
    return pd.concat(tables, ignore_index=True)[COLUMNS] if tables else pd.DataFrame(columns=COLUMNS)
