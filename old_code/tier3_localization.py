"""
Tier 3: for lines/behaviors that pass Tier 2, identify WHICH kernel
parameter(s) drive the interaction term -- both a cheap Wald test (from the
existing Hessian machinery) and a fish-level cluster bootstrap (more
trustworthy near parameter boundaries).
"""
from typing import Dict
import numpy as np
import pandas as pd
from scipy.stats import norm

from ..ablation_screen.multi_group_process import MultiGroupProcess


def wald_test_covariate(model: MultiGroupProcess, covariate_name: str = "genotype:drug") -> pd.DataFrame:
    hessian = model.estimate_hessian(None)
    try:
        cov = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(hessian)
    se_all = np.sqrt(np.clip(np.diag(cov), 0, None))

    sl = model.beta_index_range(covariate_name)
    betas, se = model.params_[sl], se_all[sl]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, betas / se, np.nan)
    p = 2.0 * (1.0 - norm.cdf(np.abs(z)))

    return pd.DataFrame({
        "parameter": model.base_process.param_names,
        "beta_link_space": betas,
        "se_link_space": se,
        "z": z,
        "p_value": p,
    })


def bootstrap_interaction(
    model_full: MultiGroupProcess,
    covariate_name: str = "genotype:drug",
    n_boot: int = 200,
    seed: int = 0,
    n_jobs: int = -1,
    ci: float = 95.0,
) -> pd.DataFrame:
    boot_params = model_full.bootstrap_params(n_boot=n_boot, seed=seed, n_jobs=n_jobs)
    sl = model_full.beta_index_range(covariate_name)
    boot_betas = boot_params[:, sl]

    alpha = (100.0 - ci) / 2.0
    lo = np.percentile(boot_betas, alpha, axis=0)
    hi = np.percentile(boot_betas, 100 - alpha, axis=0)

    return pd.DataFrame({
        "parameter": model_full.base_process.param_names,
        "beta_boot_mean": boot_betas.mean(axis=0),
        f"ci_{alpha:.1f}%": lo,
        f"ci_{100-alpha:.1f}%": hi,
        "excludes_zero": (lo > 0) | (hi < 0),
    })


def group_param_table(model_full: MultiGroupProcess) -> pd.DataFrame:
    """Natural-space parameter values for each of the 4 groups, side by side --
    the human-readable companion to the link-space Wald/bootstrap tables."""
    rows = []
    for group in model_full.groups:
        rows.append({"group": group.name, **model_full.group_params(group.name)})
    return pd.DataFrame(rows)


def localize_effect(model_full: MultiGroupProcess, n_boot: int = 200, n_jobs: int = -1) -> Dict[str, pd.DataFrame]:
    return {
        "wald": wald_test_covariate(model_full, "genotype:drug"),
        "bootstrap": bootstrap_interaction(model_full, "genotype:drug", n_boot=n_boot, n_jobs=n_jobs),
        "group_params": group_param_table(model_full),
    }