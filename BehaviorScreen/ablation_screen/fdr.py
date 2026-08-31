import numpy as np
import pandas as pd


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Standard BH step-up procedure. NaNs pass through as NaN, excluded from
    the multiple-testing family (not treated as p=1)."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    q = np.full(n, np.nan)

    valid_idx = np.where(~np.isnan(pvals))[0]
    if len(valid_idx) == 0:
        return q

    order = valid_idx[np.argsort(pvals[valid_idx])]
    n_valid = len(order)
    ranked = pvals[order]
    raw_q = ranked * n_valid / np.arange(1, n_valid + 1)
    q_sorted = np.minimum.accumulate(raw_q[::-1])[::-1]
    q[order] = np.clip(q_sorted, 0, 1)
    return q


def add_fdr(df: pd.DataFrame, pval_col: str = "p_value", alpha: float = 0.05, prefix: str = "") -> pd.DataFrame:
    df = df.copy()
    if pval_col not in df.columns:
        raise KeyError(
            f"add_fdr: expected column '{pval_col}' not found in DataFrame with "
            f"columns {list(df.columns)} -- did the upstream tier actually produce rows?"
        )
    df[f"{prefix}q_value"] = benjamini_hochberg(df[pval_col].values)
    df[f"{prefix}significant"] = df[f"{prefix}q_value"] < alpha
    return df