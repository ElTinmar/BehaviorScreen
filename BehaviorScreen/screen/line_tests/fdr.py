import numpy as np
import pandas as pd

ALPHA = 0.05  # FDR level of every line test (q < ALPHA: significant)


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


def holm(pvals: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values. NaNs pass through as NaN, excluded from the family."""
    pvals = np.asarray(pvals, dtype=float)
    adjusted = np.full(len(pvals), np.nan)
    valid = np.flatnonzero(~np.isnan(pvals))
    order = valid[np.argsort(pvals[valid])]
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(pvals[order] * np.arange(len(order), 0, -1)))
    return adjusted


def add_fdr(df: pd.DataFrame, pval_col: str = "p_value", prefix: str = "") -> pd.DataFrame:
    df = df.copy()
    if pval_col not in df.columns:
        raise KeyError(
            f"add_fdr: expected column '{pval_col}' not found in DataFrame with "
            f"columns {list(df.columns)} -- did the upstream tier actually produce rows?"
        )
    df[f"{prefix}q_value"] = benjamini_hochberg(df[pval_col].values)
    df[f"{prefix}significant"] = df[f"{prefix}q_value"] < ALPHA
    return df

def add_fdr_per_behavior(df: pd.DataFrame, pval_col: str = "p_value", prefix: str = "") -> pd.DataFrame:
    """One FDR family per behavior, and per contrast family when there is a 'family' column."""
    keys = ["family", "behavior"] if "family" in df.columns else ["behavior"]
    if df.empty:
        return add_fdr(df, pval_col, prefix)
    parts = [add_fdr(group.copy(), pval_col, prefix) for _, group in df.groupby(keys, dropna=False)]
    return pd.concat(parts, ignore_index=True)
