"""
Tier 1 hits by model variant (paths.VARIANTS): one row per contrast and behavior with each variant's q-value and hit,
and how the variants agree, from the merged tables of each variant (test-lines --combine [--model ...]); variants not
merged yet are left out. The best model's run is the screen's; the others say how much its hits depend on the model.
    python -m BehaviorScreen.screen.line_tests.compare_variants
Writes line_tests/variant_comparison.csv and prints the agreement per behavior.
"""
import pandas as pd

from BehaviorScreen.screen.paths import LINE_TESTS, VARIANTS, line_tests_dir

KEYS = ["family", "line", "behavior"]


def agreement(row: pd.Series) -> str:
    """'all', 'none', or the variants calling the cell a hit ('best only', 'best + base', ...); NA where one did not test it."""
    if row.isna().any():
        return pd.NA
    hits = [variant for variant, hit in row.items() if hit]
    return "all" if len(hits) == len(row) else "none" if not hits else " + ".join(hits) + (" only" if len(hits) == 1 else "")


def main() -> None:
    paths = {v: line_tests_dir(v) / "tier1_results.csv" for v in VARIANTS}
    tables = {v: pd.read_csv(path, dtype={"line": str}) for v, path in paths.items() if path.exists()}
    if "best" not in tables or len(tables) < 2:
        raise SystemExit(f"needs the merged tables of the best model and of another variant; found {sorted(tables)}")
    merged = None
    for variant, table in tables.items():
        table = table.loc[table["status"] == "ok", KEYS + ["p_value", "q_value", "significant"]]
        table = table.rename(columns={c: f"{c}_{variant}" for c in ("p_value", "q_value", "significant")})
        merged = table if merged is None else merged.merge(table, on=KEYS, how="outer")
    hits = merged[[f"significant_{v}" for v in tables]].rename(columns=lambda c: c.removeprefix("significant_"))
    merged["agreement"] = hits.apply(agreement, axis=1)
    merged.to_csv(LINE_TESTS / "variant_comparison.csv", index=False)
    print(pd.crosstab(merged["behavior"], merged["agreement"].fillna("not run in all")).to_string())


if __name__ == "__main__":
    main()
