"""--compare sets the hits of the model variants and of the model-free screen side by side; --status names the model-free run."""
from pathlib import Path

import pandas as pd
import pytest

from BehaviorScreen.screen.line_tests import compare_variants, status, submit
from BehaviorScreen.screen.paths import COUNT_RESULTS, line_tests_dir


def results(folder: Path, hits: dict) -> None:
    folder.mkdir(parents=True)
    rows = [{"family": "ntr", "line": line, "behavior": behavior, "status": "ok", "p_value": 0.001 if hit else 0.5,
             "q_value": 0.01 if hit else 0.6, "significant": hit} for (line, behavior), hit in hits.items()]
    pd.DataFrame(rows).to_csv(folder / "tier1_results.csv", index=False)


def test_compare_includes_the_model_free_screen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.chdir(tmp_path)
    results(line_tests_dir("best"), {("L1", "okr"): True, ("L2", "okr"): True, ("L1", "prey_capture_ipsi"): False})
    results(COUNT_RESULTS, {("L1", "okr"): True, ("L2", "okr"): False, ("L1", "prey_capture"): True})
    compare_variants.main()
    table = pd.read_csv("line_tests/variant_comparison.csv").set_index(["line", "behavior"])
    assert table.loc[("L1", "okr"), "agreement"] == "all" and table.loc[("L2", "okr"), "agreement"] == "best only"
    assert pd.isna(table.loc[("L1", "prey_capture"), "agreement"])  # not an experiment of the model screen
    status.main()
    assert "model-free (line_tests/model_free/): last run" in capsys.readouterr().out


def test_model_free_runs_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["test-lines", "--model-free", "okr"])
    with pytest.raises(SystemExit):
        submit.main()
