"""line_tests/tier1.html: each variant's maps with their cells, each cell linked to its arm surface grid."""
import json
from pathlib import Path

from BehaviorScreen.screen.line_tests.tier1_browser import variants, write_browser


def test_cells_link_their_grids_below_the_page(tmp_path: Path) -> None:
    best = tmp_path / "best"
    (best / "arm_surface_grids" / "all").mkdir(parents=True)
    (best / "arm_surface_grids" / "all" / "L1_spont.png").touch()
    (best / "tier1_map.png").touch()
    boxes = [{"line": line, "behavior": "spont", "parameter": "B", "box": [0, 0, 1, 1]} for line in ("L1", "L2")]
    (best / "tier1_map.json").write_text(json.dumps({"cells": boxes}))
    (tmp_path / "model_free").mkdir()
    found = variants(tmp_path)
    assert [v["name"] for v in found] == ["best"]
    cells = found[0]["cells"]
    assert [c["grid"] for c in cells] == ["best/arm_surface_grids/all/L1_spont.png", None]
    assert "best/tier1_map.png" in write_browser(tmp_path).read_text()
