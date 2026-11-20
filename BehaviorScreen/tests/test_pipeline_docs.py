"""docs/pipeline.md: its whole-pipeline diagram is stitched from its stage diagrams as they are now."""
import subprocess
import sys
from pathlib import Path

STITCH = Path(__file__).resolve().parents[2] / "docs" / "stitch_pipeline.py"


def test_whole_pipeline_diagram_is_up_to_date() -> None:
    result = subprocess.run([sys.executable, str(STITCH), "--check"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
