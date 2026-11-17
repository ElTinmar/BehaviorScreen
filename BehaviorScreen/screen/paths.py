"""Where the screen's analyses write by default, relative to the folder they run in (the repository, on Slurm)."""
from pathlib import Path

CONTROL_FITS = Path("control_fits")  # fit-controls: the control models
LINE_TESTS = Path("line_tests")  # test-lines: Tier 1 and 2 of each line against the controls
