"""
Where the screen's analyses read and write by default: the merged tables in this machine's data_root (machines.yaml),
the results relative to the folder they run in (the repository, on Slurm).
"""
import argparse
from pathlib import Path
from typing import Optional

from BehaviorScreen.jobs import this_machine

CONTROL_FITS = Path("control_fits")  # fit-controls: the control models
LINE_TESTS = Path("line_tests")  # test-lines: Tier 1 and 2 of each line against the controls
COUNT_RESULTS = Path("count_results")  # count_readouts: the model-free Tier 1
PLATE_VARIANCE = Path("plate_variance")
FISH_GAIN_VARIANCE = Path("fish_gain_variance")
EXCLUSIONS = Path("exclusions.csv")
SAMPLE_SIZES = Path("line_sample_sizes.png")
PREY_JTURN_PHASE = Path("prey_jturn_phase.png")


# The model of each behavior a Tier 1 run uses (tier1_config.variant_model): the best one (the screen's), the best one
# without event history, or the ladder's first rung (a constant rate or hazard, no fish gain): how much hits depend on it.
VARIANTS = ("best", "best-poisson", "base")
# What a line-test run saves per behavior (<variant folder>/<behavior>/<table>.csv) and merges across them
LINE_TEST_TABLES = ("tier1_results", "tier1_parameters", "tier2_results", "bootstrap_parameter_deltas")


def line_tests_dir(variant: str) -> Path:
    """Tier 1 results of a model variant: LINE_TESTS/best, LINE_TESTS/best_poisson, LINE_TESTS/base."""
    return LINE_TESTS / variant.replace("-", "_")


def add_data_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, help="Folder of the merged tables (default: this machine's data_root in machines.yaml).")


def data_root(path: Optional[Path]) -> Path:
    return path or this_machine().data_root
