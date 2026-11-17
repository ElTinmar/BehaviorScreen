"""
The screen's configuration, shared by run_screen and the commands run after it (arm_absolute, count_readouts,
plate_variance, simulate_power): the merged event tables, Tier 1 models (each experiment's best control model with
every parameter pinned except tier1_config.AMPLITUDE_PARAMETERS), contrasts and the arms they compare.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from BehaviorScreen.screen.loader import BehavioralDataLoader
from BehaviorScreen.screen.control_models.model_config import EVENT_TABLES, MUTANT_REFERENCES, get_model_config
from BehaviorScreen.screen.line_tests.dataset_utils import Arms, Contrast, mutant_contrast, ntr_contrast
from BehaviorScreen.screen.line_names import warn_unnamed
from BehaviorScreen.screen.line_tests.tier1_config import MAP_COLUMNS, PINNED_AMPLITUDES, tier1_behaviors

# Tables merged across screen folders with merge_screen (adds line/condition columns)
EVENTS_CSV = {"bouts": "bouts_all.csv", "saccades": "saccades_augmented_all.csv"}


def load_tables(data_root: Path) -> Dict[str, BehavioralDataLoader]:
    """The merged event tables of data_root, with their trials and quality control."""
    return {events: BehavioralDataLoader(data_root / csv, [data_root / "valid_trials_all.csv"], [data_root / "qc_all.csv"], **EVENT_TABLES[events])
            for events, csv in EVENTS_CSV.items()}


def screen_contrasts(loaders: Dict[str, BehavioralDataLoader]) -> List[Contrast]:
    """Drug against vehicle within WT and each NTR line, and each mutant against its wild-type background."""
    lines = sorted(loaders["bouts"].trials["line"].astype(str).unique())
    warn_unnamed(lines)
    ntr_lines = [line for line in lines if line not in MUTANT_REFERENCES and line != "WT"]
    return ([ntr_contrast("WT")] + [ntr_contrast(line) for line in ntr_lines]
            + [mutant_contrast(line, reference) for line, reference in MUTANT_REFERENCES.items() if line in lines])


@dataclass(frozen=True)
class Screen:
    behavior_config: dict
    contrasts: List[Contrast]
    arms: Arms

    @property
    def behaviors(self) -> List[str]:
        return list(self.behavior_config)

    @property
    def map_order(self) -> List[str]:
        """Map columns in protocol order (tier1_config.MAP_COLUMNS), then the behaviors it leaves out."""
        return [b for b in MAP_COLUMNS if b in self.behavior_config] + [b for b in self.behavior_config if b not in MAP_COLUMNS]

    @property
    def null_process_factory(self) -> dict:
        return {b: cfg["null"] for b, cfg in self.behavior_config.items()}


def load_screen(data_root: Path, fits_root: Path) -> Screen:
    """The tables of data_root and the Tier 1 models built on the control fits in fits_root (run_analysis --mode fit)."""
    loaders = load_tables(data_root)
    if untested := set(get_model_config()) - set(MAP_COLUMNS):
        print(f"Experiments with control models but no Tier 1 model: {sorted(untested)}")
    config = tier1_behaviors(fits_root, PINNED_AMPLITUDES)
    print("Per-behavior model and free parameters (everything else pinned to the control fit):")
    for behavior, cfg in config.items():
        print(f"  {behavior}: {cfg['model_name']}, free={cfg['free']}")
    contrasts = screen_contrasts(loaders)
    per_behavior = lambda key: {b: cfg[key] for b, cfg in config.items()}
    return Screen(config, contrasts, Arms(loaders, per_behavior("events"), per_behavior("dataset"), per_behavior("architecture"), contrasts))
