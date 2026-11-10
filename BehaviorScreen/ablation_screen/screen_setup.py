"""
The screen's configuration, shared by run_screen and the commands run after
it (arm_bootstrap): data, Tier 1 models (each experiment's best control model with
every parameter pinned except tier1_config.AMPLITUDE_PARAMETERS), contrasts
and the arms they compare. Importing it loads the event tables.
"""
import os
from pathlib import Path

from BehaviorScreen.point_process.point_process.dataset import BehavioralDataLoader
from BehaviorScreen.point_process.analysis.model_config import EVENT_TABLES, get_model_config
from BehaviorScreen.ablation_screen.dataset_utils import Arms, mutant_contrast, ntr_contrast
from BehaviorScreen.ablation_screen.line_names import warn_unnamed
from BehaviorScreen.ablation_screen.tier1_config import MAP_COLUMNS, PINNED_AMPLITUDES, tier1_behaviors

# Paths: run_screen's arguments set them (or the environment does)
OUTPUT_DIR = Path(os.environ.get("SCREEN_OUTPUT_DIR", "./tier1_results"))
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

ROOT = Path(os.environ.get("SCREEN_DATA_ROOT", "/media/martin/DATA_18TB/Screen"))
FITS_ROOT = Path(os.environ.get("SCREEN_FITS_ROOT", "./figures"))  # run_analysis --mode fit output on the control screen
# Tables merged across screen folders with merge_csv (adds line/condition columns)
EVENTS_CSV = {"bouts": "bouts_all.csv", "saccades": "saccades_augmented_all.csv"}
loaders = {
    events: BehavioralDataLoader(ROOT / csv, [ROOT / "valid_trials_all.csv"], [ROOT / "qc_all.csv"], **EVENT_TABLES[events])
    for events, csv in EVENTS_CSV.items()
}

# Map columns in protocol order, labelled "stimulus · response" with the stimulus names
# of screen.yaml and the bout or saccade categories each experiment counts.
untested = set(get_model_config()) - set(MAP_COLUMNS)
if untested:
    print(f"Experiments with control models but no Tier 1 model: {sorted(untested)}")
BEHAVIOR_CONFIG = tier1_behaviors(FITS_ROOT, PINNED_AMPLITUDES)
BEHAVIORS = list(BEHAVIOR_CONFIG)
MAP_ORDER = [b for b in MAP_COLUMNS if b in BEHAVIORS] + [b for b in BEHAVIORS if b not in MAP_COLUMNS]
DATASET_CONFIGS = {b: cfg['dataset'] for b, cfg in BEHAVIOR_CONFIG.items()}
BEHAVIOR_PROCESS_FACTORY = {b: cfg['architecture'] for b, cfg in BEHAVIOR_CONFIG.items()}
NULL_PROCESS_FACTORY = {b: cfg['null'] for b, cfg in BEHAVIOR_CONFIG.items()}

print("Per-behavior model and free parameters (everything else pinned to the control fit):")
for behavior, cfg in BEHAVIOR_CONFIG.items():
    print(f"  {behavior}: {cfg['model_name']}, free={cfg['free']}")

# Straight mutants against their wild-type background. wik and AB measure how
# far background (and recording day) alone moves each behavior from WT (TL).
WT = ("WT", "danieau")
MUTANT_REFERENCES = {
    "wik": WT,
    "AB": WT,
    "lak": WT,              # atoh7 th241, TL (Tuebingen)
    "gr": WT,               # nr3c1 s357, TL (Muto et al. 2005)
    "blu": WT,              # vglut2a tc257z, TL/Tuebingen (Smear et al. 2007)
    "mecp2": ("AB", "danieau"),  # outcrossed to AB (Pietri et al. 2013)
}
all_lines = sorted(loaders["bouts"].trials["line"].astype(str).unique())
warn_unnamed(all_lines)
NTR_LINES = [l for l in all_lines if l not in MUTANT_REFERENCES and l != "WT"]
CONTRASTS = (
    [ntr_contrast("WT")]
    + [ntr_contrast(line) for line in NTR_LINES]
    + [mutant_contrast(line, reference) for line, reference in MUTANT_REFERENCES.items() if line in all_lines]
)

arms = Arms(loaders, {b: cfg["events"] for b, cfg in BEHAVIOR_CONFIG.items()}, DATASET_CONFIGS, BEHAVIOR_PROCESS_FACTORY, CONTRASTS)
