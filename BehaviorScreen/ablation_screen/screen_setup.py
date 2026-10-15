"""
The screen's configuration, shared by run_screen and the commands run after
it (arm_bootstrap): data, Tier 1 models (each experiment's control fit with
every parameter pinned except tier1_config.AMPLITUDE_PARAMETERS), contrasts
and the arms they compare. Importing it loads the event tables.
"""
from pathlib import Path

from BehaviorScreen.point_process.point_process.dataset import BehavioralDataLoader
from BehaviorScreen.point_process.analysis.model_config import (
    EVENT_TABLES, flat, gamma, get_model_config, poisson, prey, sided, sided_survival,
)
from BehaviorScreen.point_process.point_process.poisson_process import RateKernelFactory
from BehaviorScreen.point_process.point_process.survival_process import SurvivalKernelFactory, SurvivalProcess
from BehaviorScreen.ablation_screen.dataset_utils import Arms, mutant_contrast, ntr_contrast
from BehaviorScreen.ablation_screen.tier1_config import MAP_COLUMNS, tier1_behaviors

OUTPUT_DIR = Path("./tier1_results")
OUTPUT_DIR.mkdir(exist_ok=True)

ROOT = Path('/media/martin/DATA_18TB/Screen')
FITS_ROOT = Path("./figures")  # run_analysis --mode fit output on the control screen
# Tables merged across screen folders with merge_csv (adds line/condition columns)
EVENTS_CSV = {"bouts": "bouts_all.csv", "saccades": "saccades_augmented_all.csv"}
loaders = {
    events: BehavioralDataLoader(ROOT / csv, [ROOT / "valid_trials_all.csv"], [ROOT / "qc_all.csv"], **EVENT_TABLES[events])
    for events, csv in EVENTS_CSV.items()
}

looming = lambda: SurvivalKernelFactory.gaussian_bump_baseline_habituating(t_init=5, t_bounds=(4, 6))
# Directional stimuli: both sides in one sided model (strength: ipsi amplitudes; direction: D).
TIER1_MODELS = {
    "prey_capture": lambda: gamma(sided(prey("Peak_Baseline_Ripple"), flat())),
    "phototaxis": lambda: gamma(sided(RateKernelFactory.phototaxis_dip_exgaussian_peak(), RateKernelFactory.phototaxis_contra())),
    "omr_lateral": lambda: gamma(sided(flat(), RateKernelFactory.omr_lateral_contra())),
    "omr_forward": lambda: gamma(poisson(RateKernelFactory.omr_forward())),
    "okr": lambda: gamma(sided(flat(), flat())),
    "looming": lambda: gamma(sided_survival(looming())),  # strength: H (escape at all); direction: D (away)
    "dark_flash": lambda: gamma(SurvivalProcess(SurvivalKernelFactory.exgaussian_bump_baseline_habituating())),
    "spont_dark": lambda: gamma(poisson(RateKernelFactory.spont())),
    "spont_bright": lambda: gamma(poisson(RateKernelFactory.spont())),
    "after_looming": lambda: gamma(poisson(RateKernelFactory.after_looming())),
    "prey_capture_convergent_saccade": lambda: gamma(poisson(prey("Peak_Baseline_Ripple"))),
    "prey_capture_divergent_saccade": lambda: gamma(poisson(prey("Peak_Baseline_Ripple"))),
    "okr_conjugate_saccades": lambda: gamma(sided(flat(), flat())),
    "looming_divergent_saccade": lambda: gamma(SurvivalProcess(looming())),
}
# Map columns in protocol order, labelled "stimulus · response" with the stimulus names
# of screen.yaml and the bout or saccade categories each experiment counts.
untested = set(get_model_config()) - set(TIER1_MODELS)
if untested:
    print(f"Experiments with control models but no Tier 1 model: {sorted(untested)}")
# Amplitudes also pinned to the control fit. Escape baselines are near zero (about
# one event a minute), so their per-arm log ratios are noise; the prey saccade
# transients have too few evoked events per fish. Transients stay free for prey
# J-turns (A) and phototaxis (A_peak).
PINNED_AMPLITUDES = {
    **{name: ["B"] for name in ("looming", "looming_divergent_saccade", "dark_flash")},
    **{name: ["A"] for name in ("prey_capture_convergent_saccade", "prey_capture_divergent_saccade")},
}
BEHAVIOR_CONFIG = tier1_behaviors(TIER1_MODELS, FITS_ROOT, PINNED_AMPLITUDES)
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
    "blu": WT,              # vglut2a tz257, TL/Tuebingen (Smear et al. 2007)
    "mecp2": ("AB", "danieau"),  # outcrossed to AB (Pietri et al. 2013)
}
all_lines = sorted(loaders["bouts"].trials["line"].astype(str).unique())
NTR_LINES = [l for l in all_lines if l not in MUTANT_REFERENCES and l != "WT"]
CONTRASTS = (
    [ntr_contrast("WT")]
    + [ntr_contrast(line) for line in NTR_LINES]
    + [mutant_contrast(line, reference) for line, reference in MUTANT_REFERENCES.items() if line in all_lines]
)

arms = Arms(loaders, {b: cfg["events"] for b, cfg in BEHAVIOR_CONFIG.items()}, DATASET_CONFIGS, BEHAVIOR_PROCESS_FACTORY, CONTRASTS)
