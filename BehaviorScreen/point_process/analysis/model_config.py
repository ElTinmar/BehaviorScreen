"""Experiments: the dataset each one is built from and the candidate models, in a fixed order."""

from pathlib import Path
from typing import Callable, Dict, List, Optional

from megabouts.utils import bouts_category_name_short
from BehaviorScreen.core import SACCADE_CATEGORY_NAMES, Laterality, Stim
from BehaviorScreen.stim_specs import load_yaml_config
from BehaviorScreen.point_process.point_process.frailty import (
    BaselineOnlyFrailtyHawkesProcess,
    GammaMixedEffectsProcess,
    SideFrailty,
    ZeroInflatedBaselineOnlyFrailtyHawkesProcess,
    ZeroInflatedGammaMixedEffectsProcess,
)
from BehaviorScreen.point_process.point_process.hawkes_process import (
    HawkesProcess,
    HistoryKernelFactory,
)
from BehaviorScreen.point_process.point_process.point_process import PointProcess
from BehaviorScreen.point_process.point_process.poisson_process import (
    PoissonProcess,
    PreyCapture,
    RateKernel,
    RateKernelFactory,
)
from BehaviorScreen.point_process.point_process.sided_process import SidedPoissonProcess, SidedRenewalProcess, SidedSurvivalProcess, sided_base
from BehaviorScreen.point_process.point_process.renewal_process import (
    RenewalKernelFactory,
    RenewalProcess,
)
from BehaviorScreen.point_process.point_process.survival_process import (
    SurvivalKernelFactory,
    SurvivalProcess,
)

IPSI, CONTRA, NONDIRECTIONAL = (
    Laterality.IPSILATERAL,
    Laterality.CONTRALATERAL,
    Laterality.NONDIRECTIONAL,
)
PHOTOTAXIS_FILTER = {"foreground_color": "[0.1, 0.1, 0.0, 1.0]"}
PREY_STIM_FREQ = 90 / (2 * 70)  # stimulus speed (deg/s) / sweep range (deg)

# Tracking-quality filters of screen.yaml applied to the models; classifier-confidence
# filters are not (readouts merge neighbouring categories instead).
SCREEN_CONFIG = load_yaml_config(Path(__file__).parents[2] / "screen.yaml")
TRACKING_FILTERS = {
    "bouts": ("bout_filters", ["distance_center", "centroid_mismatch_max"]),
    "saccades": ("saccade_filters", ["distance_center", "features_valid"]),
}


def _filters(events: str) -> dict:
    section, columns = TRACKING_FILTERS[events]
    return {column: SCREEN_CONFIG[section][column] for column in columns}


# BehavioralDataLoader arguments of each event table; an experiment names its table.
EVENT_TABLES = {
    "bouts": dict(
        category_column="category",
        category_names=bouts_category_name_short,
        tracking_column="tracking_ok",
        event_filters=_filters("bouts"),
        recording_filters=SCREEN_CONFIG["common_filters"],
    ),
    "saccades": dict(
        category_column="saccade_category",
        category_names=SACCADE_CATEGORY_NAMES,
        tracking_column="tracking_ok",  # body tracking: switch to an eye-tracking flag once qc writes one
        event_filters=_filters("saccades"),
        recording_filters=SCREEN_CONFIG["common_filters"],
    ),
}

# Bout categories pooled into one response per stimulus.
FLASH_ESCAPE = ["O", "SAT"]  # flash-evoked SATs match O-bends in angle, speed and latency
LOOMING_ESCAPE = ["SLC", "LLC"]
FORWARD_SWIMS = ["S1", "S2", "BS"]
LATERAL_TURNS = ["RT", "HAT"]
OKR_SWIMS = ["S1", "S2", "RT", "HAT"]

flat = RateKernelFactory.homogeneous_poisson
gamma = GammaMixedEffectsProcess
zero_inflated = ZeroInflatedGammaMixedEffectsProcess
side_gamma = SideFrailty  # sided processes: fish gain and fish side preference


def poisson(kernel: RateKernel) -> PoissonProcess:
    return PoissonProcess(kernel)


def renewal(kernel: RateKernel) -> RenewalProcess:
    return RenewalProcess(kernel, RenewalKernelFactory.delayed_excitation())


def refractory_renewal(kernel: RateKernel) -> RenewalProcess:
    """
    A smooth recovery from the last bout of any kind (refractoriness) times the
    stimulus-driven rate plus a burst rate after the last selected bout.
    """
    return RenewalProcess(kernel, RenewalKernelFactory.exponential_recovery(), RenewalKernelFactory.burst_excitation())


def hawkes(kernel: RateKernel) -> HawkesProcess:
    return HawkesProcess(kernel, HistoryKernelFactory.exponential())


def prey(variant: str) -> RateKernel:
    return PreyCapture.variant(variant, PREY_STIM_FREQ)


def sided(ipsi: RateKernel, contra: RateKernel) -> SidedPoissonProcess:
    return SidedPoissonProcess(ipsi, contra)


def sided_renewal(ipsi: RateKernel, contra: RateKernel) -> SidedRenewalProcess:
    return SidedRenewalProcess(
        ipsi, contra, RenewalKernelFactory.exponential_recovery(), RenewalKernelFactory.burst_excitation()
    )


def sided_survival(kernel: RateKernel) -> SidedSurvivalProcess:
    """Escapes to either side as competing risks; D: odds of escaping away (contra)."""
    return SidedSurvivalProcess(kernel)


def sided_models(
    ipsi: Callable[[], RateKernel], contra: Callable[[], RateKernel], renewal: bool = False
) -> List[PointProcess]:
    """
    Both sides of a stimulus in one model: without a fish effect, with a fish gain, with a gain and a
    side preference, and with a trial gain as well; for bouts (renewal), the last two also as renewal
    processes with bursts, the gains scaling the stimulus-driven rate.
    """
    models = [sided(ipsi(), contra()), gamma(sided(ipsi(), contra())), side_gamma(sided(ipsi(), contra()))]
    models.append(side_gamma(sided(ipsi(), contra()), trial_gain=True))
    if renewal:
        models += [side_gamma(sided_renewal(ipsi(), contra())), side_gamma(sided_renewal(ipsi(), contra()), trial_gain=True)]
    return models


def dataset(category, laterality, t_end: float, binning_dt: float = 0.05, **selection) -> dict:
    """prepare_dataset arguments; category and laterality may be lists (pooled) or None (all)."""
    return dict(
        selection,
        category=category,
        laterality=laterality,
        binning_dt=binning_dt,
        t_start=0.0,
        t_end=t_end,
    )


def flat_models() -> List[PointProcess]:
    return [
        poisson(flat()),
        gamma(poisson(flat())),
        zero_inflated(poisson(flat())),
        gamma(poisson(flat()), trial_gain=True),
        gamma(refractory_renewal(flat())),
        gamma(refractory_renewal(flat()), trial_gain=True),
    ]


def rate_models(kernel: Callable[[], RateKernel]) -> List[PointProcess]:
    return [
        poisson(flat()),
        poisson(kernel()),
        gamma(poisson(flat())),
        gamma(poisson(kernel())),
        zero_inflated(poisson(kernel())),
        gamma(poisson(kernel()), trial_gain=True),
        gamma(refractory_renewal(kernel())),
        gamma(refractory_renewal(kernel()), trial_gain=True),
    ]


def survival_models(*kernels: Callable[[], object]) -> List[PointProcess]:
    survival = lambda kernel: SurvivalProcess(kernel())
    return (
        [SurvivalProcess(SurvivalKernelFactory.constant_hazard())]
        + [survival(k) for k in kernels]
        + [gamma(survival(k)) for k in kernels]
    )


def experiment(
    dataset_args: dict, models: List[PointProcess], events: str = "bouts"
) -> dict:
    """
    The null model describes the same data as the candidates: sided models also
    predict each event's side, so their null is the sided constant-rate model.
    """
    if models[0].is_survival:
        null = SurvivalProcess(SurvivalKernelFactory.constant_hazard())
    else:
        null = sided(flat(), flat()) if sided_base(models[0]) is not None else poisson(flat())
    return {
        "events": events,
        "dataset": dataset_args,
        "null_model": null,
        "models": models,
    }


def get_model_config() -> Dict[str, dict]:
    """
    Fresh (unfitted) models every call. Keep the model order: best_model.json
    stores an index (new models go at the end of a list).
    """
    looming_bump = lambda: SurvivalKernelFactory.gaussian_bump_baseline(
        t_init=5, t_bounds=(4, 6)
    )
    looming_habituating = (
        lambda: SurvivalKernelFactory.gaussian_bump_baseline_habituating(
            t_init=5, t_bounds=(4, 6)
        )
    )
    flash_bump = lambda: SurvivalKernelFactory.gaussian_bump_baseline(
        t_init=0.2, t_bounds=(0.01, 1)
    )
    flash_habituating = (
        lambda: SurvivalKernelFactory.gaussian_bump_baseline_habituating(
            t_init=0.2, t_bounds=(0.01, 1)
        )
    )
    gratings = ["grating right", "grating left"]

    return {
        "prey_capture_ipsi": experiment(
            dataset("JT", IPSI, 24.0, stim=Stim.PREY_CAPTURE),
            [poisson(flat())]
            + [
                poisson(prey(v))
                for v in [
                    "TimeOnly",
                    "Peak",
                    "Baseline",
                    "Peak_Baseline",
                    "Peak_Baseline_Ripple",
                    "Peak_Baseline_Shared",
                    "Peak_Baseline_Shared_Ripple",
                    "Peak_Baseline_Ripple_Shared",
                ]
            ]
            + [
                gamma(poisson(flat())),
                gamma(poisson(prey("Peak_Baseline_Ripple"))),
                zero_inflated(poisson(prey("Peak_Baseline_Ripple"))),
                zero_inflated(poisson(prey("Peak_Baseline_Ripple")), fit_c=True),
                # ZeroInflatedBaselineOnlyFrailtyHawkesProcess(
                    # hawkes(prey("Peak_Baseline_Ripple"))
                # ),
                # hawkes(prey("Peak_Baseline_Ripple")),
                # BaselineOnlyFrailtyHawkesProcess(hawkes(prey("Peak_Baseline_Ripple"))),
                # renewal(prey("Peak_Baseline_Ripple")),
                gamma(poisson(prey("Peak_Baseline_Ripple")), trial_gain=True),
                gamma(refractory_renewal(prey("Peak_Baseline_Ripple"))),
                gamma(refractory_renewal(prey("Peak_Baseline_Ripple")), trial_gain=True),
            ],
        ),
        "prey_capture_contra": experiment(
            dataset("JT", CONTRA, 24.0, stim=Stim.PREY_CAPTURE),
            [
                poisson(flat()),
                gamma(poisson(flat())),
                # hawkes(flat()),
                # BaselineOnlyFrailtyHawkesProcess(hawkes(flat())),
                zero_inflated(poisson(flat())),
                # ZeroInflatedBaselineOnlyFrailtyHawkesProcess(hawkes(flat())),
                gamma(poisson(flat()), trial_gain=True),
                gamma(refractory_renewal(flat())),
                gamma(refractory_renewal(flat()), trial_gain=True),
            ],
        ),
        "prey_capture_convergent_saccade": experiment(
            dataset("Convergent", NONDIRECTIONAL, 24.0, stim=Stim.PREY_CAPTURE),
            [poisson(flat())]
            + [
                poisson(prey(v))
                for v in [
                    "TimeOnly",
                    "Peak",
                    "Baseline",
                    "Peak_Baseline",
                    "Peak_Baseline_Ripple",
                    "Peak_Baseline_Shared",
                    "Peak_Baseline_Shared_Ripple",
                    "Peak_Baseline_Ripple_Shared",
                ]
            ]
            + [
                gamma(poisson(flat())),
                gamma(poisson(prey("Peak_Baseline_Ripple"))),
                zero_inflated(poisson(prey("Peak_Baseline_Ripple"))),
                zero_inflated(poisson(prey("Peak_Baseline_Ripple")), fit_c=True),
                # ZeroInflatedBaselineOnlyFrailtyHawkesProcess(
                    # hawkes(prey("Peak_Baseline_Ripple"))
                # ),
                # hawkes(prey("Peak_Baseline_Ripple")),
                # BaselineOnlyFrailtyHawkesProcess(hawkes(prey("Peak_Baseline_Ripple"))),
                # renewal(prey("Peak_Baseline_Ripple")),
                # gamma(renewal(prey("Peak_Baseline_Ripple"))),
            ],
            events="saccades",
        ),
        "prey_capture_divergent_saccade": experiment(
            dataset("Divergent", NONDIRECTIONAL, 24.0, stim=Stim.PREY_CAPTURE),
            [poisson(flat())]
            + [
                poisson(prey(v))
                for v in [
                    "TimeOnly",
                    "Peak",
                    "Baseline",
                    "Peak_Baseline",
                    "Peak_Baseline_Ripple",
                    "Peak_Baseline_Shared",
                    "Peak_Baseline_Shared_Ripple",
                    "Peak_Baseline_Ripple_Shared",
                ]
            ]
            + [
                gamma(poisson(flat())),
                gamma(poisson(prey("Peak_Baseline_Ripple"))),
                zero_inflated(poisson(prey("Peak_Baseline_Ripple"))),
                zero_inflated(poisson(prey("Peak_Baseline_Ripple")), fit_c=True),
                # ZeroInflatedBaselineOnlyFrailtyHawkesProcess(
                    # hawkes(prey("Peak_Baseline_Ripple"))
                # ),
                # hawkes(prey("Peak_Baseline_Ripple")),
                # BaselineOnlyFrailtyHawkesProcess(hawkes(prey("Peak_Baseline_Ripple"))),
                # renewal(prey("Peak_Baseline_Ripple")),
                # gamma(renewal(prey("Peak_Baseline_Ripple"))),
            ],
            events="saccades",
        ),
        "phototaxis_ipsi": experiment(
            dataset(
                "RT", IPSI, 24.0, stim=Stim.PHOTOTAXIS, trial_filters=PHOTOTAXIS_FILTER
            ),
            [
                poisson(flat()),
                poisson(RateKernelFactory.phototaxis_ipsi()),
                poisson(RateKernelFactory.phototaxis_dip_exgaussian_peak()),
                gamma(poisson(flat())),
                gamma(poisson(RateKernelFactory.phototaxis_ipsi())),
                gamma(poisson(RateKernelFactory.phototaxis_dip_exgaussian_peak())),
                gamma(poisson(RateKernelFactory.phototaxis_dip_exgaussian_peak()), trial_gain=True),
                gamma(refractory_renewal(RateKernelFactory.phototaxis_dip_exgaussian_peak())),
                gamma(refractory_renewal(RateKernelFactory.phototaxis_dip_exgaussian_peak()), trial_gain=True),
            ],
        ),
        "phototaxis_contra": experiment(
            dataset(
                "RT",
                CONTRA,
                24.0,
                stim=Stim.PHOTOTAXIS,
                trial_filters=PHOTOTAXIS_FILTER,
            ),
            rate_models(RateKernelFactory.phototaxis_contra),
        ),
        "omr_lateral_ipsi": experiment(
            dataset(LATERAL_TURNS, IPSI, 9.0, epoch_name=gratings), flat_models(),
        ),
        "omr_lateral_contra": experiment(
            dataset(LATERAL_TURNS, CONTRA, 9.0, epoch_name=gratings),
            rate_models(RateKernelFactory.omr_lateral_contra),
        ),
        "omr_forward": experiment(
            dataset(FORWARD_SWIMS, NONDIRECTIONAL, 9.0, epoch_name="grating forward"),
            rate_models(RateKernelFactory.omr_forward),
        ),
        "okr_ipsi": experiment(
            dataset(OKR_SWIMS, IPSI, 9.0, stim=Stim.OKR), flat_models(),
        ),
        "okr_contra": experiment(
            dataset(OKR_SWIMS, CONTRA, 9.0, stim=Stim.OKR), flat_models(),
        ),
        # Both sides of a directional stimulus: strength (ipsi amplitudes) and directionality D
        "prey_capture": experiment(
            dataset("JT", [IPSI, CONTRA], 24.0, stim=Stim.PREY_CAPTURE),
            sided_models(lambda: prey("Peak_Baseline_Ripple"), flat, renewal=True),
        ),
        "phototaxis": experiment(
            dataset("RT", [IPSI, CONTRA], 24.0, stim=Stim.PHOTOTAXIS, trial_filters=PHOTOTAXIS_FILTER),
            sided_models(RateKernelFactory.phototaxis_dip_exgaussian_peak, RateKernelFactory.phototaxis_contra, renewal=True),
        ),
        "omr_lateral": experiment(
            dataset(LATERAL_TURNS, [IPSI, CONTRA], 9.0, epoch_name=gratings),
            sided_models(RateKernelFactory.omr_lateral, RateKernelFactory.omr_lateral, renewal=True),
        ),
        "okr": experiment(
            dataset(OKR_SWIMS, [IPSI, CONTRA], 9.0, stim=Stim.OKR), sided_models(flat, flat, renewal=True),
        ),
        "okr_conjugate_saccades": experiment(
            dataset("Conjugate", [IPSI, CONTRA], 9.0, stim=Stim.OKR), sided_models(flat, flat), events="saccades",
        ),
        "looming": experiment(
            dataset(LOOMING_ESCAPE, [IPSI, CONTRA], 9.0, stim=Stim.LOOMING),
            survival_models(looming_bump, looming_habituating)
            + [sided_survival(looming_habituating()), gamma(sided_survival(looming_habituating())), side_gamma(sided_survival(looming_habituating()))],
        ),
        "looming_divergent_saccade": experiment(
            dataset("Divergent", NONDIRECTIONAL, 9.0, stim=Stim.LOOMING),
            survival_models(looming_bump, looming_habituating),
            events="saccades",
        ),
        "dark_flash": experiment(
            dataset(
                FLASH_ESCAPE, NONDIRECTIONAL, 5.0, binning_dt=0.025, epoch_name="flash dark"
            ),
            survival_models(
                flash_bump,
                flash_habituating,
                SurvivalKernelFactory.exgaussian_bump_baseline_habituating,
            ),
        ),
        "spont_dark": experiment(
            dataset(None, None, 24.0, epoch_name="spontaneous dark"),
            rate_models(RateKernelFactory.spont),
        ),
        "spont_bright": experiment(
            dataset(None, None, 24.0, epoch_name="spontaneous bright"),
            rate_models(RateKernelFactory.spont),
        ),
        "after_looming": experiment(
            dataset(
                None,
                None,
                49.0,
                epoch_name=["looming break after left", "looming break after right"],
            ),
            [
                gamma(poisson(RateKernelFactory.after_looming())),
                zero_inflated(poisson(RateKernelFactory.after_looming())),
                gamma(poisson(RateKernelFactory.after_looming()), trial_gain=True),
                gamma(refractory_renewal(RateKernelFactory.after_looming())),
                gamma(refractory_renewal(RateKernelFactory.after_looming()), trial_gain=True),
            ],
        ),
        "okr_conjugate_contra_saccades": experiment(
            dataset("Conjugate", CONTRA, 9.0, stim=Stim.OKR),
            [
                poisson(flat()),
                gamma(poisson(flat())),
                zero_inflated(poisson(flat())),
                # hawkes(flat()),
                # zero_inflated(renewal(flat())),
            ],
            events="saccades",
        ),
        "okr_conjugate_ipsi_saccades": experiment(
            dataset("Conjugate", IPSI, 9.0, stim=Stim.OKR),
            [
                poisson(flat()),
                gamma(poisson(flat())),
                zero_inflated(poisson(flat())),
                # hawkes(flat()),
                # zero_inflated(renewal(flat())),
            ],
            events="saccades",
        ),
    }
