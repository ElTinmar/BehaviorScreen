"""Experiments: the dataset each one is built from and the candidate models, in a fixed order."""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from megabouts.utils import bouts_category_name_short
from BehaviorScreen.core import SACCADE_CATEGORY_NAMES, Laterality, Stim
from BehaviorScreen.filters import load_yaml_config
from BehaviorScreen.point_process.frailty import (
    BaselineOnlyFrailtyHawkesProcess,
    GammaMixedEffectsProcess,
    SideFrailty,
    ZeroInflatedFamily,
    ZeroInflatedBaselineOnlyFrailtyHawkesProcess,
    ZeroInflatedGammaMixedEffectsProcess,
)
from BehaviorScreen.point_process.hawkes_process import (
    HawkesProcess,
    HistoryKernelFactory,
)
from BehaviorScreen.screen.loader import BehavioralDataLoader
from BehaviorScreen.point_process.point_process import PointProcess
from BehaviorScreen.point_process.poisson_process import (
    PoissonProcess,
    PreyCapture,
    RateKernel,
    RateKernelFactory,
)
from BehaviorScreen.point_process.sided_process import SidedPoissonProcess, SidedRenewalProcess, SidedSurvivalProcess, sided_base
from BehaviorScreen.point_process.renewal_process import (
    RenewalKernel,
    RenewalKernelFactory,
    RenewalProcess,
)
from BehaviorScreen.point_process.survival_process import (
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


# Straight mutants and other strains, each compared with its wild-type background (line: reference arm).
# wik and AB measure how far background (and recording day) alone moves each behavior from WT (TL).
WT = ("WT", "danieau")
MUTANT_REFERENCES = {
    "wik": WT,
    "AB": WT,
    "lak": WT,              # atoh7 th241, TL (Tuebingen)
    "gr": WT,               # nr3c1 s357, TL (Muto et al. 2005)
    "blu": WT,              # vglut2a tc257z, TL/Tuebingen (Smear et al. 2007)
    "mecp2": ("AB", "danieau"),  # outcrossed to AB (Pietri et al. 2013)
}
CONTROL_CONDITIONS = ("vehicle", "danieau")


def control_loader(loader: BehavioralDataLoader) -> BehavioralDataLoader:
    """The fish of the control fits: the untreated arms (vehicle, danieau) of WT and of the NTR lines."""
    return loader.subset(conditions=CONTROL_CONDITIONS, exclude_lines=tuple(MUTANT_REFERENCES))


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
decay = RateKernelFactory.exponential_decay  # prey capture's simplest stimulus kernel, for teaching
gamma = GammaMixedEffectsProcess
zero_inflated = ZeroInflatedGammaMixedEffectsProcess
side_gamma = SideFrailty  # sided processes: fish gain and fish side preference


def two_class(process: PointProcess) -> PointProcess:
    """
    A fish gain on the driven rate that is either low (c < 0.3, with probability pi) or
    Gamma around the rest: more low-activity fish and fewer extremes than one Gamma.
    """
    if sided_base(process) is not None:
        return side_gamma(process, gains=ZeroInflatedFamily(fit_c=True, c_init=0.1))
    return zero_inflated(process, fit_c=True, c_init=0.1)


def poisson(kernel: RateKernel) -> PoissonProcess:
    return PoissonProcess(kernel)


def renewal(kernel: RateKernel) -> RenewalProcess:
    return RenewalProcess(kernel, RenewalKernelFactory.delayed_excitation())


def refractory_renewal(kernel: RateKernel) -> RenewalProcess:
    """
    A recovery from the last bout of any kind (refractoriness, sharp enough for the
    near-rhythmic intervals between bouts) times the stimulus-driven rate and a burst
    after the last selected bout, a multiple of it.
    """
    return RenewalProcess(kernel, RenewalKernelFactory.weibull_recovery(), stimulus_burst())


def stimulus_burst(shape_k: float = 1.0) -> RenewalKernel:
    """
    Bursts as a multiple of the stimulus-driven rate: they follow the stimulus but not the fish and
    trial gains. k = 1 (an alpha function) from the profile of k on prey_capture controls.
    """
    return RenewalKernelFactory.burst_excitation(shape_k, rate_init=2.0, relative=True)


def hawkes(kernel: RateKernel) -> HawkesProcess:
    return HawkesProcess(kernel, HistoryKernelFactory.exponential())


def prey(variant: str) -> RateKernel:
    return PreyCapture.variant(variant, PREY_STIM_FREQ)


def sided(ipsi: RateKernel, contra: RateKernel) -> SidedPoissonProcess:
    return SidedPoissonProcess(ipsi, contra)


def sided_renewal(ipsi: RateKernel, contra: RateKernel, shape_k: float = 1.0) -> SidedRenewalProcess:
    return SidedRenewalProcess(ipsi, contra, RenewalKernelFactory.weibull_recovery(), stimulus_burst(shape_k))


def sided_survival(kernel: RateKernel) -> SidedSurvivalProcess:
    """Escapes to either side as competing risks; D: odds of escaping away (contra)."""
    return SidedSurvivalProcess(kernel)


def sided_models(
    ipsi: Callable[[], RateKernel], contra: Callable[[], RateKernel], renewal: bool = False,
    burst_shapes: Sequence[float] = (), renewal_ipsi: Optional[Callable[[], RateKernel]] = None,
) -> List[PointProcess]:
    """
    Both sides of a stimulus in one model: without a fish effect, with a fish gain, with a gain and a
    side preference, and with a trial gain as well; for bouts (renewal), the last two also as renewal
    processes with bursts that follow the stimulus, the gains scaling the stimulus-driven rate only,
    and last the renewal one with a two-class fish gain (two_class). burst_shapes: the renewal one
    with a trial gain again with other burst shapes k (a profile of k, warm-started from k = 1).
    renewal_ipsi: the ipsi kernel of the renewal models, if not ipsi.
    """
    models = [sided(ipsi(), contra()), gamma(sided(ipsi(), contra())), side_gamma(sided(ipsi(), contra()))]
    models.append(side_gamma(sided(ipsi(), contra()), trial_gain=True))
    if renewal:
        ipsi = renewal_ipsi or ipsi
        models += [side_gamma(sided_renewal(ipsi(), contra())), side_gamma(sided_renewal(ipsi(), contra()), trial_gain=True)]
        models += [side_gamma(sided_renewal(ipsi(), contra(), k), trial_gain=True) for k in burst_shapes]
        models.append(two_class(sided_renewal(ipsi(), contra())))
    return models


def renewal_hour_rungs(kernel: Callable[[], RateKernel]) -> List[PointProcess]:
    """
    The time-of-day factor (time_of_day) on a renewal ladder's winner (fish and trial gains) and on its
    first fish-gain rung without history: the effect beyond bout history, and all of it.
    """
    return [gamma(refractory_renewal(kernel()), trial_gain=True, time_of_day=True), gamma(poisson(kernel()), time_of_day=True)]


def sided_hour_rungs(
    ipsi: Callable[[], RateKernel], contra: Callable[[], RateKernel], renewal_ipsi: Optional[Callable[[], RateKernel]] = None,
) -> List[PointProcess]:
    """renewal_hour_rungs for sided_models: the renewal rung with a trial gain and the one with a side preference, without history."""
    return [side_gamma(sided_renewal((renewal_ipsi or ipsi)(), contra()), trial_gain=True, time_of_day=True),
            side_gamma(sided(ipsi(), contra()), time_of_day=True)]


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
        two_class(refractory_renewal(flat())),
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
        two_class(refractory_renewal(kernel())),
    ]


def survival_models(*kernels: Callable[[], object], sided: bool = False) -> List[PointProcess]:
    """Sided ladders also predict each escape's side, ending with the fish side preference."""
    survival = lambda kernel: sided_survival(kernel()) if sided else SurvivalProcess(kernel())
    models = (
        [survival(SurvivalKernelFactory.constant_hazard)]
        + [survival(k) for k in kernels]
        + [gamma(survival(k)) for k in kernels]
    )
    return models + [side_gamma(survival(kernels[-1]))] if sided else models


def experiment(
    dataset_args: dict, models: List[PointProcess], events: str = "bouts"
) -> dict:
    """
    The null model describes the same data as the candidates: sided models also
    predict each event's side, so their null is the sided constant-rate model.
    It is the first rung of the ladder (models), unless a candidate already is it.
    """
    is_sided = sided_base(models[0]) is not None
    if models[0].is_survival:
        hazard = SurvivalKernelFactory.constant_hazard()
        null = sided_survival(hazard) if is_sided else SurvivalProcess(hazard)
    else:
        null = sided(flat(), flat()) if is_sided else poisson(flat())
    null = next((m for m in models if m.name == null.name), null)
    models = models if any(m is null for m in models) else [null] + models
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
                two_class(refractory_renewal(prey("Peak_Baseline_Ripple"))),
                poisson(decay()),
                gamma(poisson(decay())),
            ] + renewal_hour_rungs(lambda: prey("Peak_Baseline_Ripple")),
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
                two_class(refractory_renewal(flat())),
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
                poisson(decay()),
                gamma(poisson(decay())),
                gamma(poisson(prey("Peak_Baseline_Ripple")), time_of_day=True),
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
                gamma(poisson(prey("Baseline_Ripple"))),
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
                poisson(decay()),
                gamma(poisson(decay())),
                gamma(poisson(prey("Baseline_Ripple")), time_of_day=True),
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
                two_class(refractory_renewal(RateKernelFactory.phototaxis_dip_exgaussian_peak())),
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
            rate_models(RateKernelFactory.omr_forward) + renewal_hour_rungs(RateKernelFactory.omr_forward),
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
            sided_models(lambda: prey("Peak_Baseline_Ripple"), flat, renewal=True) + [sided(decay(), flat()), side_gamma(sided(decay(), flat()))],
        ),
        "phototaxis": experiment(
            dataset("RT", [IPSI, CONTRA], 24.0, stim=Stim.PHOTOTAXIS, trial_filters=PHOTOTAXIS_FILTER),
            sided_models(RateKernelFactory.phototaxis_dip_exgaussian_peak, RateKernelFactory.phototaxis_contra, renewal=True)
            + sided_hour_rungs(RateKernelFactory.phototaxis_dip_exgaussian_peak, RateKernelFactory.phototaxis_contra),
        ),
        "omr_lateral": experiment(
            dataset(LATERAL_TURNS, [IPSI, CONTRA], 9.0, epoch_name=gratings),
            sided_models(RateKernelFactory.omr_lateral, RateKernelFactory.omr_lateral, renewal=True, renewal_ipsi=RateKernelFactory.omr_lateral_ipsi)
            + sided_hour_rungs(RateKernelFactory.omr_lateral, RateKernelFactory.omr_lateral, renewal_ipsi=RateKernelFactory.omr_lateral_ipsi),
        ),
        "okr": experiment(
            dataset(OKR_SWIMS, [IPSI, CONTRA], 9.0, stim=Stim.OKR), sided_models(flat, flat, renewal=True) + sided_hour_rungs(flat, flat),
        ),
        "okr_conjugate_saccades": experiment(
            dataset("Conjugate", [IPSI, CONTRA], 9.0, stim=Stim.OKR), sided_models(flat, flat) + [side_gamma(sided(flat(), flat()), time_of_day=True)],
            events="saccades",
        ),
        "looming": experiment(
            dataset(LOOMING_ESCAPE, [IPSI, CONTRA], 9.0, stim=Stim.LOOMING),
            survival_models(looming_bump, looming_habituating, sided=True)
            + [side_gamma(sided_survival(looming_habituating()), time_of_day=True)],
        ),
        "looming_divergent_saccade": experiment(
            dataset("Divergent", NONDIRECTIONAL, 9.0, stim=Stim.LOOMING),
            survival_models(looming_bump, looming_habituating) + [gamma(SurvivalProcess(looming_habituating()), time_of_day=True)],
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
            ) + [gamma(SurvivalProcess(SurvivalKernelFactory.exgaussian_bump_baseline_habituating()), time_of_day=True)],
        ),
        "spont_dark": experiment(
            dataset(None, None, 24.0, epoch_name="spontaneous dark"),
            rate_models(RateKernelFactory.spont) + renewal_hour_rungs(RateKernelFactory.spont),
        ),
        "spont_bright": experiment(
            dataset(None, None, 24.0, epoch_name="spontaneous bright"),
            rate_models(RateKernelFactory.spont) + renewal_hour_rungs(RateKernelFactory.spont),
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
                two_class(refractory_renewal(RateKernelFactory.after_looming())),
            ] + renewal_hour_rungs(RateKernelFactory.after_looming),
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
