"""Experiments: the dataset each one is built from and the candidate models, in a fixed order."""
from typing import Callable, Dict, List

from BehaviorScreen.core import Laterality, Stim
from BehaviorScreen.point_process.point_process.frailty import (
    BaselineOnlyFrailtyHawkesProcess,
    GammaMixedEffectsProcess,
    ZeroInflatedBaselineOnlyFrailtyHawkesProcess,
    ZeroInflatedGammaMixedEffectsProcess,
)
from BehaviorScreen.point_process.point_process.hawkes_process import HawkesProcess, HistoryKernelFactory
from BehaviorScreen.point_process.point_process.point_process import PointProcess
from BehaviorScreen.point_process.point_process.poisson_process import (
    PoissonProcess,
    PreyCapture,
    RateKernel,
    RateKernelFactory,
)
from BehaviorScreen.point_process.point_process.renewal_process import RenewalKernelFactory, RenewalProcess
from BehaviorScreen.point_process.point_process.survival_process import SurvivalKernelFactory, SurvivalProcess

IPSI, CONTRA, NONDIRECTIONAL = Laterality.IPSILATERAL, Laterality.CONTRALATERAL, Laterality.NONDIRECTIONAL
PHOTOTAXIS_FILTER = {"foreground_color": "[0.1, 0.1, 0.0, 1.0]"}
PREY_STIM_FREQ = 90 / (2 * 70)  # stimulus speed (deg/s) / sweep range (deg)

flat = RateKernelFactory.homogeneous_poisson
gamma = GammaMixedEffectsProcess
zero_inflated = ZeroInflatedGammaMixedEffectsProcess


def poisson(kernel: RateKernel) -> PoissonProcess:
    return PoissonProcess(kernel)


def renewal(kernel: RateKernel) -> RenewalProcess:
    return RenewalProcess(kernel, RenewalKernelFactory.delayed_excitation())


def dead_time_renewal(kernel: RateKernel, dead_time_s: float) -> RenewalProcess:
    return RenewalProcess(kernel, RenewalKernelFactory.dead_time_delayed_excitation(dead_time_s))


def hawkes(kernel: RateKernel) -> HawkesProcess:
    return HawkesProcess(kernel, HistoryKernelFactory.exponential())


def prey(variant: str) -> RateKernel:
    return PreyCapture.variant(variant, PREY_STIM_FREQ)


def dataset(category: str, laterality, t_end: float, binning_dt: float = 0.05, **selection) -> dict:
    """prepare_dataset arguments; selection is stim or epoch_name, and trial_filters."""
    return dict(selection, category=category, laterality=laterality, binning_dt=binning_dt, t_start=0.0, t_end=t_end)


def flat_models(dead_time_s: float) -> List[PointProcess]:
    return [
        poisson(flat()),
        gamma(poisson(flat())),
        zero_inflated(poisson(flat())),
        zero_inflated(renewal(flat())),
        zero_inflated(dead_time_renewal(flat(), dead_time_s)),
    ]


def rate_models(kernel: Callable[[], RateKernel], dead_time_s: float, flat_renewal: bool = True) -> List[PointProcess]:
    models = [
        poisson(flat()),
        poisson(kernel()),
        gamma(poisson(flat())),
        gamma(poisson(kernel())),
        zero_inflated(poisson(kernel())),
        zero_inflated(renewal(kernel())),
    ]
    models += [zero_inflated(renewal(flat()))] if flat_renewal else []
    return models + [zero_inflated(dead_time_renewal(kernel(), dead_time_s))]


def survival_models(*kernels: Callable[[], object]) -> List[PointProcess]:
    survival = lambda kernel: SurvivalProcess(kernel())
    return (
        [SurvivalProcess(SurvivalKernelFactory.constant_hazard())]
        + [survival(k) for k in kernels]
        + [gamma(survival(k)) for k in kernels]
    )


def experiment(dataset_args: dict, models: List[PointProcess]) -> dict:
    null = (
        SurvivalProcess(SurvivalKernelFactory.constant_hazard())
        if models[0].is_survival
        else poisson(flat())
    )
    return {"dataset": dataset_args, "null_model": null, "models": models}


def get_model_config(dead_time_s: float) -> Dict[str, dict]:
    """
    Fresh (unfitted) models every call. Keep the model order: best_model.json
    stores an index (new models go at the end of a list). dead_time_s: the
    dead time of the renewal models that have one (BehavioralDataLoader.dead_time).
    """
    looming_bump = lambda: SurvivalKernelFactory.gaussian_bump_baseline(t_init=5, t_bounds=(4, 6))
    looming_habituating = lambda: SurvivalKernelFactory.gaussian_bump_baseline_habituating(t_init=5, t_bounds=(4, 6))
    flash_bump = lambda: SurvivalKernelFactory.gaussian_bump_baseline(t_init=0.2, t_bounds=(0.01, 1))
    flash_habituating = lambda: SurvivalKernelFactory.gaussian_bump_baseline_habituating(t_init=0.2, t_bounds=(0.01, 1))
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
                ZeroInflatedBaselineOnlyFrailtyHawkesProcess(hawkes(prey("Peak_Baseline_Ripple"))),
                hawkes(prey("Peak_Baseline_Ripple")),
                BaselineOnlyFrailtyHawkesProcess(hawkes(prey("Peak_Baseline_Ripple"))),
                zero_inflated(renewal(prey("Peak_Baseline_Ripple"))),
            ],
        ),
        "prey_capture_contra": experiment(
            dataset("JT", CONTRA, 24.0, stim=Stim.PREY_CAPTURE),
            [
                poisson(flat()),
                gamma(poisson(flat())),
                hawkes(flat()),
                BaselineOnlyFrailtyHawkesProcess(hawkes(flat())),
                zero_inflated(poisson(flat())),
                ZeroInflatedBaselineOnlyFrailtyHawkesProcess(hawkes(flat())),
                zero_inflated(renewal(flat())),
            ],
        ),
        "phototaxis_ipsi": experiment(
            dataset("RT", IPSI, 24.0, stim=Stim.PHOTOTAXIS, trial_filters=PHOTOTAXIS_FILTER),
            [
                poisson(flat()),
                poisson(RateKernelFactory.phototaxis_ipsi()),
                poisson(RateKernelFactory.phototaxis_dip_exgaussian_peak()),
                gamma(poisson(flat())),
                gamma(poisson(RateKernelFactory.phototaxis_ipsi())),
                gamma(poisson(RateKernelFactory.phototaxis_dip_exgaussian_peak())),
                zero_inflated(renewal(RateKernelFactory.phototaxis_exgaussian_peak_no_dip())),
            ],
        ),
        "phototaxis_contra": experiment(
            dataset("RT", CONTRA, 24.0, stim=Stim.PHOTOTAXIS, trial_filters=PHOTOTAXIS_FILTER),
            rate_models(RateKernelFactory.phototaxis_contra, dead_time_s),
        ),
        "omr_lateral_ipsi": experiment(dataset("RT", IPSI, 9.0, epoch_name=gratings), flat_models(dead_time_s)),
        "omr_lateral_contra": experiment(
            dataset("RT", CONTRA, 9.0, epoch_name=gratings), rate_models(RateKernelFactory.omr_lateral_contra, dead_time_s)
        ),
        "omr_forward": experiment(
            dataset("BS", NONDIRECTIONAL, 9.0, epoch_name="grating forward"),
            rate_models(RateKernelFactory.omr_forward, dead_time_s, flat_renewal=False),
        ),
        "okr_ipsi": experiment(dataset("S1", IPSI, 9.0, stim=Stim.OKR), flat_models(dead_time_s)),
        "okr_contra": experiment(dataset("S1", CONTRA, 9.0, stim=Stim.OKR), flat_models(dead_time_s)),
        "looming_ipsi": experiment(
            dataset("SLC", IPSI, 9.0, stim=Stim.LOOMING), survival_models(looming_bump, looming_habituating)
        ),
        "looming_contra": experiment(
            dataset("SLC", CONTRA, 9.0, stim=Stim.LOOMING), survival_models(looming_bump, looming_habituating)
        ),
        "dark_flash": experiment(
            dataset("O", NONDIRECTIONAL, 5.0, binning_dt=0.025, epoch_name="flash dark"),
            survival_models(
                flash_bump, flash_habituating, SurvivalKernelFactory.exgaussian_bump_baseline_habituating
            ),
        ),
        "spont_dark": experiment(
            dataset("RT", NONDIRECTIONAL, 24.0, epoch_name="spontaneous dark"), rate_models(RateKernelFactory.spont, dead_time_s)
        ),
        "spont_bright": experiment(
            dataset("RT", NONDIRECTIONAL, 24.0, epoch_name="spontaneous bright"), rate_models(RateKernelFactory.spont, dead_time_s)
        ),
        "after_looming": experiment(
            dataset(
                "RT",
                [IPSI, CONTRA],
                49.0,
                epoch_name=["looming break after left", "looming break after right"],
            ),
            [
                gamma(poisson(RateKernelFactory.after_looming())),
                zero_inflated(poisson(RateKernelFactory.after_looming())),
                zero_inflated(renewal(RateKernelFactory.after_looming())),
                zero_inflated(renewal(flat())),
            ],
        ),
    }
