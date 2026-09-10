"""
TEMPORARY. Validates the order-statistic/Fisher-combination calibration
fix (PointProcess.combined_stream_calibration_test) directly against the
EXACT short-duration configurations that failed under the old pooled-gap
KS test -- confirms the new approach is valid without inflating
duration_s, before fully retiring the old test/panel.

Delete once you've confirmed this and are confident in the replacement.
"""
import numpy as np
import pytest

from scipy.stats import kstest

from .conftest import make_scaffold_dataset, simulate_dataset_from_model, dataset_from_streams

from BehaviorScreen.point_process.poisson_process import PoissonProcess, RateKernelFactory
from BehaviorScreen.point_process.hawkes_process import HawkesProcess, HistoryKernelFactory
from BehaviorScreen.point_process.renewal_process import RenewalProcess, RenewalKernelFactory
from BehaviorScreen.point_process.kernel_shapes import logit_bounded


def test_poisson_homogeneous_short_duration(rng_factory):
    """Same config as the ORIGINAL failing Layer 3 Poisson test (T=15)."""
    rng = rng_factory(200)
    model = PoissonProcess(RateKernelFactory.homogeneous_poisson())
    model.set_params(np.array([0.5]))
    scaffold = make_scaffold_dataset(num_fish=100, num_trials=6, duration_s=15.0)
    dataset = simulate_dataset_from_model(model, scaffold, rng)

    calib = model.combined_stream_gap_calibration_test(dataset)
    print(f"\nn_streams_tested={calib['n_streams_tested']}, "
          f"fisher_stat={calib['fisher_statistic']:.4f}, p={calib['combined_p_value']:.4g}")
    assert calib["combined_p_value"] > 0.01


def test_omr_forward_short_duration(rng_factory):
    """Same config as the ORIGINAL failing Layer 3 omr_forward test (T=3)."""
    rng = rng_factory(201)
    z_dip = float(logit_bounded(0.6, 0.995))
    model = PoissonProcess(RateKernelFactory.omr_forward())
    model.set_params(np.array([0.5, z_dip, 0.3]))
    scaffold = make_scaffold_dataset(num_fish=150, num_trials=10, duration_s=10.0)
    dataset = simulate_dataset_from_model(model, scaffold, rng)

    calib = model.combined_stream_gap_calibration_test(dataset)
    print(f"\nn_streams_tested={calib['n_streams_tested']}, "
          f"fisher_stat={calib['fisher_statistic']:.4f}, p={calib['combined_p_value']:.4g}")
    assert calib["combined_p_value"] > 0.01



def test_omr_forward_inspect_raw_residuals(rng_factory):
    rng = rng_factory(201)
    z_dip = float(logit_bounded(0.6, 0.995))
    model = PoissonProcess(RateKernelFactory.omr_forward())
    model.set_params(np.array([0.5, z_dip, 0.3]))
    scaffold = make_scaffold_dataset(num_fish=150, num_trials=10, duration_s=10.0)
    dataset = simulate_dataset_from_model(model, scaffold, rng)

    all_u = []
    for f_idx, t_idx, t_ev in dataset.iter_streams():
        if len(t_ev) < 2:
            continue
        t_sorted = np.sort(t_ev)
        probes = np.append(t_sorted, dataset.duration_s)
        cum = model.cumulative_integrated_intensity(probes, t_idx)
        diffs = np.diff(np.insert(cum, 0, 0.0))
        exact = diffs[:-1]
        all_u.extend((1.0 - np.exp(-exact)).tolist())

    all_u = np.array(all_u)
    print(f"\nn={len(all_u)}")
    print(f"mean={all_u.mean():.4f} (expect 0.5 for Uniform(0,1))")
    print(f"std={all_u.std():.4f} (expect ~0.289 for Uniform(0,1))")
    print(f"min={all_u.min():.4f}, max={all_u.max():.4f}")
    print(f"histogram (10 bins): {np.histogram(all_u, bins=10, range=(0,1))[0]}")


def test_hawkes_gap_based_fisher(rng_factory):
    rng = rng_factory(202)
    model = HawkesProcess(RateKernelFactory.homogeneous_poisson(), HistoryKernelFactory.exponential())
    model.set_params(np.array([0.4, 0.3, 2.0]))
    scaffold = make_scaffold_dataset(num_fish=80, num_trials=6, duration_s=20.0)
    dataset = simulate_dataset_from_model(model, scaffold, rng)
    calib = model.combined_stream_gap_calibration_test(dataset)
    print(f"\nn_streams={calib['n_streams_tested']}, p={calib['combined_p_value']:.4g}")
    assert calib["combined_p_value"] > 0.01


def test_renewal_gap_based_fisher(rng_factory):
    rng = rng_factory(203)
    model = RenewalProcess(RateKernelFactory.homogeneous_poisson(), RenewalKernelFactory.exponential_excitation())
    model.set_params(np.array([0.5, 1.0, 0.3]))
    scaffold = make_scaffold_dataset(num_fish=80, num_trials=6, duration_s=20.0)
    dataset = simulate_dataset_from_model(model, scaffold, rng)
    calib = model.combined_stream_gap_calibration_test(dataset)
    print(f"\nn_streams={calib['n_streams_tested']}, p={calib['combined_p_value']:.4g}")
    assert calib["combined_p_value"] > 0.01


def test_poisson_gap_based_fisher_still_works(rng_factory):
    """Sanity check: the gap-based method (now the universal default) should
    ALSO pass for Poisson, same as the order-statistic version did."""
    rng = rng_factory(200)
    model = PoissonProcess(RateKernelFactory.homogeneous_poisson())
    model.set_params(np.array([0.5]))
    scaffold = make_scaffold_dataset(num_fish=100, num_trials=6, duration_s=15.0)
    dataset = simulate_dataset_from_model(model, scaffold, rng)
    calib = model.combined_stream_gap_calibration_test(dataset)
    print(f"\nn_streams={calib['n_streams_tested']}, p={calib['combined_p_value']:.4g}")
    assert calib["combined_p_value"] > 0.01

def test_hawkes_meta_calibration_at_longer_duration(rng_factory):
    from scipy.stats import kstest

    """If the mild low-p skew at T=20 is the SAME finite-Lambda(duration_s)
    mechanism (just weaker), it should measurably shrink at a longer
    duration, holding the same rate/history params fixed."""
    for T in [20.0, 60.0, 150.0]:
        ps = []
        for seed in range(20):
            rng = rng_factory(3000 + seed)
            model = HawkesProcess(RateKernelFactory.homogeneous_poisson(), HistoryKernelFactory.exponential())
            model.set_params(np.array([0.4, 0.3, 2.0]))
            scaffold = make_scaffold_dataset(num_fish=80, num_trials=6, duration_s=T)
            dataset = simulate_dataset_from_model(model, scaffold, rng)
            calib = model.combined_stream_gap_calibration_test(dataset)
            ps.append(calib["combined_p_value"])
        ps = np.array(ps)
        stat, meta_p = kstest(ps, "uniform")
        print(f"\nT={T}: median_p={np.median(ps):.3f}, frac<0.05={np.mean(ps<0.05):.2f}, "
              f"meta_stat={stat:.4f}, meta_p={meta_p:.4f}")