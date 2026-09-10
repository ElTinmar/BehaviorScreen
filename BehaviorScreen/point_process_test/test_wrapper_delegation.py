"""
Tests for wrapper delegation required by simulation and GOF diagnostics.

PartiallyFixedProcess owns a fully parameterized base process while exposing
only a subset of its parameters to the optimizer. Methods involving latent
frailty or model-specific compensators must delegate to the synchronized
base process rather than silently using PointProcess defaults.

These tests verify delegation of:

- parameter expansion/synchronization;
- fish-level gain simulation;
- model-specific time-rescaling residuals;
- stream_compensator_profile;
- full-dataset simulation;
- survival-process identity.
"""

import numpy as np

from .conftest import make_scaffold_dataset

from BehaviorScreen.point_process.poisson_process import (
    PoissonProcess,
    RateKernelFactory,
)
from BehaviorScreen.point_process.survival_process import (
    SurvivalProcess,
    SurvivalKernelFactory,
)
from BehaviorScreen.point_process.mixed_effects_process import (
    GammaMixedEffectsProcess,
)
from BehaviorScreen.point_process.partially_fixed_process import (
    PartiallyFixedProcess,
)


class TestPartiallyFixedParameterSynchronization:

    def test_set_params_expands_and_synchronizes_base_model(self):
        """
        Setting the free wrapper parameters should construct the complete
        parameter vector and write it onto the wrapped base process.
        """
        true_B = 0.5
        fixed_r = 3.0

        base_process = GammaMixedEffectsProcess(
            PoissonProcess(RateKernelFactory.homogeneous_poisson())
        )

        wrapped = PartiallyFixedProcess(
            base_process,
            fixed_values={
                "r_dispersion": fixed_r,
            },
        )

        assert wrapped.param_names == ["B"]

        wrapped.set_params(np.array([true_B]))

        np.testing.assert_allclose(
            wrapped.params_,
            np.array([true_B]),
        )

        np.testing.assert_allclose(
            wrapped.base_process.params_,
            np.array([true_B, fixed_r]),
        )

        assert wrapped.base_process.param_dict_["B"] == true_B
        assert wrapped.base_process.param_dict_["r_dispersion"] == fixed_r

        # GammaMixedEffectsProcess itself must synchronize its own
        # Poisson base process.
        assert wrapped.base_process.base_process.param_dict_["B"] == true_B


class TestPartiallyFixedGainDelegation:

    def test_draw_fish_gains_matches_base_process(self):
        """
        PartiallyFixedProcess._draw_fish_gains() must delegate to the
        frailty model. Falling back to PointProcess would return all ones
        and remove frailty from bootstrap simulations.
        """
        true_B = 0.5
        fixed_r = 2.5

        base_process = GammaMixedEffectsProcess(
            PoissonProcess(RateKernelFactory.homogeneous_poisson())
        )

        wrapped = PartiallyFixedProcess(
            base_process,
            fixed_values={
                "r_dispersion": fixed_r,
            },
        )
        wrapped.set_params(np.array([true_B]))

        wrapped_gains = wrapped._draw_fish_gains(
            num_fish=20,
            n_sims=3,
            rng=np.random.default_rng(500),
        )

        base_gains = wrapped.base_process._draw_fish_gains(
            num_fish=20,
            n_sims=3,
            rng=np.random.default_rng(500),
        )

        np.testing.assert_allclose(
            wrapped_gains,
            base_gains,
        )

        assert wrapped_gains.shape == (20, 3)

        # A Gamma frailty draw should not collapse to the PointProcess
        # default of gain=1 for every fish.
        assert not np.allclose(
            wrapped_gains,
            1.0,
        )

    def test_simulate_dataset_through_wrapper_uses_frailty(
        self,
    ):
        """
        Full-dataset simulation through the wrapper should be reproducible
        with direct simulation through the synchronized base process when
        both receive identical RNG states.
        """
        true_B = 0.6
        fixed_r = 3.0

        base_process = GammaMixedEffectsProcess(
            PoissonProcess(RateKernelFactory.homogeneous_poisson())
        )

        wrapped = PartiallyFixedProcess(
            base_process,
            fixed_values={
                "r_dispersion": fixed_r,
            },
        )
        wrapped.set_params(np.array([true_B]))

        scaffold = make_scaffold_dataset(
            num_fish=15,
            num_trials=4,
            duration_s=5.0,
        )

        wrapped_dataset = wrapped.simulate_dataset(
            template=scaffold,
            rng=np.random.default_rng(501),
        )

        base_dataset = wrapped.base_process.simulate_dataset(
            template=scaffold,
            rng=np.random.default_rng(501),
        )

        np.testing.assert_array_equal(
            wrapped_dataset.event_times,
            base_dataset.event_times,
        )
        np.testing.assert_array_equal(
            wrapped_dataset.event_trials_idx,
            base_dataset.event_trials_idx,
        )
        np.testing.assert_array_equal(
            wrapped_dataset.event_fish_idx,
            base_dataset.event_fish_idx,
        )
        np.testing.assert_array_equal(
            wrapped_dataset.fish_trial_mask,
            base_dataset.fish_trial_mask,
        )


class TestPartiallyFixedTimeRescalingDelegation:

    def test_stream_tau_values_match_frailty_base_process(
        self,
    ):
        """
        The wrapper must use GammaMixedEffectsProcess's predictable
        frailty-adjusted compensator rather than PointProcess's generic
        population-average compensator.
        """
        true_B = 0.5
        fixed_r = 2.0

        base_process = GammaMixedEffectsProcess(
            PoissonProcess(RateKernelFactory.homogeneous_poisson())
        )

        wrapped = PartiallyFixedProcess(
            base_process,
            fixed_values={
                "r_dispersion": fixed_r,
            },
        )
        wrapped.set_params(np.array([true_B]))

        scaffold = make_scaffold_dataset(
            num_fish=20,
            num_trials=4,
            duration_s=4.0,
        )

        dataset = wrapped.simulate_dataset(
            template=scaffold,
            rng=np.random.default_rng(502),
        )

        wrapped_tau = wrapped._stream_tau_values(dataset)
        base_tau = wrapped.base_process._stream_tau_values(dataset)

        assert set(wrapped_tau) == set(base_tau)

        for stream_key in wrapped_tau:
            wrapped_pairs = wrapped_tau[stream_key]
            base_pairs = base_tau[stream_key]

            assert len(wrapped_pairs) == len(base_pairs)

            wrapped_values = np.array(
                [value for value, _ in wrapped_pairs],
                dtype=float,
            )
            base_values = np.array(
                [value for value, _ in base_pairs],
                dtype=float,
            )

            wrapped_censored = np.array(
                [censored for _, censored in wrapped_pairs],
                dtype=bool,
            )
            base_censored = np.array(
                [censored for _, censored in base_pairs],
                dtype=bool,
            )

            np.testing.assert_allclose(
                wrapped_values,
                base_values,
            )
            np.testing.assert_array_equal(
                wrapped_censored,
                base_censored,
            )

    def test_complete_time_rescaling_matches_base_process(
        self,
    ):
        """
        Public time_rescaling() output should agree with the synchronized
        wrapped frailty process.
        """
        true_B = 0.5
        fixed_r = 3.0

        base_process = GammaMixedEffectsProcess(
            PoissonProcess(RateKernelFactory.homogeneous_poisson())
        )

        wrapped = PartiallyFixedProcess(
            base_process,
            fixed_values={
                "r_dispersion": fixed_r,
            },
        )
        wrapped.set_params(np.array([true_B]))

        scaffold = make_scaffold_dataset(
            num_fish=20,
            num_trials=4,
            duration_s=5.0,
        )

        dataset = wrapped.simulate_dataset(
            template=scaffold,
            rng=np.random.default_rng(503),
        )

        wrapped_result = wrapped.time_rescaling(
            dataset,
            acf_lags=5,
        )
        base_result = wrapped.base_process.time_rescaling(
            dataset,
            acf_lags=5,
        )

        np.testing.assert_allclose(
            wrapped_result["residuals"],
            base_result["residuals"],
        )
        np.testing.assert_array_equal(
            wrapped_result["censored"],
            base_result["censored"],
        )

        np.testing.assert_allclose(
            wrapped_result["residual_grid"],
            base_result["residual_grid"],
        )
        np.testing.assert_allclose(
            wrapped_result["survival_estimate"],
            base_result["survival_estimate"],
        )

        np.testing.assert_allclose(
            wrapped_result["acf"],
            base_result["acf"],
            equal_nan=True,
        )

        assert wrapped_result["n_rescaled"] == base_result["n_rescaled"]
        assert wrapped_result["n_exact"] == base_result["n_exact"]


class TestPartiallyFixedCompensatorProfileDelegation:

    def test_survival_compensator_profile_matches_base_process(
        self,
    ):
        """
        A partially fixed survival model must preserve the terminating,
        right-censored stream convention implemented by SurvivalProcess.
        """
        base_process = SurvivalProcess(
            SurvivalKernelFactory.gaussian_bump_baseline(
                t_init=0.3,
                t_bounds=(0.05, 0.6),
            )
        )

        fixed_mu = 0.3

        wrapped = PartiallyFixedProcess(
            base_process,
            fixed_values={
                "mu": fixed_mu,
            },
        )

        # Free order after fixing mu:
        # H, sigma, B
        wrapped.set_params(
            np.array(
                [
                    3.0,  # H
                    0.08,  # sigma
                    0.05,  # B
                ]
            )
        )

        duration_s = 1.0
        trial = 0

        for event_times in [
            np.array([], dtype=float),
            np.array([0.25], dtype=float),
            np.array([0.25, 0.70], dtype=float),
        ]:
            wrapped_profile = wrapped.stream_compensator_profile(
                event_times,
                trial,
                duration_s,
            )

            base_profile = wrapped.base_process.stream_compensator_profile(
                event_times,
                trial,
                duration_s,
            )

            wrapped_probes, wrapped_cum, wrapped_last_censored, wrapped_exposure = (
                wrapped_profile
            )
            base_probes, base_cum, base_last_censored, base_exposure = base_profile

            np.testing.assert_allclose(
                wrapped_probes,
                base_probes,
            )
            np.testing.assert_allclose(
                wrapped_cum,
                base_cum,
            )
            assert wrapped_last_censored == base_last_censored
            assert wrapped_exposure == base_exposure

    def test_is_survival_delegates_to_base_process(self):
        """
        Wrapper identity should preserve whether the underlying model is a
        terminating survival process.
        """
        survival_wrapped = PartiallyFixedProcess(
            SurvivalProcess(
                SurvivalKernelFactory.gaussian_bump_baseline(
                    t_init=0.3,
                    t_bounds=(0.05, 0.6),
                )
            ),
            fixed_values={
                "mu": 0.3,
            },
        )

        survival_wrapped.set_params(
            np.array(
                [
                    3.0,
                    0.08,
                    0.05,
                ]
            )
        )

        assert survival_wrapped.is_survival
        assert survival_wrapped.base_process.is_survival

        poisson_wrapped = PartiallyFixedProcess(
            PoissonProcess(RateKernelFactory.omr_forward()),
            fixed_values={
                "tau_dip": 0.4,
            },
        )

        # Free parameters are B and z_dip.
        poisson_wrapped.set_params(
            np.array(
                [
                    0.6,
                    0.5,
                ]
            )
        )

        assert not poisson_wrapped.is_survival
        assert not poisson_wrapped.base_process.is_survival
