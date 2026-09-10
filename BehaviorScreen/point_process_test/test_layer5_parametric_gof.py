import numpy as np
import pytest

from .conftest import (
    make_scaffold_dataset,
    simulate_dataset_from_model,
)

from BehaviorScreen.point_process.poisson_process import (
    PoissonProcess,
    RateKernelFactory,
)
from BehaviorScreen.point_process.hawkes_process import (
    HawkesProcess,
    HistoryKernelFactory,
)
from BehaviorScreen.point_process.survival_process import (
    SurvivalProcess,
    SurvivalKernelFactory,
)

class TestParametricGOFIntegrity:

    def test_homogeneous_poisson_bootstrap_outputs_are_consistent(
        self,
        rng_factory,
    ):
        rng = rng_factory(300)

        true_model = PoissonProcess(
            RateKernelFactory.homogeneous_poisson()
        )
        true_model.set_params(np.array([0.5]))

        scaffold = make_scaffold_dataset(
            num_fish=40,
            num_trials=5,
            duration_s=8.0,
        )

        dataset = simulate_dataset_from_model(
            true_model,
            scaffold,
            rng,
        )

        fitted = PoissonProcess(
            RateKernelFactory.homogeneous_poisson()
        )
        fitted.fit(dataset)

        result = fitted.parametric_gof_bootstrap(
            dataset,
            n_boot=30,
            seed=301,
            refit_n_starts=1,
            n_jobs=1,
        )

        assert result["n_requested"] == 30
        assert result["n_successful"] == 30
        assert result["n_failed"] == 0

        summary = result["summary"]
        assert {
            "calibration_ks",
            "calibration_cvm",
            "zero_stream_fraction",
            "variance_fish_total",
        }.issubset(set(summary["statistic"]))

        n_grid = len(result["u_grid"])

        for name in [
            "observed_calibration_cdf",
            "bootstrap_cdf_median",
            "bootstrap_cdf_lower",
            "bootstrap_cdf_upper",
            "simultaneous_cdf_lower",
            "simultaneous_cdf_upper",
        ]:
            assert len(result[name]) == n_grid

    def test_bootstrap_envelope_ordering(self, rng_factory):
        rng = rng_factory(302)

        generating = PoissonProcess(
            RateKernelFactory.homogeneous_poisson()
        )
        generating.set_params(np.array([0.6]))

        scaffold = make_scaffold_dataset(
            num_fish=30,
            num_trials=4,
            duration_s=6.0,
        )

        dataset = simulate_dataset_from_model(
            generating,
            scaffold,
            rng,
        )

        fitted = PoissonProcess(
            RateKernelFactory.homogeneous_poisson()
        )
        fitted.fit(dataset)

        result = fitted.parametric_gof_bootstrap(
            dataset,
            n_boot=30,
            seed=303,
            n_jobs=1,
        )

        lower = result["bootstrap_cdf_lower"]
        median = result["bootstrap_cdf_median"]
        upper = result["bootstrap_cdf_upper"]

        valid = (
            np.isfinite(lower)
            & np.isfinite(median)
            & np.isfinite(upper)
        )

        assert np.all(lower[valid] <= median[valid])
        assert np.all(median[valid] <= upper[valid])

    def test_simultaneous_envelope_matches_bootstrap_suprema(
        self,
        rng_factory,
    ):
        rng = rng_factory(304)

        generating = PoissonProcess(
            RateKernelFactory.homogeneous_poisson()
        )
        generating.set_params(np.array([0.5]))

        scaffold = make_scaffold_dataset(
            num_fish=30,
            num_trials=4,
            duration_s=6.0,
        )

        dataset = simulate_dataset_from_model(
            generating,
            scaffold,
            rng,
        )

        fitted = PoissonProcess(
            RateKernelFactory.homogeneous_poisson()
        )
        fitted.fit(dataset)

        result = fitted.parametric_gof_bootstrap(
            dataset,
            n_boot=40,
            seed=305,
            ci=95.0,
            n_jobs=1,
        )

        critical = result["simultaneous_critical_value"]
        assert np.isfinite(critical)
        assert critical >= 0.0

        u = result["u_grid"]
        lower = result["simultaneous_cdf_lower"]
        upper = result["simultaneous_cdf_upper"]

        valid = np.isfinite(lower) & np.isfinite(upper)

        np.testing.assert_allclose(
            lower[valid],
            np.clip(u[valid] - critical, 0.0, 1.0),
        )
        np.testing.assert_allclose(
            upper[valid],
            np.clip(u[valid] + critical, 0.0, 1.0),
        )

class TestParametricGOFSurvival:

    def test_censored_bootstrap_uses_km_support(
        self,
        rng_factory,
    ):
        rng = rng_factory(306)

        generating = SurvivalProcess(
            SurvivalKernelFactory.constant_hazard()
        )
        generating.set_params(np.array([0.2]))

        scaffold = make_scaffold_dataset(
            num_fish=80,
            num_trials=5,
            duration_s=2.0,
        )

        dataset = simulate_dataset_from_model(
            generating,
            scaffold,
            rng,
        )

        fitted = SurvivalProcess(
            SurvivalKernelFactory.constant_hazard()
        )
        fitted.fit(dataset)

        result = fitted.parametric_gof_bootstrap(
            dataset,
            n_boot=30,
            seed=307,
            min_km_at_risk=1,
            n_jobs=1,
        )

        observed = result["observed"]

        assert observed["has_censoring"]
        assert observed["n_censored"] > 0
        assert observed["n_exact"] > 0

        assert len(observed["km_residual_grid"]) == len(
            observed["km_survival"]
        )
        assert len(observed["km_survival"]) == len(
            observed["km_n_at_risk"]
        )

        # KM survival must be non-increasing.
        assert np.all(
            np.diff(observed["km_survival"]) <= 1e-12
        )

        # Risk set must also be non-increasing at failure knots.
        assert np.all(
            np.diff(observed["km_n_at_risk"]) <= 0
        )

        # Unsupported bootstrap tails must be NaN, not silently continued.
        contributors = result["bootstrap_cdf_contributors"]
        reliable = result["bootstrap_cdf_reliable_grid"]

        assert np.all(
            contributors[reliable]
            >= np.ceil(
                result["min_envelope_fraction"]
                * result["n_successful"]
            )
        )

@pytest.mark.slow
class TestParametricGOFPower:

    def test_poisson_rejects_strong_hawkes_data(
        self,
        rng_factory,
    ):
        """
        A homogeneous Poisson model should fail residual-distribution
        calibration when fitted to strongly self-exciting Hawkes data.

        The test uses the omnibus time-rescaling CvM statistic rather than
        max event-lag ACF. Hawkes excitation directly changes the waiting-time
        distribution, but it need not produce a large positive correlation
        between successive transformed intervals.
        """
        rng = rng_factory(308)

        generating = HawkesProcess(
            RateKernelFactory.homogeneous_poisson(),
            HistoryKernelFactory.exponential(),
        )

        # Branching ratio = 0.8 / 1.5 ~= 0.53: strong but subcritical.
        generating.set_params(
            np.array([
                0.35,  # baseline
                0.80,  # alpha
                1.50,  # beta
            ])
        )

        scaffold = make_scaffold_dataset(
            num_fish=100,
            num_trials=5,
            duration_s=15.0,
        )

        dataset = simulate_dataset_from_model(
            generating,
            scaffold,
            rng,
        )

        misspecified = PoissonProcess(
            RateKernelFactory.homogeneous_poisson()
        )
        misspecified.fit(dataset)

        result = (
            misspecified.parametric_gof_bootstrap(
                dataset,
                n_boot=60,
                seed=309,
                refit_n_starts=1,
                n_jobs=1,
            )
        )

        summary = result["summary"].set_index(
            "statistic"
        )

        p_cvm = float(
            summary.loc[
                "calibration_cvm",
                "p_upper",
            ]
        )

        assert p_cvm <= 0.05, (
            "The parametric-bootstrap time-rescaling diagnostic did not "
            "detect a strongly self-exciting Hawkes process fitted as "
            f"homogeneous Poisson; CvM p={p_cvm:.4f}."
        )

        
class TestParametricGOFNullBehavior:

    def test_correct_poisson_is_not_extremely_rejected(
        self,
        rng_factory,
    ):
        rng = rng_factory(310)

        generating = PoissonProcess(
            RateKernelFactory.homogeneous_poisson()
        )
        generating.set_params(np.array([0.5]))

        scaffold = make_scaffold_dataset(
            num_fish=80,
            num_trials=6,
            duration_s=10.0,
        )

        dataset = simulate_dataset_from_model(
            generating,
            scaffold,
            rng,
        )

        fitted = PoissonProcess(
            RateKernelFactory.homogeneous_poisson()
        )
        fitted.fit(dataset)

        result = fitted.parametric_gof_bootstrap(
            dataset,
            n_boot=60,
            seed=311,
            n_jobs=1,
        )

        summary = result["summary"].set_index("statistic")
        p_ks = summary.loc["calibration_ks", "p_upper"]

        # Deliberately weak regression assertion. This is not claiming that
        # every correctly specified random dataset must have p > .05.
        assert p_ks > 0.01