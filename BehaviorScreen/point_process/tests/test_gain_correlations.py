"""gain_correlations: correlations of posterior fish gains across behaviors, corrected for shrinkage."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import gamma, norm

from .conftest import reference_dataset

from BehaviorScreen.point_process.frailty import GammaGain, GammaMixedEffectsProcess
from BehaviorScreen.point_process.frailty_analysis import collect_fish_gains, gain_correlations, plot_gain_correlations
from BehaviorScreen.point_process.poisson_process import PoissonProcess, RateKernelFactory

R = 2.2
EXPECTED_EVENTS = {"spont": 300.0, "prey": 30.0, "flash": 3.0}  # per fish at gain 1


def simulated_gains(n_fish: int, rho: float, seed: int) -> tuple:
    """Gamma(R, R) gains correlated across behaviors (Gaussian copula), their Poisson counts and posterior means."""
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal(np.zeros(3), (1 - rho) * np.eye(3) + rho, n_fish)
    true = pd.DataFrame(gamma.ppf(norm.cdf(z), R, scale=1 / R), columns=list(EXPECTED_EVENTS))
    posterior = GammaGain(R, R)
    rows = [pd.DataFrame({"fish_id": np.arange(n_fish), "behavior": b, "gain_variance": 1 / R,
                          "estimated_gain": posterior.posterior(rng.poisson(true[b] * S), np.full(n_fish, S))["estimated_gain"]})
            for b, S in EXPECTED_EVENTS.items()]
    return true, pd.concat(rows, ignore_index=True)


def test_correction_recovers_the_gains_correlation():
    true, gains = simulated_gains(3000, 0.6, seed=1)
    result = gain_correlations(gains, n_boot=200)
    target = true.corr()
    reliability = result["reliability"]["reliability"]
    assert min(reliability["spont"], reliability["prey"]) > 0.9 and reliability["flash"] < 0.7  # few events: strong shrinkage
    assert result["observed"].loc["spont", "flash"] < target.loc["spont", "flash"] - 0.1  # attenuated
    for a, b in (("spont", "prey"), ("spont", "flash"), ("prey", "flash")):
        assert result["corrected"].loc[a, b] == pytest.approx(target.loc[a, b], abs=0.06), (a, b)
        assert result["corrected_low"].loc[a, b] < target.loc[a, b] < result["corrected_high"].loc[a, b], (a, b)


def test_independent_gains_stay_uncorrelated():
    _, gains = simulated_gains(3000, 0.0, seed=2)
    corrected = gain_correlations(gains, n_boot=50)["corrected"]
    assert np.abs(corrected.to_numpy()[np.triu_indices(3, 1)]).max() < 0.1


def test_from_fitted_models():
    """collect_fish_gains on fitted models feeds gain_correlations and its figure."""
    data = reference_dataset(num_fish=60)
    models = {}
    for name, kernel in (("a", RateKernelFactory.homogeneous_poisson), ("b", RateKernelFactory.homogeneous_poisson)):
        model = GammaMixedEffectsProcess(PoissonProcess(kernel()))
        model.fit(data)
        models[name] = (model, data)
    gains = collect_fish_gains(models)
    assert set(gains.columns) == {"fish_id", "behavior", "estimated_gain", "gain_variance"} and len(gains) == 120
    result = gain_correlations(gains, n_boot=20)
    assert result["observed"].loc["a", "b"] == pytest.approx(1.0)  # the same data twice
    assert result["n_fish"].loc["a", "b"] == 60
    plot_gain_correlations(result, "test")
