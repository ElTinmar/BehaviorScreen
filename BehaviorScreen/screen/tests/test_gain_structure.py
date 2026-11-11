"""Dependence between the gains of fish of a line, a day and a plate, and the null simulated with it."""
from typing import List, Tuple

import numpy as np
import pytest

from BehaviorScreen.screen.line_tests.gain_structure import GainStructure, calibrate, dependence, fish_groups, log_rates
from BehaviorScreen.screen.line_tests.tier1_config import PinnedArchitecture
from BehaviorScreen.screen.line_tests.tier1_omnibus import fit_tier1, tier1_simulated_test
from BehaviorScreen.screen.control_models.model_config import gamma, poisson, zero_inflated
from BehaviorScreen.point_process.dataset import PointProcessDataset
from BehaviorScreen.point_process.poisson_process import RateKernelFactory


def pool(lines: int = 24, days: int = 2, plates: int = 3, fish: int = 4) -> Tuple[PointProcessDataset, List[str]]:
    """An empty dataset of lines x days x plates x fish, named as recordings, and each fish's line."""
    names = [(f"L{l}", f"0{p}_07dpf_L{l}_Wed_{10 + d}_Feb_2026_{10 + p}h00min00sec_fish_{w}")
             for l in range(lines) for d in range(days) for p in range(plates) for w in range(fish)]
    empty = np.array([], float)
    template = PointProcessDataset(event_times=empty, event_trials_idx=empty.astype(int), event_fish_idx=empty.astype(int),
                                   fish_trial_mask=np.ones((len(names), 6), bool), fish_ids=np.array([n for _, n in names]), duration_s=10.0, marks={})
    return template, [line for line, _ in names]


def test_groups_come_from_the_file_names() -> None:
    template, lines = pool(lines=2)
    groups = fish_groups(template.fish_ids, lines)
    assert groups.nunique().to_dict() == {"line": 2, "day": 4, "plate": 12}
    assert fish_groups(np.array(["a", "b"]), "X").to_dict("list") == {"line": ["X", "X"], "day": ["X/", "X/"], "plate": ["a", "b"]}


def test_latent_values_share_what_the_structure_says() -> None:
    template, lines = pool(lines=2)
    groups, rng = fish_groups(template.fish_ids, lines), np.random.default_rng(0)
    z = np.array([GainStructure(line=0.1, day=0.2, plate=0.3).latent(groups, rng) for _ in range(4000)])
    correlation = lambda i, j: np.corrcoef(z[:, i], z[:, j])[0, 1]
    assert z.std() == pytest.approx(1.0, abs=0.02)
    assert correlation(0, 1) == pytest.approx(0.6, abs=0.04)  # same plate
    assert correlation(0, 4) == pytest.approx(0.3, abs=0.04)  # same day, another plate
    assert correlation(0, 12) == pytest.approx(0.1, abs=0.04)  # same line, another day
    assert correlation(0, 24) == pytest.approx(0.0, abs=0.04)  # another line


@pytest.mark.parametrize("model, params", [
    (gamma(poisson(RateKernelFactory.spont())), [0.4, -0.02, 4.0]),
    (zero_inflated(poisson(RateKernelFactory.spont()), fit_c=True, c_init=0.1), None),
])
def test_dependent_gains_keep_their_distribution(model, params) -> None:
    model.set_params(np.array(model.initial_guesses if params is None else params))
    rng = np.random.default_rng(0)
    independent = model._fish_gains(20000, rng)
    latent = np.repeat(rng.standard_normal(10000), 2)  # pairs of fish with the same latent value
    dependent = model._fish_gains(20000, rng, latent)
    assert dependent[::2] == pytest.approx(dependent[1::2])
    assert np.quantile(dependent, [0.1, 0.5, 0.9]) == pytest.approx(np.quantile(independent, [0.1, 0.5, 0.9]), rel=0.05, abs=0.02)


def test_dependence_of_values_with_known_components() -> None:
    template, lines = pool(lines=60)
    groups, rng = fish_groups(template.fish_ids, lines), np.random.default_rng(0)
    values = GainStructure(line=0.2, day=0.2, plate=0.2).latent(groups, rng)
    found = dependence(values, groups)
    assert found["plate"] == pytest.approx(0.2 / 0.6, abs=0.06)  # plates within a day: of plate + fish
    assert found["day"] > 0.1 and found["line"] > 0.1


def test_calibrated_structure_reproduces_the_dependence_of_the_data() -> None:
    template, lines = pool()
    groups, rng = fish_groups(template.fish_ids, lines), np.random.default_rng(1)
    truth = gamma(poisson(RateKernelFactory.spont()))
    truth.set_params(np.array([0.4, -0.02, 4.0]))
    data = truth.simulate_dataset(template, rng, GainStructure(line=0.1, day=0.15, plate=0.25).latent(groups, rng))
    model = gamma(poisson(RateKernelFactory.spont()))
    model.fit(data)
    structure, table = calibrate(model, data, groups)
    assert table["simulated"].to_numpy() == pytest.approx(table["observed"].to_numpy(), abs=0.05)
    assert structure.plate > 0.1 and structure.line + structure.day + structure.plate <= 0.9
    assert dependence(log_rates(data), groups)["plate"] == pytest.approx(table.set_index("level").at["plate", "observed"])


def test_data_without_dependence_get_no_shared_gain() -> None:
    template, lines = pool()
    groups, rng = fish_groups(template.fish_ids, lines), np.random.default_rng(2)
    model = gamma(poisson(RateKernelFactory.spont()))
    model.set_params(np.array([0.4, -0.02, 4.0]))
    structure, _ = calibrate(model, model.simulate_dataset(template, rng), groups)
    assert max(structure.line, structure.day, structure.plate) < 0.1


def test_level_without_replicates_gets_no_share() -> None:
    template, lines = pool(lines=12, days=1)  # one day per line: nothing says how much days differ
    groups, rng = fish_groups(template.fish_ids, lines), np.random.default_rng(4)
    model = gamma(poisson(RateKernelFactory.spont()))
    model.set_params(np.array([0.4, -0.02, 4.0]))
    data = model.simulate_dataset(template, rng, GainStructure(plate=0.4).latent(groups, rng))
    structure, table = calibrate(model, data, groups)
    assert structure.day == 0.0 and structure.plate > 0.1 and np.isnan(table.set_index("level").at["day", "observed"])


def test_simulated_null_gives_a_p_value() -> None:
    template, _ = pool(lines=1, days=1, plates=6)
    model = gamma(poisson(RateKernelFactory.spont()))
    model.set_params(np.array([0.4, -0.02, 4.0]))
    data = model.simulate_dataset(template, np.random.default_rng(3))
    reference, test = data.fish(np.arange(12)), data.fish(np.arange(12, 24))
    architecture = PinnedArchitecture(model)
    deviance, _, _, pooled = fit_tier1(architecture, reference, test)
    result = tier1_simulated_test(architecture, pooled, reference, test, deviance, GainStructure(plate=0.3), n_sim=40, n_jobs=1)
    assert result["n_sim_success"] == 40 and 1 / 41 <= result["p_value_plate"] <= 1
