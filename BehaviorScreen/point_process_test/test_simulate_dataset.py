"""
Tests for PointProcess.simulate_dataset().

These are deterministic plumbing/unit tests rather than statistical
goodness-of-fit tests. They verify that full-dataset simulation:

- preserves the fish x trial experimental design;
- never generates events in inactive fish x trial cells;
- preserves dataset metadata;
- generates event times inside the observation window;
- is reproducible for a fixed random seed;
- draws one gain per fish and reuses it across all of that fish's trials.
"""

import numpy as np

from .conftest import make_scaffold_dataset

from BehaviorScreen.point_process.dataset import PointProcessDataset
from BehaviorScreen.point_process.poisson_process import (
    PoissonProcess,
    RateKernelFactory,
)


class GainRecordingPoisson(PoissonProcess):
    """
    Test spy that records every gain passed to simulate_stream().

    _draw_fish_gains() returns deterministic fish-specific gains:

        fish 0 -> 1
        fish 1 -> 2
        fish 2 -> 3
        ...

    simulate_stream() returns no events because this test only concerns
    whether simulate_dataset() reuses one gain across each fish's trials.
    """

    def __init__(self):
        super().__init__(
            RateKernelFactory.homogeneous_poisson()
        )
        self.recorded_calls = []
        self.draw_call_arguments = []

    def _draw_fish_gains(
        self,
        num_fish: int,
        n_sims: int,
        rng,
    ) -> np.ndarray:
        self.draw_call_arguments.append(
            (int(num_fish), int(n_sims))
        )

        gains = np.arange(
            1,
            num_fish + 1,
            dtype=float,
        )

        return np.repeat(
            gains[:, None],
            n_sims,
            axis=1,
        )

    def simulate_stream(
        self,
        dataset: PointProcessDataset,
        t_idx: int,
        gain: float,
        rng,
    ) -> np.ndarray:
        self.recorded_calls.append(
            {
                "trial_idx": int(t_idx),
                "gain": float(gain),
            }
        )

        return np.array([], dtype=float)


class TestSimulateDataset:

    def test_preserves_design_metadata_and_event_bounds(
        self,
        rng_factory,
    ):
        """
        The simulated dataset must retain the original observation design
        and metadata. Every generated event must belong to an active
        fish x trial cell and lie inside [0, duration_s).
        """
        scaffold_original = make_scaffold_dataset(
            num_fish=10,
            num_trials=4,
            duration_s=3.0,
            binning_dt=0.05,
        )

        fish_trial_mask = (
            scaffold_original.fish_trial_mask.copy()
        )

        # Introduce several inactive design cells.
        fish_trial_mask[0, 1] = False
        fish_trial_mask[3, 2] = False
        fish_trial_mask[7, 0] = False
        fish_trial_mask[9, 3] = False

        scaffold = PointProcessDataset(
            event_times=np.array([], dtype=float),
            event_trials_idx=np.array([], dtype=int),
            event_fish_idx=np.array([], dtype=int),
            fish_trial_mask=fish_trial_mask,
            fish_ids=scaffold_original.fish_ids.copy(),
            bout_name="test_bout",
            laterality="test_laterality",
            duration_s=scaffold_original.duration_s,
            binning_dt=scaffold_original.binning_dt,
        )

        model = PoissonProcess(
            RateKernelFactory.homogeneous_poisson()
        )
        model.set_params(np.array([2.0]))

        simulated = model.simulate_dataset(
            template=scaffold,
            rng=rng_factory(400),
        )

        np.testing.assert_array_equal(
            simulated.fish_trial_mask,
            scaffold.fish_trial_mask,
        )
        np.testing.assert_array_equal(
            simulated.fish_ids,
            scaffold.fish_ids,
        )

        assert simulated.duration_s == scaffold.duration_s
        assert simulated.binning_dt == scaffold.binning_dt
        assert simulated.bout_name == scaffold.bout_name
        assert simulated.laterality == scaffold.laterality

        assert simulated.event_times.dtype.kind == "f"
        assert simulated.event_trials_idx.dtype.kind in "iu"
        assert simulated.event_fish_idx.dtype.kind in "iu"

        assert np.all(simulated.event_times >= 0.0)
        assert np.all(
            simulated.event_times < scaffold.duration_s
        )

        assert np.all(simulated.event_fish_idx >= 0)
        assert np.all(
            simulated.event_fish_idx < scaffold.num_fish
        )

        assert np.all(simulated.event_trials_idx >= 0)
        assert np.all(
            simulated.event_trials_idx
            < scaffold.num_trials
        )

        # No generated event may belong to an inactive cell.
        for f_idx, t_idx in zip(
            simulated.event_fish_idx,
            simulated.event_trials_idx,
        ):
            assert scaffold.fish_trial_mask[f_idx, t_idx]

        # With this seed, rate, and exposure, the test should actually
        # exercise non-empty event-array construction.
        assert len(simulated.event_times) > 0

    def test_inactive_streams_are_not_simulated(
        self,
        rng_factory,
    ):
        """
        simulate_stream() should be called exactly once for every active
        fish x trial cell and never for an inactive cell.
        """
        num_fish = 3
        num_trials = 4

        mask = np.array(
            [
                [True, False, True, True],
                [False, True, True, False],
                [True, True, False, True],
            ],
            dtype=bool,
        )

        scaffold = PointProcessDataset(
            event_times=np.array([], dtype=float),
            event_trials_idx=np.array([], dtype=int),
            event_fish_idx=np.array([], dtype=int),
            fish_trial_mask=mask,
            fish_ids=np.array(
                [f"fish_{i}" for i in range(num_fish)]
            ),
            duration_s=2.0,
            binning_dt=0.05,
        )

        model = GainRecordingPoisson()
        model.set_params(np.array([0.5]))

        model.simulate_dataset(
            template=scaffold,
            rng=rng_factory(401),
        )

        assert len(model.recorded_calls) == int(mask.sum())

        expected_calls = []

        # PointProcessDataset.iter_streams() walks fish first, then trial.
        for f_idx in range(num_fish):
            for t_idx in range(num_trials):
                if mask[f_idx, t_idx]:
                    expected_calls.append(
                        {
                            "trial_idx": t_idx,
                            "gain": float(f_idx + 1),
                        }
                    )

        assert model.recorded_calls == expected_calls

    def test_one_gain_is_reused_across_each_fish_trials(
        self,
        rng_factory,
    ):
        """
        A fish-level frailty draw must be shared across all of that fish's
        trials, rather than redrawn independently for every stream.
        """
        num_fish = 3
        num_trials = 4

        scaffold = make_scaffold_dataset(
            num_fish=num_fish,
            num_trials=num_trials,
            duration_s=2.0,
        )

        model = GainRecordingPoisson()
        model.set_params(np.array([0.5]))

        model.simulate_dataset(
            template=scaffold,
            rng=rng_factory(402),
        )

        # simulate_dataset() should request exactly one draw per fish:
        # shape (num_fish, 1).
        assert model.draw_call_arguments == [
            (num_fish, 1)
        ]

        assert len(model.recorded_calls) == (
            num_fish * num_trials
        )

        # Calls are ordered by fish and then trial.
        gains_by_fish_trial = np.array(
            [
                call["gain"]
                for call in model.recorded_calls
            ],
            dtype=float,
        ).reshape(num_fish, num_trials)

        for f_idx in range(num_fish):
            expected_gain = float(f_idx + 1)

            np.testing.assert_array_equal(
                gains_by_fish_trial[f_idx],
                np.full(
                    num_trials,
                    expected_gain,
                ),
            )

    def test_simulation_is_reproducible_for_fixed_seed(
        self,
        rng_factory,
    ):
        """
        Two simulations from the same model, scaffold, and random seed
        should generate identical event arrays.
        """
        scaffold = make_scaffold_dataset(
            num_fish=8,
            num_trials=3,
            duration_s=4.0,
        )

        model = PoissonProcess(
            RateKernelFactory.homogeneous_poisson()
        )
        model.set_params(np.array([0.8]))

        simulated_a = model.simulate_dataset(
            template=scaffold,
            rng=rng_factory(403),
        )

        simulated_b = model.simulate_dataset(
            template=scaffold,
            rng=rng_factory(403),
        )

        np.testing.assert_array_equal(
            simulated_a.event_times,
            simulated_b.event_times,
        )
        np.testing.assert_array_equal(
            simulated_a.event_trials_idx,
            simulated_b.event_trials_idx,
        )
        np.testing.assert_array_equal(
            simulated_a.event_fish_idx,
            simulated_b.event_fish_idx,
        )
        np.testing.assert_array_equal(
            simulated_a.fish_trial_mask,
            simulated_b.fish_trial_mask,
        )

    def test_empty_simulation_returns_correct_array_types(
        self,
        rng_factory,
    ):
        """
        A dataset with no generated events must still contain correctly
        typed empty arrays and preserve the complete observation design.
        """
        scaffold = make_scaffold_dataset(
            num_fish=5,
            num_trials=3,
            duration_s=1.0,
        )

        # The spy always returns an empty stream.
        model = GainRecordingPoisson()
        model.set_params(np.array([0.5]))

        simulated = model.simulate_dataset(
            template=scaffold,
            rng=rng_factory(404),
        )

        assert simulated.event_times.shape == (0,)
        assert simulated.event_trials_idx.shape == (0,)
        assert simulated.event_fish_idx.shape == (0,)

        assert simulated.event_times.dtype.kind == "f"
        assert simulated.event_trials_idx.dtype.kind in "iu"
        assert simulated.event_fish_idx.dtype.kind in "iu"

        np.testing.assert_array_equal(
            simulated.fish_trial_mask,
            scaffold.fish_trial_mask,
        )

        # Empty streams must nevertheless still be present when iterating
        # over the active experimental design.
        streams = list(simulated.iter_streams())

        assert len(streams) == (
            scaffold.num_fish * scaffold.num_trials
        )
        assert all(
            len(t_ev) == 0
            for _, _, t_ev in streams
        )