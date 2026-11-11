"""
Dataset building that tolerates missing data, plus model-free per-fish
summary statistics used as a fallback when a parametric fit is flagged
(see model_qc.py).
"""
import copy
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from BehaviorScreen.point_process.analysis.model_config import CONTROL_CONDITIONS, control_loader
from BehaviorScreen.point_process.point_process.dataset import BehavioralDataLoader, PointProcessDataset
from BehaviorScreen.point_process.point_process.point_process import PointProcess
from BehaviorScreen.ablation_screen.tier1_config import PinnedArchitecture

Arm = Tuple[str, str]  # (line, condition)
RECORDING = r"(?P<plate>.*_(?P<day>\d{2}_[A-Za-z]{3}_\d{4})_\d{2}h\d{2}min\d{2}sec)_fish_\d+$"  # a fish's file name


def recording_days(fish_ids: np.ndarray) -> np.ndarray:
    """Each fish's recording day, from its file name ('' for names without one: a single day)."""
    return pd.Series(fish_ids, dtype=str).str.extract(RECORDING)["day"].fillna("").to_numpy()


def control_arms(arms: FrozenSet[Arm]) -> FrozenSet[Arm]:
    return frozenset(arm for arm in arms if arm[1] in CONTROL_CONDITIONS)


def safe_prepare_dataset(
    loader: BehavioralDataLoader, dataset_config: dict, context: str = ""
) -> Optional[PointProcessDataset]:
    """
    None when the data cannot form a dataset (no trials, no usable trial):
    one empty (line, condition, behavior) cell must not stop a batch run.
    The loader raises ValueError for every such data problem.
    """
    try:
        return loader.prepare_dataset(**dataset_config)
    except ValueError as e:
        warnings.warn(f"[{context}] skipped (data issue): {e}")
        return None


@dataclass(frozen=True)
class Contrast:
    """
    One row of the screen: test arm against reference arm, each a
    (line, condition). family keeps hypotheses apart for FDR (drug effect
    within an NTR line vs mutant against its wild-type background).
    """

    name: str
    family: str
    reference: Tuple[str, str]
    test: Tuple[str, str]


def ntr_contrast(line: str, vehicle: str = "vehicle", drug: str = "ronidazole") -> Contrast:
    return Contrast(line, "ntr", (line, vehicle), (line, drug))


def mutant_contrast(line: str, reference: Tuple[str, str] = ("WT", "danieau"), condition: str = "danieau") -> Contrast:
    return Contrast(line, "mutant", reference, (line, condition))


class Arms:
    """
    Every (line, condition, behavior) arm of the screen, prepared and fitted
    at most once per run and per pinning. dataset() is None when the data
    cannot form a dataset; model() raises the (cached) fit error.

    The arms of an NTR contrast keep only the fish of the days both were
    recorded: siblings of one clutch, compared within the day.

    process_factories pins each behavior to its control fit, on the fish of
    model_config.control_loader. A test of a contrast pins it instead to the
    control fit refitted without the contrast's own arms (architecture(behavior,
    exclude)): then the pinned values do not depend on which of its fish are
    labelled reference or test, and its permutation test is exact.
    """

    def __init__(
        self,
        loaders: Dict[str, BehavioralDataLoader],
        events: Dict[str, str],
        dataset_configs: Dict[str, dict],
        process_factories: Dict[str, Callable[[], PointProcess]],
        contrasts: Sequence[Contrast],
    ) -> None:
        """loaders: one per event table; events: the table of each behavior."""
        self.loaders, self.events = loaders, events
        self.dataset_configs = dataset_configs
        self.process_factories = process_factories
        self.contrasts = {c.name: c for c in contrasts}
        self._same_day = {arm: other for c in contrasts if c.family == "ntr" for arm, other in ((c.reference, c.test), (c.test, c.reference))}
        self._prepared: Dict[Tuple[str, str, str], Optional[PointProcessDataset]] = {}
        self._datasets: Dict[Tuple[str, str, str], Optional[PointProcessDataset]] = {}
        self._models: Dict[Tuple[str, str, str, FrozenSet[Arm]], Union[PointProcess, Exception]] = {}
        self._architectures: Dict[Tuple[str, FrozenSet[Arm]], PinnedArchitecture] = {}
        self._controls: Dict[str, Tuple[PointProcessDataset, List[Arm]]] = {}

    def control(self, behavior: str) -> Tuple[PointProcessDataset, List[Arm]]:
        """The control fish's dataset and each fish's arm."""
        if behavior not in self._controls:
            loader = self.loaders[self.events[behavior]]
            dataset = control_loader(loader).prepare_dataset(**self.dataset_configs[behavior])
            arm_of = loader.trials.drop_duplicates("file").set_index("file")[["line", "condition"]].astype(str)
            self._controls[behavior] = dataset, [tuple(arm_of.loc[f]) for f in dataset.fish_ids]
        return self._controls[behavior]

    def architecture(self, behavior: str, exclude: FrozenSet[Arm] = frozenset()) -> PinnedArchitecture:
        """
        process_factories[behavior], pinned to the control fit refitted without the control arms in `exclude`:
        from the control fit to convergence, the optimum without them whatever the start.
        """
        exclude = control_arms(exclude)
        if not exclude:
            return self.process_factories[behavior]
        if (behavior, exclude) not in self._architectures:
            dataset, arms = self.control(behavior)
            control = self.process_factories[behavior]
            refit = copy.deepcopy(control.fitted)
            refit.initial_guesses = [float(v) for v in control.fitted.params_]
            refit.fit(dataset.fish([i for i, arm in enumerate(arms) if arm not in exclude]))
            self._architectures[behavior, exclude] = PinnedArchitecture(refit, control.pinned)
        return self._architectures[behavior, exclude]

    def contrast_arms(self, contrast: str) -> FrozenSet[Arm]:
        c = self.contrasts[contrast]
        return frozenset({c.reference, c.test})

    def process_factory(self, contrast: str, behavior: str) -> PinnedArchitecture:
        """The pinned model a contrast's tests fit: architecture without its arms."""
        return self.architecture(behavior, self.contrast_arms(contrast))

    def _prepare(self, line: str, condition: str, behavior: str) -> Optional[PointProcessDataset]:
        key = (line, condition, behavior)
        if key not in self._prepared:
            self._prepared[key] = safe_prepare_dataset(
                self.loaders[self.events[behavior]].subset(line, [condition]), self.dataset_configs[behavior], "/".join(key)
            )
        return self._prepared[key]

    def dataset(self, line: str, condition: str, behavior: str) -> Optional[PointProcessDataset]:
        key = (line, condition, behavior)
        if key not in self._datasets:
            ds, other = self._prepare(*key), self._same_day.get((line, condition))
            partner = None if other is None else self._prepare(*other, behavior)
            if ds is not None and partner is not None:
                shared = np.isin(recording_days(ds.fish_ids), recording_days(partner.fish_ids))
                ds = ds.fish(np.flatnonzero(shared)) if shared.any() else None
            self._datasets[key] = ds
        return self._datasets[key]

    def model(self, line: str, condition: str, behavior: str, exclude: FrozenSet[Arm] = frozenset()) -> PointProcess:
        """The arm fitted with architecture(behavior, exclude)."""
        key = (line, condition, behavior, control_arms(exclude))
        if key not in self._models:
            try:
                model = self.architecture(behavior, key[3])()
                model.fit(self.dataset(line, condition, behavior))
                self._models[key] = model
            except Exception as exc:
                self._models[key] = exc
        result = self._models[key]
        if isinstance(result, Exception):
            raise result
        return result

    def datasets(self, contrast: str, behavior: str) -> Tuple[Optional[PointProcessDataset], Optional[PointProcessDataset]]:
        """(reference, test) datasets of a contrast."""
        c = self.contrasts[contrast]
        return self.dataset(*c.reference, behavior), self.dataset(*c.test, behavior)

    def models(self, contrast: str, behavior: str) -> Tuple[PointProcess, PointProcess]:
        """(reference, test) fitted models of a contrast, pinned without its arms."""
        c, exclude = self.contrasts[contrast], self.contrast_arms(contrast)
        return self.model(*c.reference, behavior, exclude), self.model(*c.test, behavior, exclude)


def per_fish_summary(dataset: PointProcessDataset) -> pd.DataFrame:
    """
    Model-free per-fish metrics used as a Tier-0 fallback when a parametric
    fit is flagged by model_qc.assess_group_fit:
      - rate_hz: total events / total observed exposure time
      - response_prob: fraction of observed trials with >=1 event
      - mean_first_latency_s: mean, across responding trials, of first-event time
    """
    n_fish = dataset.num_fish
    n_trials_obs = dataset.fish_trial_mask.sum(axis=1).astype(float)
    totals = np.zeros(n_fish)
    n_resp = np.zeros(n_fish)
    latency_sum = np.zeros(n_fish)
    latency_n = np.zeros(n_fish)

    for f_idx, t_idx, t_ev in dataset.iter_streams():
        totals[f_idx] += len(t_ev)
        if len(t_ev) > 0:
            n_resp[f_idx] += 1
            latency_sum[f_idx] += float(np.min(t_ev))
            latency_n[f_idx] += 1

    exposure_s = n_trials_obs * dataset.duration_s
    with np.errstate(divide="ignore", invalid="ignore"):
        rate_hz = np.where(exposure_s > 0, totals / exposure_s, np.nan)
        response_prob = np.where(n_trials_obs > 0, n_resp / n_trials_obs, np.nan)
        mean_latency = np.where(latency_n > 0, latency_sum / np.maximum(latency_n, 1), np.nan)

    df = pd.DataFrame({
        "fish_idx": np.arange(n_fish),
        "n_trials_observed": n_trials_obs,
        "rate_hz": rate_hz,
        "response_prob": response_prob,
        "mean_first_latency_s": mean_latency,
    })
    return df[n_trials_obs > 0].reset_index(drop=True)


def compare_fish_metric(df_a: pd.DataFrame, df_b: pd.DataFrame, metric: str) -> dict:
    """Rank-based fallback comparison (Mann-Whitney), used only for
    architecture-collapse/insufficient-information fallback reporting."""
    from scipy.stats import mannwhitneyu

    a = df_a[metric].dropna().values
    b = df_b[metric].dropna().values
    if len(a) < 3 or len(b) < 3:
        return {"metric": metric, "n_a": len(a), "n_b": len(b), "p_value": np.nan}
    stat, p = mannwhitneyu(a, b, alternative="two-sided")
    n_a, n_b = len(a), len(b)
    rank_biserial_r = float(1.0 - 2.0 * stat / (n_a * n_b))
    return {
        "metric": metric, "n_a": n_a, "n_b": n_b,
        "median_a": float(np.median(a)), "median_b": float(np.median(b)),
        "rank_biserial_r": rank_biserial_r, "p_value": float(p),
    }