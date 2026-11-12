import copy
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from BehaviorScreen.core import Laterality, Stim
from BehaviorScreen.point_process.dataset import PointProcessDataset
from BehaviorScreen.filters import apply_table_filters, exclude_qc_fish


# Usable trials shorter than this fraction of the analysis window are kept but
# reported: recorded durations fall a few frames short of the nominal epoch length.
SHORT_TRIAL_FRACTION = 0.95


class BehavioralDataLoader:
    """
    Events (bouts or saccades) and the trial table (valid_trials.csv) they
    belong to. Fish listed in any qc.csv are dropped from both, and so are
    recordings failing `recording_filters`; events failing `event_filters` are
    dropped (both in the rule syntax of filters.py, e.g. {"distance_center": {"<=": 10}}).

    A stream is one trial, (file, epoch_name, trial_num): epochs pooled into
    one condition (left/right, ipsi/contra) never share a stream. Their trial
    index is the order of presentation within the condition across the whole
    screen, so it equals trial_num for a single epoch and is the same in
    every subset.
    """

    def __init__(
        self,
        events_csv: Path,
        valid_trials_csvs: Sequence[Path],
        qc_csvs: Sequence[Path] = (),
        category_column: str = "category",
        category_names: Optional[Union[Sequence[str], Mapping[int, str]]] = None,
        tracking_column: str = "tracking_ok",
        event_filters: Optional[Mapping[str, Mapping[str, Any]]] = None,
        recording_filters: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> None:
        read = lambda path: pd.read_csv(path, dtype={"line": str, "condition": str}, low_memory=False)
        events = read(events_csv)
        trials = pd.concat([read(p) for p in valid_trials_csvs], ignore_index=True)
        for qc_csv in qc_csvs:
            events = exclude_qc_fish(events, qc_csv)
            trials = exclude_qc_fish(trials, qc_csv)
        filters = {"recording": recording_filters, "event": event_filters}
        events = apply_table_filters(events, filters, ("recording", "event"))
        trials = apply_table_filters(trials, filters, ("recording",))

        if trials.duplicated(["file", "epoch_name", "trial_num"]).any():
            raise ValueError("valid_trials lists the same (file, epoch_name, trial_num) twice")
        without_trials = set(events["file"]) - set(trials["file"])
        if without_trials:
            warnings.warn(f"{len(without_trials)} fish have events but no row in valid_trials; their events are ignored")

        events = events.dropna(subset=["epoch_name", "trial_num"])
        unmarked = events[category_column].isna() | events["laterality"].isna()
        if unmarked.any():
            counts = events.loc[unmarked, "epoch_name"].value_counts().to_string()
            warnings.warn(f"{unmarked.sum()} events without {category_column} or laterality ignored, by epoch:\n{counts}")
        self.events = events[~unmarked].astype({"trial_num": int, category_column: int, "laterality": int})
        self.trials = trials.astype({"presented": bool, tracking_column: bool})
        self.screen_trials = self.trials
        self.category_column, self.tracking_column = category_column, tracking_column
        self.category_names = None if category_names is None else dict(
            category_names.items() if isinstance(category_names, Mapping) else enumerate(category_names)
        )

    def _category_codes(self, category: Union[str, int, Sequence[Union[str, int]]]) -> List[int]:
        values = [category] if isinstance(category, (str, int)) else list(category)
        if any(isinstance(v, str) for v in values) and self.category_names is None:
            raise ValueError(f"category names unknown for {self.category_column}; use codes")
        codes = {name: code for code, name in (self.category_names or {}).items()}
        return [codes[v] if isinstance(v, str) else int(v) for v in values]

    @staticmethod
    def _select_trials(
        trials: pd.DataFrame,
        stim: Optional[Union[Stim, int]],
        epoch_name: Optional[Union[str, Sequence[str]]],
        trial_filters: Optional[Dict[str, Any]],
    ) -> pd.DataFrame:
        if stim is not None:
            trials = trials[trials["stim"] == int(stim)]
        elif epoch_name is not None:
            names = [epoch_name] if isinstance(epoch_name, str) else list(epoch_name)
            trials = trials[trials["epoch_name"].isin(names)]
        else:
            raise ValueError("Either 'stim' or 'epoch_name' must be provided.")
        for column, value in (trial_filters or {}).items():
            trials = trials[trials[column] == value]
        return trials

    def _condition_trials(
        self,
        stim: Optional[Union[Stim, int]],
        epoch_name: Optional[Union[str, Sequence[str]]],
        trial_filters: Optional[Dict[str, Any]],
    ) -> Tuple[pd.DataFrame, int]:
        """Expected trials of the condition with their trial index `trial`, and the number of trials."""
        keys = ["epoch_name", "trial_num"]
        presentation_order = (
            self._select_trials(self.screen_trials, stim, epoch_name, trial_filters)
            .groupby(keys)["epoch_idx"]
            .median()
            .dropna()
            .sort_values()
        )
        trials = self._select_trials(self.trials, stim, epoch_name, trial_filters)
        if trials.empty or presentation_order.empty:
            raise ValueError(
                f"No trials for stim={stim!r}, epoch_name={epoch_name!r}, "
                f"trial_filters={trial_filters!r}"
            )
        trial_index = pd.Series(
            np.arange(len(presentation_order)), index=presentation_order.index, name="trial"
        )
        return trials.merge(trial_index.reset_index(), on=keys), len(trial_index)

    def subset(
        self, line: Optional[str] = None, conditions: Optional[Sequence[str]] = None, exclude_lines: Sequence[str] = ()
    ) -> "BehavioralDataLoader":
        """
        The same screen restricted to a line and/or conditions, without the lines in
        exclude_lines (columns added by merge_csv). Trial indices stay those of the whole screen.
        """
        def keep(table: pd.DataFrame) -> pd.DataFrame:
            if "line" not in table.columns or "condition" not in table.columns:
                raise ValueError("subset needs 'line' and 'condition' columns (merge with merge_csv)")
            mask = np.ones(len(table), dtype=bool)
            if line is not None:
                mask &= table["line"].astype(str) == line
            if conditions is not None:
                mask &= table["condition"].isin(conditions)
            return table[mask & ~table["line"].astype(str).isin(exclude_lines)]

        subset = copy.copy(self)
        subset.events, subset.trials = keep(self.events), keep(self.trials)
        return subset

    def prepare_dataset(
        self,
        category: Optional[Union[str, int, Sequence[Union[str, int]]]] = None,
        laterality: Optional[Union[Laterality, Sequence[Laterality]]] = None,
        stim: Optional[Union[Stim, int]] = None,
        epoch_name: Optional[Union[str, Sequence[str]]] = None,
        trial_filters: Optional[Dict[str, Any]] = None,
        binning_dt: float = 0.02,
        t_start: float = 0.0,
        t_end: float = 24.0,
    ) -> PointProcessDataset:
        """
        Streams are the usable trials (presented and tracked), all treated as
        observed over [t_start, t_end); events are those within that window, of
        the requested categories and lateralities (None: all), marked with
        both.
        """
        codes = None if category is None else self._category_codes(category)
        lateralities = None if laterality is None else [int(l) for l in np.atleast_1d(laterality)]
        category_label = "all" if codes is None else "+".join(
            self.category_names.get(c, str(c)) if self.category_names is not None else str(c) for c in codes
        )
        laterality_label = "all" if lateralities is None else "+".join(Laterality(l).name for l in lateralities)
        label = f"{category_label}/{laterality_label}"

        trials, n_trials = self._condition_trials(stim, epoch_name, trial_filters)
        usable = trials["presented"] & trials[self.tracking_column]
        too_short = usable & (trials["trial_duration_s"] < SHORT_TRIAL_FRACTION * t_end)
        if too_short.any():
            short = trials.loc[too_short, ["file", "epoch_name", "trial_num", "trial_duration_s"]]
            warnings.warn(
                f"[{label}] {len(short)} usable trials shorter than {SHORT_TRIAL_FRACTION:.0%} of t_end={t_end}s "
                f"kept, as if observed until t_end:\n{short.to_string(index=False)}"
            )
        streams = trials.loc[usable, ["file", "epoch_name", "trial_num", "trial"]]
        if streams.empty:
            raise ValueError(f"[{label}] no usable trial")

        fish_ids = np.sort(streams["file"].unique())
        fish_map = {f_id: idx for idx, f_id in enumerate(fish_ids)}
        fish_trial_mask = np.zeros((len(fish_ids), n_trials), dtype=bool)
        fish_trial_mask[streams["file"].map(fish_map).to_numpy(), streams["trial"].to_numpy()] = True

        events = self.events.merge(streams, on=["file", "epoch_name", "trial_num"])
        events = events[(events["trial_time"] >= t_start) & (events["trial_time"] < t_end)]
        before = self._last_bouts_before(streams, t_start)

        print(
            f"[{label}] fish_trial_mask occupancy: {fish_trial_mask.mean():.1%} "
            f"({len(fish_ids)} fish x {n_trials} trials); "
            f"min fish/trial = {fish_trial_mask.sum(axis=0).min()}, "
            f"min trials/fish = {fish_trial_mask.sum(axis=1).min()}"
        )

        dataset = PointProcessDataset(
            event_times=(events["trial_time"] - t_start).to_numpy(float),
            event_trials_idx=events["trial"].to_numpy(int),
            event_fish_idx=events["file"].map(fish_map).to_numpy(int),
            fish_trial_mask=fish_trial_mask,
            fish_ids=fish_ids,
            fish_hours=self._fish_hours(fish_ids),
            duration_s=t_end - t_start,
            binning_dt=binning_dt,
            category_label=category_label,
            laterality=laterality_label,
            marks={
                "category": events[self.category_column].to_numpy(),
                "laterality": events["laterality"].to_numpy(),
            },
        )
        history = replace(dataset, **{
            "event_times": np.concatenate([dataset.event_times, (before["time"] - t_start).to_numpy(float)]),
            "event_trials_idx": np.concatenate([dataset.event_trials_idx, before["trial"].to_numpy(int)]),
            "event_fish_idx": np.concatenate([dataset.event_fish_idx, before["file"].map(fish_map).to_numpy(int)]),
            "marks": {key: np.concatenate([values, before[column].to_numpy()])
                      for (key, values), column in zip(dataset.marks.items(), (self.category_column, "laterality"))},
        })
        criteria = {"category": codes, "laterality": lateralities}
        return replace(dataset.select(**{key: values for key, values in criteria.items() if values is not None}), history=history)

    def _fish_hours(self, fish_ids: np.ndarray) -> Optional[np.ndarray]:
        """Each fish's recording start time (h), from the table's cos/sin_daytime (NaN for a fish without events); None without them."""
        if not {"cos_daytime", "sin_daytime"} <= set(self.events.columns):
            return None
        angle = self.events.groupby("file")[["cos_daytime", "sin_daytime"]].first().reindex(fish_ids)
        return np.mod(np.arctan2(angle.sin_daytime, angle.cos_daytime).to_numpy(), 2 * np.pi) * 24 / (2 * np.pi)

    def _last_bouts_before(self, streams: pd.DataFrame, t_start: float) -> pd.DataFrame:
        """
        Per stream (file, epoch_name, trial_num, trial), the fish's last bout of any
        epoch, category or side before the window (trial time t_start): its trial
        time ("time", negative relative to the window), trial and marks. Needs the
        events' absolute event_timestamp (ns); streams without an event give no start.
        """
        columns = ["file", "trial", "time", self.category_column, "laterality"]
        if "event_timestamp" not in self.events.columns:
            return pd.DataFrame(columns=columns)
        keys = ["file", "epoch_name", "trial_num"]
        bouts = self.events.assign(absolute=self.events["event_timestamp"] * 1e-9)
        starts = (bouts["absolute"] - bouts["trial_time"]).groupby([bouts[k] for k in keys]).median().rename("start").reset_index()
        windows = streams.merge(starts, on=keys).assign(window=lambda d: d["start"] + t_start).sort_values("window")
        last = pd.merge_asof(
            windows, bouts[["file", "absolute", self.category_column, "laterality"]].sort_values("absolute"),
            left_on="window", right_on="absolute", by="file", direction="backward", allow_exact_matches=False,
        ).dropna(subset=["absolute"])
        return last.assign(time=last["absolute"] - last["start"])[columns].astype({self.category_column: int, "laterality": int})
