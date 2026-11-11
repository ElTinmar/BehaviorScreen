"""
Gaps between each selected bout and the bout before it (of any category or
side), for one experiment of model_config: how many fall under the dead time.

    python -m BehaviorScreen.screen.side_analysis.check_bout_gaps <data_root> prey_capture
"""
import sys
from pathlib import Path

import numpy as np

from BehaviorScreen.screen.control_models.model_config import BOUT_DEAD_TIME_S, EVENT_TABLES, get_model_config
from BehaviorScreen.screen.loader import BehavioralDataLoader

root, experiment = Path(sys.argv[1]), sys.argv[2]
config = get_model_config()[experiment]
loader = BehavioralDataLoader(root / "bouts_all.csv", [root / "valid_trials_all.csv"], [root / "qc_all.csv"], **EVENT_TABLES["bouts"])
ds = loader.prepare_dataset(**config["dataset"])

to_previous, all_gaps = [], []
for f, t, events in ds.iter_streams():
    bouts = ds.history_times(f, t)
    previous = np.searchsorted(bouts, events) - 1
    to_previous.append(np.where(previous >= 0, events - bouts[np.maximum(previous, 0)], np.nan))
    all_gaps.append(np.diff(bouts))
to_previous = np.concatenate(to_previous)
to_previous = to_previous[~np.isnan(to_previous)]
all_gaps = np.concatenate(all_gaps)

quantiles = [0.0, 0.001, 0.01, 0.05, 0.1]
print(f"{experiment}: {len(ds.event_times)} selected bouts, {len(ds.history.event_times)} bouts in all")
print(f"selected bout to previous bout: {np.mean(to_previous < BOUT_DEAD_TIME_S):.1%} under {BOUT_DEAD_TIME_S} s, "
      f"quantiles {dict(zip(quantiles, np.quantile(to_previous, quantiles).round(3)))}")
print(f"any bout to previous bout: {np.mean(all_gaps < BOUT_DEAD_TIME_S):.1%} under {BOUT_DEAD_TIME_S} s, "
      f"{np.sum(all_gaps == 0)} zero gaps, quantiles {dict(zip(quantiles, np.quantile(all_gaps, quantiles).round(3)))}")
