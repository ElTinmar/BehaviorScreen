"""
One-off check (outside the repo): why an arm's fitted rate differs from its pooled data rate.
    python check_arm_fit.py [--model best-poisson] LINE CONDITION BEHAVIOR   e.g. 1042 vehicle spont_bright
"""
import argparse

import numpy as np

from BehaviorScreen.screen.line_tests.screen_setup import load_screen
from BehaviorScreen.screen.paths import CONTROL_FITS, data_root

parser = argparse.ArgumentParser()
parser.add_argument("line"); parser.add_argument("condition"); parser.add_argument("behavior")
parser.add_argument("--model", default="best-poisson")
args = parser.parse_args()

screen = load_screen(data_root(None), CONTROL_FITS, args.model)
arms, contrast = screen.arms, next(c for c in screen.arms.contrasts.values() if (args.line, args.condition) in (c.reference, c.test))
ds = arms.dataset(args.line, args.condition, args.behavior)
model = arms.model(args.line, args.condition, args.behavior, arms.contrast_arms(contrast.name))
trials = ds.fish_trial_mask.sum(axis=1)
events = np.bincount(ds.event_fish_idx, minlength=ds.num_fish)
rates = events / (trials * ds.duration_s)
print(f"{model.name}\nfitted: {dict(zip(model.param_names, np.round(model.params_, 4)))}")
print(f"fixed (pinned): { {k: round(v, 4) for k, v in getattr(model, 'fixed_values', {}).items()} }")
for f in np.argsort(-rates):
    print(f"  {ds.fish_ids[f]}: {trials[f]:3d} trials, {events[f]:5d} events, {rates[f]:.3f} Hz")
print(f"pooled rate (events / exposure): {events.sum() / (trials.sum() * ds.duration_s):.3f} Hz")
print(f"mean of per-fish rates:          {rates.mean():.3f} Hz")
print(f"model expected rate (mean):      {np.average(model.compute_expected_rate(ds).mean(axis=1), weights=ds.n_fish_per_trial):.3f} Hz")
grid = np.linspace(0.5, 2.0, 7) * model.params_[0]
print("NLL around the fitted first free parameter:", {round(float(b), 3): round(float(model.nll(np.r_[b, model.params_[1:]], ds)), 2) for b in grid})
