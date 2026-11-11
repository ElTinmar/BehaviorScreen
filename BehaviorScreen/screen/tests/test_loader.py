import numpy as np
import pandas as pd
import pytest

from BehaviorScreen.screen.loader import BehavioralDataLoader


def test_loader_reads_recording_times():
    loader = object.__new__(BehavioralDataLoader)
    hours = {"a": 9.25, "b": 17.5}
    angle = lambda h: 2 * np.pi * h / 24
    loader.events = pd.DataFrame({"file": ["a", "a", "b"], "cos_daytime": [np.cos(angle(hours[f])) for f in "aab"],
                                  "sin_daytime": [np.sin(angle(hours[f])) for f in "aab"]})
    assert loader._fish_hours(np.array(["b", "a", "c"])) == pytest.approx([17.5, 9.25, np.nan], nan_ok=True)
    loader.events = loader.events.drop(columns=["cos_daytime", "sin_daytime"])
    assert loader._fish_hours(np.array(["a"])) is None
