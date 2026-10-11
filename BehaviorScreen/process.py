from pathlib import Path
from typing import NamedTuple, Optional

import cv2
import numpy as np
import pandas as pd
from video_tools import OpenCV_VideoReader

from BehaviorScreen.core import AGAROSE_WELL_DIMENSIONS, WellDimensions
from BehaviorScreen.protocol import PROTOCOL_SPEC


def get_background_image_safe(
    video: OpenCV_VideoReader, num_samples: int = 100
) -> np.ndarray:

    height = video.get_height()
    width = video.get_width()
    num_frames = video.get_number_of_frame()
    samples = np.linspace(0, num_frames, num_samples, endpoint=False, dtype=int)
    video_samples = np.zeros((height, width, num_samples), dtype=np.uint8)

    for i, frame_idx in enumerate(samples):
        video.seek_to(frame_idx)
        _, frame = video.next_frame()
        video_samples[:, :, i] = frame[:, :, 0]

    return np.median(video_samples, axis=2).astype(np.uint8)


class Circle(NamedTuple):
    center: tuple[int, int]
    radius: int


class Well(NamedTuple):
    x_mm: float
    y_mm: float
    radius_mm: float


def get_circle_hough(
    image: np.ndarray,
    pix_per_mm: float,
    tolerance_mm: float,
    well_dimensions: WellDimensions,
    blur_kernel_large_mm: float,
    blur_kernel_small_mm: float,
) -> Optional[Circle]:

    blur_kernel_large_px = int(pix_per_mm * blur_kernel_large_mm) | 1
    blur_kernel_small_px = int(pix_per_mm * blur_kernel_small_mm) | 1
    background = cv2.GaussianBlur(
        image, (blur_kernel_large_px, blur_kernel_large_px), 0
    )
    flat = cv2.divide(image, background, scale=255)
    blurred = cv2.GaussianBlur(flat, (blur_kernel_small_px, blur_kernel_small_px), 0)

    tolerance = int(tolerance_mm * pix_per_mm)
    circle_radius = int(pix_per_mm * well_dimensions["well_radius_mm"])
    well_distance = pix_per_mm * well_dimensions["distance_between_well_centers_mm"]

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=well_distance - tolerance,
        param1=50,
        param2=30,
        minRadius=circle_radius - tolerance,
        maxRadius=circle_radius + tolerance,
    )

    if circles is None:
        return None

    x, y, r = np.uint16(np.around(circles[0, 0]))
    return Circle((int(x), int(y)), int(r))


def save_detected_circle(path: Path, image: np.ndarray, circle: Circle) -> None:
    img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    cv2.circle(img_color, circle.center, circle.radius, (0, 255, 0), 2)
    cv2.circle(img_color, circle.center, 2, (0, 0, 255), 3)
    cv2.imwrite(str(path), img_color)


def detect_well(
    background: np.ndarray,
    pix_per_mm: float,
    png: Optional[Path] = None,
    tolerance_mm: float = 2,
    blur_kernel_large_mm: float = 2.5,
    blur_kernel_small_mm: float = 0.3,
) -> Optional[Well]:
    """Well position in the ROI, in mm. None if no circle is found."""
    circle = get_circle_hough(
        background,
        pix_per_mm,
        tolerance_mm,
        AGAROSE_WELL_DIMENSIONS,
        blur_kernel_large_mm,
        blur_kernel_small_mm,
    )
    if circle is None:
        return None
    if png is not None:
        save_detected_circle(png, background, circle)
    return Well(
        circle.center[0] / pix_per_mm,
        circle.center[1] / pix_per_mm,
        circle.radius / pix_per_mm,
    )


def get_epoch_trial_presentation(trials: pd.DataFrame) -> pd.DataFrame:
    """
    One row per expected (epoch_name, trial_num) of the protocol, plus any
    extra occurrences. `trials` is Recording.trials, optionally with extra
    per-trial columns (e.g. tracking quality) that are carried through.
    """
    rows = []
    for spec in PROTOCOL_SPEC:
        occurrences = trials[trials.epoch_name == spec.name]
        n_found = len(occurrences)

        for trial_num in range(max(spec.expected_repeats, n_found)):
            row = {
                "epoch_name": spec.name,
                "stim": int(spec.stim),
                "expected_repeats": spec.expected_repeats,
                "trial_num": trial_num,
            }

            if trial_num >= n_found:
                rows.append(
                    row
                    | {
                        "presented": False,
                        "matches_expected_parameters": np.nan,
                        "trial_duration_s": np.nan,
                        "start_timestamp": np.nan,
                        "stop_timestamp": np.nan,
                    }
                )
                continue

            trial = occurrences.iloc[trial_num]
            matches = all(
                trial[col] in allowed for col, allowed in (spec.parameters or {}).items()
            )
            # Stimulus parameters (colors, speeds, ...) are kept so that
            # screen.yaml parameter rules can select trials, not only events.
            rows.append(
                trial.drop(["stim_select", "epoch_name", "trial_num"]).to_dict()
                | row
                | {"presented": matches, "matches_expected_parameters": matches}
            )

    return pd.DataFrame(rows)


def compute_angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Signed angle (rad) from v1 to v2. v1: (N, 2); v2: (N, 2) or (2,)."""
    cos_angle = np.sum(v1 * v2, axis=-1)
    sin_angle = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
    return np.arctan2(sin_angle, cos_angle)


def get_target_time(trial_duration: float, target_fps: float) -> np.ndarray:
    num_points = int(target_fps * trial_duration)
    return np.linspace(0, trial_duration, num_points, endpoint=False)


def interpolate_ts(
    target_time: np.ndarray, time: np.ndarray, values: np.ndarray
) -> np.ndarray:
    return np.interp(target_time, time, values)
