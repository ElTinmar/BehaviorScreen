import pandas as pd
from typing import (
    Tuple, 
    Iterable,
    NamedTuple
)
import numpy as np
import cv2

from BehaviorScreen.load import (
    BehaviorData, 
    BehaviorFiles, 
    Directories
)
from BehaviorScreen.core import (
    Stim, 
    WellDimensions, 
    AGAROSE_WELL_DIMENSIONS, 
    STIM_PARAMETERS
)

def get_background_image(
        behavior_data: BehaviorData
    ) -> np.ndarray:

    return np.asarray(behavior_data.metadata['background']['image_ROI'], dtype=np.uint8)

def get_background_image_safe(
        behavior_data: BehaviorData, 
        num_samples: int = 100
    ) -> np.ndarray:

    height = behavior_data.video.get_height()
    width = behavior_data.video.get_width()
    num_frames = behavior_data.video.get_number_of_frame()
    samples = np.linspace(0, num_frames, num_samples, endpoint=False, dtype=int)
    video_samples = np.zeros((height, width, num_samples), dtype=np.uint8)

    for i, frame_idx in enumerate(samples):
        behavior_data.video.seek_to(frame_idx)
        _, frame = behavior_data.video.next_frame()
        video_samples[:,:,i] = frame[:,:,0]

    background_image = np.median(video_samples, axis=2).astype(np.uint8)
    return background_image

class Circle(NamedTuple):
    center: Tuple[int, int]
    radius: int

def get_circle(
        image: np.ndarray, 
        pix_per_mm: float,
        tolerance_mm: float,
        well_dimensions: WellDimensions
    ) -> Circle:

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    target_radius_mm = well_dimensions['well_radius_mm']

    if not contours:
        raise RuntimeError('circle not found')
    
    def distance(contour):
        perimeter_px = cv2.arcLength(cv2.convexHull(contour), True)
        perimeter_mm = perimeter_px/pix_per_mm
        radius_mm = perimeter_mm/(2*np.pi)
        return np.abs(radius_mm-target_radius_mm)

    best_contour = min(contours, key=distance)
    if len(best_contour) < 5:
        raise RuntimeError('circle not found')
    
    (x, y), radius = cv2.minEnclosingCircle(best_contour)

    if abs(radius/pix_per_mm - target_radius_mm) > tolerance_mm:
        raise RuntimeError('circle not found')

    return Circle((int(x), int(y)), int(radius))

def save_detected_circle(
        directories: Directories,
        behavior_file: BehaviorFiles,
        image: np.ndarray, 
        circle: Circle
    ) -> None:

    if len(image.shape) == 2:  # grayscale
        img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img_color = image.copy()

    cv2.circle(img_color, circle.center, circle.radius, (0, 255, 0), 2)
    cv2.circle(img_color, circle.center, 2, (0, 0, 255), 3)

    #TODO directories.results may not exist 
    result = directories.results / f"{behavior_file.metadata.stem}_WELLS.png"
    cv2.imwrite(str(result), img_color)

def get_well_coords_mm(
        directories: Directories,
        behavior_file: BehaviorFiles,
        behavior_data: BehaviorData, 
        tolerance_mm = 2,
    ) -> Tuple[float, float, float]:

    background_image = get_background_image(behavior_data)
    pix_per_mm = behavior_data.metadata['calibration']['pix_per_mm']
    
    try:
        circle = get_circle(
            background_image, 
            pix_per_mm, 
            tolerance_mm, 
            AGAROSE_WELL_DIMENSIONS
        )
    except:
        print(f"Error in get_circle: {behavior_file.metadata}")
        raise

    save_detected_circle(directories, behavior_file, background_image, circle)
    return (
        circle.center[0]/pix_per_mm, 
        circle.center[1]/pix_per_mm, 
        circle.radius/pix_per_mm
    ) 
    
def get_trials(
        behavior_data: BehaviorData, 
        keep_stim: Iterable[Stim] = [stim for stim in Stim]
    ) -> pd.DataFrame:

    last_timestamp = max(
        behavior_data.tracking['timestamp'].max(),
        behavior_data.video_timestamps['timestamp'].max()
    )

    rows = []
    for i, stim_dict in enumerate(behavior_data.stimuli):
        start_timestamp = stim_dict["timestamp"]
        stop_timestamp = behavior_data.stimuli[i + 1]["timestamp"] if i + 1 < len(behavior_data.stimuli) else last_timestamp
        stim_select = int(stim_dict["stim_select"])

        row = {
            "stim_select": stim_select,
            "start_timestamp": start_timestamp,
            "stop_timestamp": stop_timestamp,
            "looming_center_mm_x": stim_dict.get("looming_center_mm", [pd.NA, pd.NA])[0],
            "looming_center_mm_y": stim_dict.get("looming_center_mm", [pd.NA, pd.NA])[1],
            "foreground_color": str(stim_dict["foreground_color"]),
            "background_color": str(stim_dict["background_color"]),
        }
        for parameter in STIM_PARAMETERS:
            row.update({parameter: stim_dict.get(parameter, pd.NA)})

        if Stim(stim_select) in keep_stim:
            rows.append(row)

    return pd.DataFrame(rows)


def compute_angle_between_vectors(v1: np.ndarray, v2: np.ndarray):
    # v1, v2: (N, 2)
    cos_angle = np.sum(v1 * v2, axis=1)
    sin_angle = np.cross(v1, v2)
    angle = np.arctan2(sin_angle, cos_angle)
    return angle

def get_target_time(trial_duration, target_fps) -> np.ndarray:
    num_points = int(target_fps * trial_duration)
    return np.linspace(0, trial_duration, num_points, endpoint=False)

def interpolate_ts(target_time, time, values) -> np.ndarray:
    return np.interp(target_time, time, values)