import pandas as pd
from typing import (
    Tuple, 
    List, 
    Iterable
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

def get_circles(
        image: np.ndarray, 
        pix_per_mm: float,
        tolerance_mm: float,
        well_dimensions: WellDimensions
    ) -> np.ndarray:

    tolerance = int(tolerance_mm * pix_per_mm) 

    circle_radius = int(pix_per_mm * well_dimensions['well_radius_mm'])
    min_radius = circle_radius - tolerance
    max_radius = circle_radius + tolerance

    well_distance = pix_per_mm * well_dimensions['distance_between_well_centers_mm']
    min_distance = well_distance - tolerance

    # TODO maybe adding a slight blurr to the images could help 
    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp = 1,
        minDist = min_distance,
        param1 = 50,
        param2 = 30,
        minRadius = min_radius,
        maxRadius = max_radius
    )[0]

    return circles

def save_detected_circles(
        directories: Directories,
        behavior_file: BehaviorFiles,
        image: np.ndarray, 
        circles: np.ndarray
    ) -> None:

    if len(image.shape) == 2:  # grayscale
        img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img_color = image.copy()

    circles = np.around(circles).astype(np.uint16)
    for x, y, radius in circles:
        center = (x, y)
        cv2.circle(img_color, center, radius, (0, 255, 0), 2)
        cv2.circle(img_color, center, 2, (0, 0, 255), 3)

    #TODO directories.results may not exist 
    result = directories.results / f"{behavior_file.metadata.stem}_WELLS.png"
    cv2.imwrite(str(result), img_color)
    
def circle_roi_index(circles: np.ndarray, rois: List[Tuple[int,int,int,int]]):
    indices = []
    for x, y, _ in circles:
        index = -1
        for i, (rx, ry, rw, rh) in enumerate(rois):
            if rx <= x < rx + rw and ry <= y < ry + rh:
                index = i
        
        if index == -1:
            RuntimeError('Circle does not belong to any ROI')
        
        indices.append(index)
    return np.argsort(indices)

def get_well_coords_mm(
        directories: Directories,
        behavior_file: BehaviorFiles,
        behavior_data: BehaviorData, 
        tolerance_mm = 2,
    ) -> np.ndarray:

    background_image = get_background_image(behavior_data)
    pix_per_mm = behavior_data.metadata['calibration']['pix_per_mm']
    circles = get_circles(
        background_image, 
        pix_per_mm, 
        tolerance_mm, 
        AGAROSE_WELL_DIMENSIONS
    )
    save_detected_circles(directories, behavior_file, background_image, circles)
    ind = circle_roi_index(circles, behavior_data.metadata['identity']['ROIs']) 
    circles_mm = 1/pix_per_mm * circles[ind,:]

    return circles_mm
    
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