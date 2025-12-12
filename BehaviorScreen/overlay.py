# overlay tracking (online/post-hoc) and stimulus on the video

from typing import Iterable, Callable
import pandas as pd

from BehaviorScreen.core import Stim, STIM_PARAMETERS
from BehaviorScreen.load import Directories, BehaviorData
import BehaviorScreen.stimulus as stimulus

PHASE_FUNCTIONS: dict[Stim, tuple[Callable, list[str]]] = {
    Stim.DARK: (stimulus.dark, ['']),
    Stim.BRIGHT: (stimulus.bright, ['']),
    Stim.PHOTOTAXIS: (stimulus.phototaxis, ['']),
    Stim.OMR: (stimulus.omr, ['omr_speed_mm_per_sec', 'omr_spatial_period_mm']),
    Stim.OKR: (stimulus.okr, ['okr_speed_deg_per_sec']),
    Stim.LOOMING: (stimulus.looming_constant_velocity_approach, [
        'looming_angle_start_deg', 
        'looming_angle_stop_deg', 
        'looming_size_to_speed_ratio_ms', 
        'looming_distance_to_screen_mm'
    ]),
    Stim.PREY_CAPTURE: (stimulus.prey_capture_arc_stimulus_cosine, [
        'prey_arc_start_deg',
        'prey_arc_stop_deg',
        'prey_speed_deg_s'
    ]),
    Stim.RAMP: (stimulus.ramp_linear, ['ramp_duration_sec'])
}

def timestamp_to_video_frame(video_timestamps: pd.DataFrame, timestamp: int) -> int:
    idx = (video_timestamps["timestamp"] - timestamp).abs().idxmin()
    return video_timestamps.loc[idx, "index"].item()
    
def stimulus_dataframe(
        behavior_data: BehaviorData, 
        keep_stim: Iterable[Stim] = [stim for stim in Stim]
    ) -> pd.DataFrame:

    last_timestamp = max(
        behavior_data.tracking['timestamp'].max(),
        behavior_data.video_timestamps['timestamp'].max()
    )

    rows = []
    for i, stim_dict in enumerate(behavior_data.stimuli):
        
        stim_select = int(stim_dict["stim_select"])
        if Stim(stim_select) not in keep_stim:
            continue
        
        start_timestamp = stim_dict["timestamp"]
        stop_timestamp = behavior_data.stimuli[i + 1]["timestamp"] if i + 1 < len(behavior_data.stimuli) else last_timestamp
        start_frame = timestamp_to_video_frame(behavior_data.video_timestamps, start_timestamp)
        stop_frame = timestamp_to_video_frame(behavior_data.video_timestamps, stop_timestamp)
    
        for frame in range(start_frame, stop_frame):

            row = {
                "stim_select": stim_select,
                "looming_center_mm_x": stim_dict.get("looming_center_mm", [pd.NA, pd.NA])[0],
                "foreground_color": str(stim_dict["foreground_color"]),
                "background_color": str(stim_dict["background_color"]),
            }

            frame_timestamp = behavior_data.video_timestamps.loc[behavior_data.video_timestamps['index']==frame, 'timestamp'].item()
            trial_time = 1e-9*(frame_timestamp - start_timestamp)
            phase_fun, arguments = PHASE_FUNCTIONS[Stim(stim_select)]
            kwargs = {
                'start_time_sec': stim_dict.get('start_time_sec'),
                'trial_time_sec': trial_time,
                'rollover_time_sec': 3600,
            }
            kwargs.update({arg: stim_dict.get(arg) for arg in arguments if arg != ''})
            phase = phase_fun(**kwargs)
            row.update({
                'frame': frame,
                'trial_time': trial_time,
                'phase': phase
            })

            for parameter in STIM_PARAMETERS:
                row.update({parameter: stim_dict.get(parameter, pd.NA)})

            rows.append(row)

    return pd.DataFrame(rows).set_index('frame')

def overlay(
        directories: Directories, 
        behavior_data: BehaviorData
    ) -> None:

    # videofile
    behavior_data.video

    # online tracking
    behavior_data.tracking

    # stimulus
    stim_df = stimulus_dataframe(behavior_data)

    behavior_data.video.reset_reader()
    num_frames =  behavior_data.video.get_number_of_frame()
    for frame_num in range(num_frames):
        frame = behavior_data.video.next_frame()
        
        if frame_num in stim_df.index:
            row = stim_df.loc[frame_num]
            stim_select = row.stim_select
        


