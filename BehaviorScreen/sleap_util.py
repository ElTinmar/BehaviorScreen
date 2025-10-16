import sleap_io as sio
from sleap import PredictedInstance
import numpy as np

def remove_all_predicted_instances(input_slp, output_slp):
    labels = sio.load_file(input_slp)
    labels.labeled_frames = [f for f in labels.labeled_frames if f and not isinstance(f.instances[0], PredictedInstance)]
    labels.save(output_slp)

def remove_unlabeled_suggested_frames(input_slp, output_slp):
    labels = sio.load_file(input_slp)
    indices = [(f.video, f.frame_idx) for f in labels.labeled_frames]
    labels.suggestions = [s for s in labels.suggestions if (s.video, s.frame_idx) in indices]
    labels.save(output_slp)

def replace_nans(input_slp, output_slp):
    labels = sio.load_file(input_slp)
    for frame in labels.labeled_frames:
        for inst in frame.instances:
            points = inst.points['xy']
            mask = np.isnan(points)
            if np.any(mask):
                print(inst)
                points[mask] = 0
    labels.save(output_slp)