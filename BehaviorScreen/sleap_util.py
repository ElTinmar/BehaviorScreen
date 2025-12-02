import sleap_io as sio
from sleap.info.write_tracking_h5 import main as write_analysis
from sleap import PredictedInstance
import numpy as np
import pandas as pd
import cv2
from pathlib import Path

from megabouts.tracking_data.convert_tracking import compute_angles_from_keypoints, interpolate_tail_keypoint
from megabouts.tracking_data import FullTrackingData
from megabouts.config import TailPreprocessingConfig
from megabouts.preprocessing import TailPreprocessing

from tqdm import tqdm

def compute_angle_between_vectors(v1, v2):
    dot_product = np.einsum("ij,ij->i", v1, v2)
    cos_angle = dot_product
    sin_angle = np.cross(v1, v2)
    angle = np.arctan2(sin_angle, cos_angle)
    return angle

def compute_eye_angle_from_keypoints(
        front_x, 
        front_y, 
        back_x, 
        back_y,
        head_x,
        head_y,
        swimbladder_x,
        swimbladder_y):
    
    v1 = np.array([head_x - swimbladder_x, head_y - swimbladder_y])
    v2 = np.array([front_x - back_x, front_y - back_y])
    return compute_angle_between_vectors(v1.T, v2.T)

def SLEAP_to_tracking(sleap_file: Path | str, mm_per_px: float):

    sleap_file = Path(sleap_file)

    if sleap_file.suffix == ".slp":
        sleap_file = export_csv(sleap_file)

    df = pd.read_csv(sleap_file)
    
    head_x = df["Head.x"] * mm_per_px
    head_y = df["Head.y"] * mm_per_px
    
    swimbladder_x = df["Swim_Bladder.x"] * mm_per_px
    swimbladder_y = df["Swim_Bladder.y"] * mm_per_px
    
    left_eye_front_x = df["Eye_Left_Front.x"] * mm_per_px
    left_eye_front_y = df["Eye_Left_Front.y"] * mm_per_px
    left_eye_back_x = df["Eye_Left_Back.x"] * mm_per_px
    left_eye_back_y = df["Eye_Left_Back.y"] * mm_per_px

    right_eye_front_x = df["Eye_Right_Front.x"] * mm_per_px
    right_eye_front_y = df["Eye_Right_Front.y"] * mm_per_px
    right_eye_back_x = df["Eye_Right_Back.x"] * mm_per_px
    right_eye_back_y = df["Eye_Right_Back.y"] * mm_per_px

    tail_x = df[[f"Tail_{i}.x" for i in range(9)]].values * mm_per_px
    tail_y = df[[f"Tail_{i}.y" for i in range(9)]].values * mm_per_px

    #tail_x_interp, tail_y_interp = interpolate_tail_keypoint(tail_x, tail_y, n_segments=10)
    tail_angles, head_yaw = compute_angles_from_keypoints(head_x, head_y, tail_x, tail_y)

    tracking_data = FullTrackingData.from_posture(
        head_x=head_x, head_y=head_y, head_yaw=head_yaw, tail_angle=tail_angles
    )
    left_eye_angle = compute_eye_angle_from_keypoints(
        left_eye_front_x, 
        left_eye_front_y, 
        left_eye_back_x, 
        left_eye_back_y,
        head_x,
        head_y,
        swimbladder_x,
        swimbladder_y
    ) 
    right_eye_angle = compute_eye_angle_from_keypoints(
        right_eye_front_x, 
        right_eye_front_y, 
        right_eye_back_x, 
        right_eye_back_y,
        head_x,
        head_y,
        swimbladder_x,
        swimbladder_y
    ) 
    eyes_df = pd.DataFrame({
        'left_eye_angle': left_eye_angle, 
        'right_eye_angle': right_eye_angle,
    })
    return tracking_data, eyes_df

def export_csv(input_slp: Path | str) -> Path:
    input_slp = Path(input_slp)
    parent = input_slp.parent

    labels = sio.load_file(input_slp)
    video = Path(labels.video.filename).with_suffix('.csv')
    output_csv = parent / video.name

    print(f'writing {output_csv}...')
    write_analysis(labels=labels, output_path=output_csv, csv=True)
    return output_csv

def export(prediction_folder: Path):
    for file in prediction_folder.glob('*.slp'):
        export_csv(file)

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

def overlay_video(
    slp_file: Path | str, 
    node_radius: int = 4,
    line_thickness: int = 1,
    alpha: float = 0.5,
    fps_out: float = 20
    ) -> None:

    # Load labels
    labels = sio.load_file(slp_file)  
    slp_file = Path(slp_file)
    parent = slp_file.parent
    video = Path(labels.video.filename)
    output_video = parent / f"prediction_{video.name}"

    # Pick the first video referenced in the labels
    if len(labels.videos) == 0:
        raise RuntimeError("No videos referenced inside the .slp file.")
    video_meta = labels.videos[0]
    # video_meta.filename can be a str or list[str]
    if isinstance(video_meta.filename, (list, tuple)):
        video_filename = video_meta.filename[0]
    else:
        video_filename = video_meta.filename

    video_filename = str(video_filename)
    if not Path(video_filename).exists():
        raise FileNotFoundError(f"Referenced video not found: {video_filename}")

    frame_map = {lf.frame_idx: lf for lf in labels.labeled_frames if lf.video == video_meta}

    if len(labels.skeletons) == 0:
        skeleton_edges = []
    else:
        skel = labels.skeletons[0]
        # map node name -> index
        node_to_idx = {n.name: i for i, n in enumerate(skel.nodes)}
        skeleton_edges = []
        for edge in skel.edges:
            try:
                src_idx = node_to_idx[edge.source.name]
                dst_idx = node_to_idx[edge.destination.name]
            except Exception:
                src_idx = int(edge[0])
                dst_idx = int(edge[1])
            skeleton_edges.append((src_idx, dst_idx))

    cap = cv2.VideoCapture(video_filename)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_filename}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or 'avc1' / 'H264' if available
    out = cv2.VideoWriter(output_video, fourcc, fps_out, (w, h))

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"Writing overlay to {output_video}  (fps={fps_out}, size={w}x{h}, frames={total})")

    def draw_instance(img, coord_array, color=(0, 255, 0)):

        for (x, y) in coord_array:
            if np.isnan(x) or np.isnan(y):
                continue
            cv2.circle(img, (int(round(x)), int(round(y))), node_radius, color, 1)

        for a, b in skeleton_edges:
            if a >= coord_array.shape[0] or b >= coord_array.shape[0]:
                continue
            xa, ya = coord_array[a]
            xb, yb = coord_array[b]
            if np.isnan(xa) or np.isnan(ya) or np.isnan(xb) or np.isnan(yb):
                continue
            cv2.line(img, (int(round(xa)), int(round(ya))),
                    (int(round(xb)), int(round(yb))),
                    color, line_thickness)


    for i in tqdm(range(len(frame_map))):
        ret, frame = cap.read()
        if not ret:
            break

        overlay = np.zeros_like(frame)

        if i in frame_map:
            labeled_frame = frame_map[i]
            for idx, inst in enumerate(labeled_frame.instances):
                try:
                    pts = inst.numpy()
                except Exception:
                    pts = np.array([[p.x, p.y] for p in inst.points]) 
                draw_instance(overlay, pts, color=(0,255,255))

        mask = overlay.any(axis=2)  
        frame[mask] = cv2.addWeighted(overlay[mask], alpha, frame[mask], 1 - alpha, 0)
        out.write(frame)

    cap.release()
    out.release()
    print("Done.")
