import sleap_io as sio
from sleap.info.write_tracking_h5 import main as write_analysis
from sleap import PredictedInstance
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

def export_csv(input_slp: Path | str) -> Path:
    input_slp = Path(input_slp)
    parent = input_slp.parent

    labels = sio.load_file(input_slp)
    video = Path(labels.video.filename).with_suffix('.csv')
    output_csv = parent / video.name

    print(f'writing {output_csv}...', flush=True)
    write_analysis(labels=labels, output_path=output_csv, csv=True)
    return output_csv

def export(prediction_folder: Path):
    for file in prediction_folder.glob('*.slp'):
        export_csv(file)

def keep_user_labeled_only(input_slp, output_slp):
    labels = sio.load_file(input_slp)
    labels.labeled_frames = [
        f for f in labels.labeled_frames 
        if any([True for i in f.instances if not isinstance(i, PredictedInstance)])
    ]
    labels.suggestions = []
    labels.save(output_slp, embed=True)

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
    print(f"Writing overlay to {output_video}  (fps={fps_out}, size={w}x{h}, frames={total})", flush=True)

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
