"""
Frames to label next for the full-fish pose model: those Lightning Pose most
likely gets wrong, spread over lines, recording days and fish.

Per frame of each fish's predictions (lightning_pose/<video>.csv), the error score
is the largest of
    likelihood   1 - the lowest keypoint likelihood
    segments     largest relative change of a head-to-tail segment from the fish's median length
    eyes         same for the eye lengths
    eye side     1 when an eye lies on the other side of the heading than usual for that fish
    spike        largest one-frame excursion of a keypoint (its offset from the mean of the
                 previous and next frames) beyond that of its skeleton neighbours, in median
                 tail-segment lengths: a fast tail moves neighbours together, an error moves one
Each fish keeps its worst frames at least --min-gap-s apart, from bouts (tail moving)
and from rest separately. Frames are then taken in rounds over the line x day groups
(see select), at most --per-fish per fish, --bout-fraction from bouts. By default about
twice as many fish are read per group as frames it would get, so each group has a choice.

Outputs:
    <output>.csv       the frames, their scores and what drove them
    <output>.slp       SLEAP project pointing to the videos (neighbouring frames help to label):
                       the frames as suggestions, each with Lightning Pose's prediction to
                       correct. Merge it into the labelling project (File > Merge Into Project),
                       label, export a labels package (embeds the labelled frames) and convert
                       it with slp2lp as usual

Runs where sleap_io is installed (the SLEAP environment):
    python select_frames.py DATA --output next_labels --skeleton-from model.pkg.slp
        [--labeled CollectedData.csv] [--frames 200] [--per-fish 3]
"""
import argparse
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import sleap_io as sio

KEYPOINTS = ["Head", "Eye_Left_Front", "Eye_Left_Back", "Eye_Right_Front", "Eye_Right_Back", "Swim_Bladder"] + [f"Tail_{i}" for i in range(10)]
AXIS = [KEYPOINTS.index(k) for k in ["Head", "Swim_Bladder"] + [f"Tail_{i}" for i in range(10)]]
EYES = [(KEYPOINTS.index(f"Eye_{s}_Front"), KEYPOINTS.index(f"Eye_{s}_Back")) for s in ("Left", "Right")]
TAIL = AXIS[2:]
# Skeleton neighbours of each keypoint: adjacent points along head-to-tail, the head for the eyes
NEIGHBOURS = [[AXIS[j] for j in (i - 1, i + 1) if 0 <= j < len(AXIS)] if k in AXIS else [AXIS[0]]
              for k in range(len(KEYPOINTS)) for i in [AXIS.index(k) if k in AXIS else -1]]
NAME = re.compile(r"^\d{2}_\d+dpf_(?P<line>[^_]+)_[A-Za-z]{3}_(?P<day>\d{2}_[A-Za-z]{3}_\d{4})_")
BOUT_SPEED = 4.0  # tail moving: mean tail-point speed above this multiple of the fish's median speed
BOUT_PAD = 5  # frames around a fast frame that still count as bout


def load_pose(csv: Path) -> Tuple[np.ndarray, np.ndarray]:
    """(T, K, 2) positions and (T, K) likelihoods of the KEYPOINTS."""
    pose = pd.read_csv(csv, header=[0, 1, 2], index_col=0).droplevel(0, axis=1)
    xy = np.stack([pose[k][["x", "y"]].to_numpy(float) for k in KEYPOINTS], 1)
    return xy, np.stack([pose[k]["likelihood"].to_numpy(float) for k in KEYPOINTS], 1)


def length_deviation(lengths: np.ndarray) -> np.ndarray:
    return np.nanmax(np.abs(lengths / np.nanmedian(lengths, axis=0) - 1), axis=1)


def frame_scores(xy: np.ndarray, likelihood: np.ndarray) -> pd.DataFrame:
    segments = np.linalg.norm(np.diff(xy[:, AXIS], axis=1), axis=2)
    eye_lengths = np.stack([np.linalg.norm(xy[:, f] - xy[:, b], axis=1) for f, b in EYES], 1)
    heading = xy[:, AXIS[0]] - xy[:, AXIS[1]]
    cross = lambda u, v: u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    side = np.stack([np.sign(cross(heading, (xy[:, f] + xy[:, b]) / 2 - xy[:, AXIS[0]])) for f, b in EYES], 1)
    excursion = xy[1:-1] - (xy[:-2] + xy[2:]) / 2
    neighbours = np.stack([excursion[:, n].mean(axis=1) for n in NEIGHBOURS], 1)
    spike = np.zeros(len(xy))
    spike[1:-1] = np.nanmax(np.linalg.norm(excursion - neighbours, axis=2), axis=1) / np.nanmedian(segments[:, 2:])
    scores = pd.DataFrame({
        "likelihood": 1 - np.nanmin(likelihood, axis=1),
        "segments": length_deviation(segments),
        "eyes": length_deviation(eye_lengths),
        "eye side": (side != np.sign(np.nanmedian(side, axis=0))).any(axis=1).astype(float),
        "spike": spike,
    }).fillna(1.0)
    speed = np.zeros(len(xy))
    speed[1:] = np.nanmean(np.linalg.norm(np.diff(xy[:, TAIL], axis=0), axis=2), axis=1)
    moving = pd.Series(speed > BOUT_SPEED * np.nanmedian(speed)).rolling(2 * BOUT_PAD + 1, center=True, min_periods=1).max()
    return scores.assign(score=scores.max(axis=1), reason=scores.idxmax(axis=1), in_bout=moving.to_numpy(bool))


def worst_frames(scores: pd.Series, gap: int, k: int) -> List[int]:
    """The k highest-scoring frames at least `gap` frames apart."""
    picked: List[int] = []
    for frame in scores.sort_values(ascending=False).index:
        if len(picked) == k:
            break
        if all(abs(frame - p) >= gap for p in picked):
            picked.append(frame)
    return picked


def fish_candidates(job: Tuple[Path, Path, float, int]) -> pd.DataFrame:
    """A fish's worst frames from bouts and from rest, with their predicted keypoints."""
    csv, video, min_gap_s, k = job
    xy, likelihood = load_pose(csv)
    scores = frame_scores(xy, likelihood)
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    gap = max(1, round(min_gap_s * fps))
    frames = [f for bout in (True, False) for f in worst_frames(scores.loc[scores["in_bout"] == bout, "score"], gap, k)]
    picked = scores.loc[frames].rename_axis("frame").reset_index()
    return picked.assign(fish=csv.stem, csv=csv, video=str(video), time_s=picked["frame"] / fps,
                         points=[xy[f] for f in frames], point_scores=[likelihood[f] for f in frames])


def experiments(data: Path, results: str, lightning_pose: str) -> List[Path]:
    """Folders under data (itself included) holding both a results and a lightning_pose folder."""
    found = []
    for folder, subfolders, _ in os.walk(data):
        if results in subfolders and lightning_pose in subfolders:
            found.append(Path(folder))
            subfolders.clear()  # an experiment's own folders hold no further experiments
    return sorted(found)


def recordings(roots: List[Path], results: str, lightning_pose: str, labeled: set) -> pd.DataFrame:
    """Fish with predictions and a video, not already labelled, with their line and day."""
    rows = []
    for root in roots:
        for csv in sorted((root / lightning_pose).glob("*.csv")):
            match, video = NAME.match(csv.stem), root / results / f"{csv.stem}.mp4"
            if match and video.exists() and csv.stem not in labeled:
                rows.append({"csv": csv, "video": video, "line": match["line"].upper(), "day": match["day"]})
    return pd.DataFrame(rows, columns=["csv", "video", "line", "day"])


def select(candidates: pd.DataFrame, n: int, per_fish: int, bout_fraction: float) -> pd.DataFrame:
    """
    Rounds over the line x day groups, each giving one frame per round: its least-sampled
    fish's worst frame (the worst among such fish). Within a round, the group offering the
    worst frame goes first, so when frames run short the worst-tracked groups are served.
    """
    pools = {fish: c.sort_values("score", ascending=False) for fish, c in candidates.groupby("csv")}  # a fish is its predictions file
    group_of = {fish: (c["line"].iloc[0], c["day"].iloc[0]) for fish, c in pools.items()}
    taken = dict.fromkeys(pools, 0)
    picks: List[pd.Series] = []
    served: set = set()  # groups that gave a frame this round
    while len(picks) < n:
        want_bout = sum(p["in_bout"] for p in picks) < bout_fraction * (len(picks) + 1)
        offers: Dict[tuple, tuple] = {}  # group: (fish's frames so far, -score, fish, frame)
        for fish, pool in pools.items():
            group = group_of[fish]
            if taken[fish] == per_fish or pool.empty or group in served:
                continue
            kind = pool[pool["in_bout"] == want_bout]
            row = (kind if len(kind) else pool).iloc[0]
            if group not in offers or (taken[fish], -row["score"]) < offers[group][:2]:
                offers[group] = (taken[fish], -row["score"], fish, row)
        if not offers:
            if not served:
                break
            served.clear()
            continue
        _, _, fish, row = min(offers.values(), key=lambda offer: offer[1])
        pools[fish] = pools[fish].drop(row.name)
        taken[fish] += 1
        served.add(group_of[fish])
        picks.append(row)
    return pd.DataFrame(picks).reset_index(drop=True)


def predicted_instance(points: np.ndarray, scores: np.ndarray, skeleton: sio.Skeleton) -> sio.PredictedInstance:
    """Lightning Pose's prediction as a SLEAP prediction; the from_numpy signature changed across sleap_io versions."""
    try:
        return sio.PredictedInstance.from_numpy(np.column_stack([points, scores]), skeleton=skeleton, score=float(np.nanmean(scores)))
    except TypeError:
        return sio.PredictedInstance.from_numpy(points=points, point_scores=scores, instance_score=float(np.nanmean(scores)), skeleton=skeleton)


def write_slp(selection: pd.DataFrame, skeleton: sio.Skeleton, path: Path) -> None:
    order = [skeleton.node_names.index(k) for k in KEYPOINTS]  # LP keypoints in the skeleton's node order
    videos = {v: sio.load_video(v) for v in selection["video"].unique()}
    frames = []
    for row in selection.itertuples():
        points, scores = np.full((len(order), 2), np.nan), np.full(len(order), np.nan)
        points[order], scores[order] = row.points, row.point_scores
        frames.append(sio.LabeledFrame(video=videos[row.video], frame_idx=int(row.frame), instances=[predicted_instance(points, scores, skeleton)]))
    labels = sio.Labels(videos=list(videos.values()), skeletons=[skeleton], labeled_frames=frames)
    labels.suggestions = [sio.SuggestionFrame(video=f.video, frame_idx=f.frame_idx) for f in frames]
    labels.save(str(path))


def project_skeleton(project: Optional[Path]) -> sio.Skeleton:
    """The labelling project's skeleton, so the frames merge into it; a head-to-tail chain otherwise."""
    if project is None:
        axis = [KEYPOINTS[i] for i in AXIS]
        return sio.Skeleton(nodes=KEYPOINTS, edges=list(zip(axis, axis[1:])) + [(KEYPOINTS[f], KEYPOINTS[b]) for f, b in EYES])
    skeleton = sio.load_file(str(project)).skeletons[0]
    missing = set(KEYPOINTS) - set(skeleton.node_names)
    if missing:
        raise ValueError(f"{project} skeleton lacks {sorted(missing)}")
    return skeleton


def main(args: argparse.Namespace) -> None:
    labeled = set()
    if args.labeled:
        labeled = set(pd.read_csv(args.labeled, header=[0, 1, 2], index_col=0).index.str.split("/").str[1])
    roots = experiments(args.data, args.results, args.lightning_pose)
    fish = recordings(roots, args.results, args.lightning_pose, labeled)
    if fish.empty:
        raise FileNotFoundError(f"no predictions with a video and an unlabelled fish under {args.data}")
    print(f"{len(roots)} experiment folders under {args.data}", flush=True)
    n_groups = fish.groupby(["line", "day"]).ngroups
    fish_per_group = args.fish_per_group or math.ceil(2 * args.frames / n_groups)
    sampled = fish.sample(frac=1, random_state=args.seed).groupby(["line", "day"]).head(fish_per_group)
    print(f"{len(fish)} fish in {n_groups} line x day groups; scoring {len(sampled)} (up to {fish_per_group} per group)", flush=True)
    jobs = [(row.csv, row.video, args.min_gap_s, args.per_fish) for row in sampled.itertuples()]
    scored = []
    with ProcessPoolExecutor(args.jobs) as pool:
        for i, fish_frames in enumerate(pool.map(fish_candidates, jobs), 1):
            print(f"[{i}/{len(jobs)}] scored {fish_frames['fish'].iloc[0]}", flush=True)
            scored.append(fish_frames)
    candidates = pd.concat(scored, ignore_index=True)
    candidates = candidates.merge(sampled[["csv", "line", "day"]], on="csv")
    selection = select(candidates, args.frames, args.per_fish, args.bout_fraction)
    columns = ["fish", "line", "day", "frame", "time_s", "in_bout", "score", "reason", "likelihood", "segments", "eyes", "eye side", "spike", "video"]
    selection[columns].to_csv(args.output.with_suffix(".csv"), index=False)
    write_slp(selection, project_skeleton(args.skeleton_from), args.output.with_suffix(".slp"))
    print(selection.groupby(["line", "day"]).size().to_string(), flush=True)
    print(f"{len(selection)} frames ({selection['in_bout'].mean():.0%} in bouts) from {selection['fish'].nunique()} fish; "
          f"reasons: {selection['reason'].value_counts().to_dict()}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pick the frames to label next for the full-fish pose model.")
    parser.add_argument("data", type=Path, help="Folder searched for experiments (folders with results and lightning_pose)")
    parser.add_argument("--output", type=Path, required=True, help="Output path without extension (.csv and .slp are written)")
    parser.add_argument("--skeleton-from", type=Path, help="SLEAP labelling project whose skeleton to use")
    parser.add_argument("--labeled", type=Path, help="CollectedData.csv: its fish are skipped")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--per-fish", type=int, default=3)
    parser.add_argument("--fish-per-group", type=int, help="Fish scored per line x day (default: twice the frames a group gets; reading is the slow part)")
    parser.add_argument("--bout-fraction", type=float, default=0.5)
    parser.add_argument("--min-gap-s", type=float, default=1.0)
    parser.add_argument("--results", default="results")
    parser.add_argument("--lightning-pose", default="lightning_pose")
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
