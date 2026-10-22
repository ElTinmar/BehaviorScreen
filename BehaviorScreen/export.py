import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from video_tools import CPU_VideoProcessor

from BehaviorScreen.load import Directories, find_files
from BehaviorScreen.process import detect_well, get_background_image_safe
from BehaviorScreen.protocol import protocol, protocol_ptx
from BehaviorScreen.recording import Recording

ROI = tuple[int, int, int, int]


def per_fish_outputs(
    directories: Directories, rec: Recording, stem: str, suffix: str, overwrite: bool
) -> Iterator[tuple[int, ROI, Path]]:
    """(fish index, ROI, output path) for every ROI whose output must be written."""
    directories.results.mkdir(parents=True, exist_ok=True)
    for i, roi in enumerate(rec.metadata["identity"]["ROIs"]):
        out_path = directories.results / f"{stem}_fish_{i}{suffix}"
        if overwrite or not out_path.exists():
            yield i, roi, out_path


def export_metadata(
    directories: Directories, rec: Recording, overwrite: bool = False
) -> None:
    """Per-fish metadata, with the ROI background and the well detected in it."""
    image = None
    for i, (x, y, w, h), out_path in per_fish_outputs(
        directories, rec, rec.name, ".metadata", overwrite
    ):
        if image is None:
            stored = rec.metadata["background"].get("image")
            # TODO fix missing background image upstream
            image = (
                get_background_image_safe(rec.video)
                if stored is None
                else np.asarray(stored, dtype=np.uint8)
            )

        roi = image[y : y + h, x : x + w]
        well = detect_well(
            roi, rec.pix_per_mm, png=out_path.with_name(f"{out_path.stem}_WELLS.png")
        )
        if well is None:
            print(f"[warn] {out_path.stem}: well not found")

        metadata = rec.metadata | {
            "background": rec.metadata["background"] | {"image_ROI": roi.tolist()},
            "export": {"fish_ID": i},
            "well": None if well is None else well._asdict(),
        }
        with open(out_path, "w") as fp:
            json.dump(metadata, fp)


def export_tracking(
    directories: Directories, rec: Recording, overwrite: bool = False
) -> None:

    n_tail_points = rec.metadata["settings"]["tracking"]["n_tail_pts_interp"]
    coords = ["centroid", "left_eye", "right_eye"]
    coords += [f"tail_point_{n:03}" for n in range(n_tail_points)]

    for i, (offset_x, offset_y, _, _), out_path in per_fish_outputs(
        directories, rec, rec.files.tracking.stem, ".csv", overwrite
    ):
        df = rec.tracking[rec.tracking.identity == i].set_index("index").copy()
        for coord in coords:
            df[coord + "_x"] -= offset_x
            df[coord + "_y"] -= offset_y
        df.to_csv(out_path)


def export_timestamps(
    directories: Directories, rec: Recording, overwrite: bool = False
) -> None:
    for _, _, out_path in per_fish_outputs(
        directories, rec, rec.files.video_timestamps.stem, ".csv", overwrite  # type: ignore
    ):
        rec.video_timestamps.to_csv(out_path, index=False)


def patch_epoch_name(stimuli: list[dict]) -> list[dict]:
    """Use epoch name with older version of ZebVR"""

    patched_stimuli = stimuli.copy()
    num_stim = len(patched_stimuli)
    labels = protocol_ptx if num_stim == len(protocol_ptx) else protocol
    for s, epoch in zip(patched_stimuli, labels):
        s["name"] = epoch.name
    return patched_stimuli


def export_stimuli(
    directories: Directories,
    rec: Recording,
    overwrite: bool = False,
    patch: bool = True,
) -> None:
    stimuli = patch_epoch_name(rec.stimuli) if patch else rec.stimuli
    for _, _, out_path in per_fish_outputs(
        directories, rec, rec.files.stimuli.stem, ".json", overwrite
    ):
        with open(out_path, "w") as fp:
            fp.writelines(json.dumps(line) + "\n" for line in stimuli)


def export_videos(
    directories: Directories,
    rec: Recording,
    quality: int = 18,
    overwrite: bool = False,
) -> None:

    video_cropper = CPU_VideoProcessor(str(rec.files.video), quality=quality, preset="fast")
    dest_folder = str(directories.results)

    for i, (x, y, w, h) in enumerate(rec.metadata["identity"]["ROIs"]):
        suffix = f"fish_{i}"
        out_path = Path(video_cropper.make_output_path(suffix, dest_folder))
        if out_path.exists() and not overwrite:
            continue
        video_cropper.crop(x, y, w, h, suffix=suffix, dest_folder=dest_folder)


def export(
    directories: Directories,
    rec: Recording,
    *,
    tracking_flag: bool = True,
    timestamps_flag: bool = True,
    stimuli_flag: bool = True,
    metadata_flag: bool = True,
    videos_flag: bool = True,
    quality: int = 18,
    overwrite: bool = False,
) -> None:

    if tracking_flag:
        export_tracking(directories, rec, overwrite)

    if timestamps_flag:
        export_timestamps(directories, rec, overwrite)

    if stimuli_flag:
        export_stimuli(directories, rec, overwrite)

    if metadata_flag:
        export_metadata(directories, rec, overwrite)

    if videos_flag:
        export_videos(directories, rec, quality, overwrite)


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description="Export per-animal behavior data (tracking, metadata, stimuli, videos)"
    )

    parser.add_argument(
        "root",
        type=Path,
        help="Root experiment folder (e.g. WT_oct_2025)",
    )

    parser.add_argument(
        "--quality",
        type=int,
        default=18,
        help="Video encoding quality (lower = better quality, default: 18)",
    )

    # Directory layout overrides
    parser.add_argument(
        "--metadata",
        default="data",
        help="Subfolder containing metadata files (default: data)",
    )

    parser.add_argument(
        "--stimuli",
        default="data",
        help="Subfolder containing stimulus log files (default: data)",
    )

    parser.add_argument(
        "--tracking",
        default="data",
        help="Subfolder containing tracking CSV files (default: data)",
    )

    parser.add_argument(
        "--temperature",
        default="data",
        help="Subfolder containing temperature logs (default: data)",
    )

    parser.add_argument(
        "--video",
        default="video",
        help="Subfolder containing raw video files (default: video)",
    )

    parser.add_argument(
        "--video-timestamp",
        default="video",
        help="Subfolder containing video timestamp files (default: video)",
    )

    parser.add_argument(
        "--results",
        default="results",
        help="Subfolder where per-animal exports will be written (default: results)",
    )

    # Feature toggles
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-tracking", action="store_true")
    parser.add_argument("--no-timestamps", action="store_true")
    parser.add_argument("--no-stimuli", action="store_true")
    parser.add_argument("--no-metadata", action="store_true")
    parser.add_argument("--no-videos", action="store_true")

    return parser


def main(args: argparse.Namespace) -> None:
    export_single_animals(
        root=args.root,
        metadata=args.metadata,
        stimuli=args.stimuli,
        tracking=args.tracking,
        temperature=args.temperature,
        video=args.video,
        video_timestamp=args.video_timestamp,
        results=args.results,
        quality=args.quality,
        tracking_flag=not args.no_tracking,
        timestamps_flag=not args.no_timestamps,
        stimuli_flag=not args.no_stimuli,
        metadata_flag=not args.no_metadata,
        videos_flag=not args.no_videos,
        overwrite=args.overwrite,
    )


def export_single_animals(
    root: Path,
    *,
    metadata: str = "data",
    stimuli: str = "data",
    tracking: str = "data",
    temperature: str = "data",
    video: str = "video",
    video_timestamp: str = "video",
    results: str = "results",
    quality: int = 18,
    tracking_flag: bool = True,
    timestamps_flag: bool = True,
    stimuli_flag: bool = True,
    metadata_flag: bool = True,
    videos_flag: bool = True,
    overwrite: bool = False,
) -> None:

    directories = Directories(
        root=root,
        metadata=metadata,
        stimuli=stimuli,
        tracking=tracking,
        temperature=temperature,
        full_tracking="",
        eyes_tracking="",
        video=video,
        video_timestamp=video_timestamp,
        results=results,
    )

    for file in find_files(directories):
        print(f"processing {file.metadata.stem}", flush=True)
        export(
            directories,
            Recording(file),
            tracking_flag=tracking_flag,
            timestamps_flag=timestamps_flag,
            stimuli_flag=stimuli_flag,
            metadata_flag=metadata_flag,
            videos_flag=videos_flag,
            quality=quality,
            overwrite=overwrite,
        )


if __name__ == "__main__":

    main(build_parser().parse_args())
