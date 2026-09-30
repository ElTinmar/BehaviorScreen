#!/usr/bin/env python3
"""Run the complete saccade analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from BehaviorScreen.load import Directories
from BehaviorScreen.eyes.augment_saccades import augment_saccades
from BehaviorScreen.eyes.classify_saccades import classify_saccades
from BehaviorScreen.eyes.detect_saccades import collect_events
from BehaviorScreen.eyes.plot_clusters import plot_cluster_traces
from BehaviorScreen.eyes.plot_saccade_heatmap import make_saccade_heatmaps


def run_analysis(
    root: Path,
    config_yaml: Path,
    model_path: Path,
    mode: str = "freeswim",
    make_plots: bool = True,
) -> None:
    """Run saccade detection, classification, augmentation, and plots."""
    root = Path(root).expanduser().resolve()
    config_yaml = Path(config_yaml).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()

    if not root.is_dir():
        raise NotADirectoryError(root)

    if not config_yaml.is_file():
        raise FileNotFoundError(config_yaml)

    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    if mode not in {"freeswim", "tethered"}:
        raise ValueError(
            "mode must be 'freeswim' or 'tethered'."
        )

    directories = Directories(
        root,
        metadata="results",
        stimuli="results",
        tracking="results",
        full_tracking="lightning_pose",
        eyes_tracking="lightning_pose",
        temperature="results",
        video="results",
        video_timestamp="results",
        results="results",
        plots="plots",
    )

    detected_csv = root / "saccades.csv"
    detected_npz = root / "saccades.npz"
    classified_csv = root / "saccades_classified.csv"
    augmented_csv = root / "saccades_augmented.csv"

    events, traces = collect_events(
        directories=directories,
        mode=mode,
        snippet_pre_ms=300.0,
        snippet_post_ms=600.0,
        likelihood_threshold=0.9,
        detection_max_gap_ms=40.0,
        metric_max_gap_ms=20.0,
        onset_threshold_fraction=0.5,
        lowess_delta_threshold=0.5,
        lowess_anneal_samples=50,
    )

    if events.empty or traces is None:
        raise RuntimeError(
            "No valid saccades were detected."
        )

    events.to_csv(
        detected_csv,
        index=False,
        float_format="%.10g",
    )
    np.savez_compressed(
        detected_npz,
        **traces,
    )

    classify_saccades(
        model_path=model_path,
        events_path=detected_csv,
        output_path=classified_csv,
    )

    augment_saccades(
        input_csv=classified_csv,
        output_csv=augmented_csv,
        directories=directories,
        rollover_time_s=3600,
    )

    if make_plots:
        plot_cluster_traces(
            csv_path=classified_csv,
            npz_path=detected_npz,
            trace_type="smooth",
            baseline_correct=True,
            output_path=root / "saccade_clusters.png",
            interactive=False,
        )

        make_saccade_heatmaps(
            input_csv=augmented_csv,
            valid_trials_csv=root / "valid_trials.csv",
            quality_control=root / "qc.csv",
            config_yaml=config_yaml,
            output_png=root / "saccades.png",
            exclude_unusable=True,
            include_unassigned=False,
            maximum_frequency=0.2,
            interactive=False,
        )


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Run the complete saccade analysis."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Root experiment directory.",
    )
    parser.add_argument(
        "yaml",
        type=Path,
        help="YAML analysis configuration.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Dowell reference model.",
    )
    parser.add_argument(
        "--mode",
        choices=("freeswim", "tethered"),
        default="freeswim",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
    )

    return parser


def main() -> None:
    """Run the command-line application."""
    args = build_parser().parse_args()

    run_analysis(
        root=args.root,
        config_yaml=args.yaml,
        model_path=args.model,
        mode=args.mode,
        make_plots=not args.skip_plots,
    )


if __name__ == "__main__":
    main()