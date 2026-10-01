#!/usr/bin/env python3
"""Run the complete saccade analysis from a selected pipeline stage."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from BehaviorScreen.load import Directories
from BehaviorScreen.eyes.augment_saccades import augment_saccades
from BehaviorScreen.eyes.classify_saccades import classify_saccades
from BehaviorScreen.eyes.detect_saccades import collect_events
from BehaviorScreen.eyes.plot_clusters import plot_cluster_traces
from BehaviorScreen.eyes.plot_saccade_heatmap import (
    make_saccade_heatmaps,
)

STAGE_ORDER = {
    "detect": 0,
    "classify": 1,
    "augment": 2,
    "plot": 3,
}


def require_file(
    path: Path,
    description: str,
) -> None:
    """Raise an informative error if a required file is missing."""
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")


def run_analysis(
    root: Path,
    config_yaml: Path,
    model_path: Path | None = None,
    mode: str = "freeswim",
    start_stage: str = "detect",
    make_plots: bool = True,
) -> None:
    """
    Run the saccade analysis from a selected stage.

    Stages
    ------
    detect
        Detect, classify, augment, and optionally plot.

    classify
        Reuse ``saccades.csv`` and ``saccades.npz``, then classify,
        augment, and optionally plot.

    augment
        Reuse ``saccades_classified.csv``, then augment and optionally
        plot.

    plot
        Reuse the existing classified and augmented results and recreate
        the plots.
    """
    root = Path(root).expanduser().resolve()
    config_yaml = Path(config_yaml).expanduser().resolve()

    if not root.is_dir():
        raise NotADirectoryError(f"Experiment root does not exist: {root}")

    if start_stage not in STAGE_ORDER:
        raise ValueError(
            f"Unknown start stage {start_stage!r}. "
            f"Expected one of {list(STAGE_ORDER)}."
        )

    if mode not in {
        "freeswim",
        "tethered",
    }:
        raise ValueError("mode must be 'freeswim' or 'tethered'.")

    if make_plots:
        require_file(
            config_yaml,
            "YAML analysis configuration",
        )

    if start_stage == "plot" and not make_plots:
        raise ValueError("--from-stage plot cannot be combined with " "--skip-plots.")

    start_index = STAGE_ORDER[start_stage]

    # A model is required only if classification will run.
    if start_index <= STAGE_ORDER["classify"]:
        if model_path is None:
            raise ValueError(
                "--model is required when starting from " "'detect' or 'classify'."
            )

        model_path = Path(model_path).expanduser().resolve()

        require_file(
            model_path,
            "Dowell reference model",
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

    # ==================================================================
    # Detection
    # ==================================================================

    if start_index <= STAGE_ORDER["detect"]:
        print()
        print("=== Detecting saccades ===")

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
            raise RuntimeError("No valid saccades were detected.")

        events.to_csv(
            detected_csv,
            index=False,
            float_format="%.10g",
        )

        np.savez_compressed(
            detected_npz,
            **traces,
        )

        print(f"Saved {len(events):,} detected events " f"to {detected_csv}")
        print(f"Saved row-aligned traces to " f"{detected_npz}")

    # ==================================================================
    # Classification
    # ==================================================================

    if start_index <= STAGE_ORDER["classify"]:
        print()
        print("=== Classifying saccades ===")

        require_file(
            detected_csv,
            "Detected-saccade CSV",
        )

        # model_path was validated above for every path that reaches this
        # stage.
        assert model_path is not None

        classify_saccades(
            model_path=model_path,
            events_path=detected_csv,
            output_path=classified_csv,
        )

    # ==================================================================
    # Augmentation
    # ==================================================================

    if start_index <= STAGE_ORDER["augment"]:
        print()
        print("=== Augmenting saccades ===")

        require_file(
            classified_csv,
            "Classified-saccade CSV",
        )

        augment_saccades(
            input_csv=classified_csv,
            output_csv=augmented_csv,
            directories=directories,
            rollover_time_s=3600,
        )

    # ==================================================================
    # Plotting
    # ==================================================================

    if make_plots:
        print()
        print("=== Plotting cluster traces ===")

        require_file(
            classified_csv,
            "Classified-saccade CSV",
        )
        require_file(
            detected_npz,
            "Detected-saccade trace NPZ",
        )

        plot_cluster_traces(
            csv_path=classified_csv,
            npz_path=detected_npz,
            max_per_cluster=500,
            baseline_correct=True,
            output_path=(root / "saccade_clusters.png"),
            interactive=False,
        )

        print()
        print("=== Plotting saccade heatmaps ===")

        require_file(
            augmented_csv,
            "Augmented-saccade CSV",
        )
        require_file(
            root / "valid_trials.csv",
            "Valid-trials CSV",
        )

        make_saccade_heatmaps(
            input_csv=augmented_csv,
            valid_trials_csv=(root / "valid_trials.csv"),
            quality_control=(root / "qc.csv"),
            config_yaml=config_yaml,
            output_png=(root / "saccades.png"),
            exclude_unusable=True,
            include_unassigned=False,
            maximum_frequency=0.2,
            interactive=False,
        )

    print()
    print(f"Saccade analysis completed from stage " f"{start_stage!r}.")


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the saccade analysis, optionally starting "
            "from an existing intermediate result."
        )
    )

    parser.add_argument(
        "root",
        type=Path,
        help="Root experiment directory.",
    )

    parser.add_argument(
        "yaml",
        type=Path,
        help=(
            "YAML analysis configuration. It is required " "when plots are generated."
        ),
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help=(
            "Dowell reference model. Required when starting "
            "from the detect or classify stage."
        ),
    )

    parser.add_argument(
        "--mode",
        choices=(
            "freeswim",
            "tethered",
        ),
        default="freeswim",
        help=(
            "Recording mode used during detection. "
            "Ignored when detection is skipped. "
            "Default: %(default)s"
        ),
    )

    parser.add_argument(
        "--from-stage",
        choices=tuple(STAGE_ORDER),
        default="detect",
        help=(
            "Start at this pipeline stage and reuse outputs "
            "from earlier stages. Default: %(default)s"
        ),
    )

    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help=("Do not create cluster-trace or frequency " "heatmap plots."),
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
        start_stage=args.from_stage,
        make_plots=not args.skip_plots,
    )


if __name__ == "__main__":
    main()
