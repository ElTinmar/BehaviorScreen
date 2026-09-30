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


def main() -> None:
    """Run detection, classification, augmentation, and plotting."""
    parser = argparse.ArgumentParser(description="Run the complete saccade analysis.")
    parser.add_argument("root", type=Path)
    parser.add_argument("yaml", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("freeswim", "tethered"),
        default="freeswim",
    )
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()

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
        mode=args.mode,
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

    classify_saccades(
        model_path=args.model.resolve(),
        events_path=detected_csv,
        output_path=classified_csv,
    )

    augment_saccades(
        input_csv=classified_csv,
        output_csv=augmented_csv,
        directories=directories,
        rollover_time_s=3600,
    )

    if not args.skip_plots:
        plot_cluster_traces(
            csv_path=classified_csv,
            npz_path=detected_npz,
            trace_type="smooth",
            baseline_correct=True,
            output_path=root / "saccade_clusters.png",
        )

        make_saccade_heatmaps(
            input_csv=augmented_csv,
            valid_trials_csv=root / "valid_trials.csv",
            quality_control=root / "qc.csv",
            config_yaml=args.yaml.resolve(),
            output_png=root / "saccades.png",
            exclude_unusable=True,
            include_unassigned=False,
            maximum_frequency=0.2,
            interactive=False,
        )

    print("Saccade analysis complete.")


if __name__ == "__main__":
    main()
