# BehaviorScreen

## Installation

### Create conda environments

```
conda env create -f BehaviorScreen.yml
conda env create -f LightningPose.yml
```

### Download the pose and saccade models

```
conda activate BehaviorScreen
python -m BehaviorScreen download-models
```

## Layout

```
BehaviorScreen/
    __main__.py                        the pipeline's stages: python -m BehaviorScreen STAGE
    jobs.py, machines.yaml             where the stages run (Slurm or this computer) and where the data are, per machine
    core.py, filters.py, filters.yaml  shared by every layer: stimulus and laterality codes, saccade classes, the events kept by models and plots
    preprocessing/                     recordings -> per-fish tables (bouts, saccades, valid trials, QC) and descriptive plots
        rig.yaml                       the recording setup: left and right in the image, well and eye-crop sizes
        plots/heatmaps.yaml            layout of the heatmaps and PSTHs: epochs of each row, time bins
    point_process/                     model library: datasets, Poisson / renewal / survival processes, fish gains, inference
    screen/                            the screen itself, built on the two above
        loader.py, merge_screen.py     screen-wide tables and how they are read
        line_names.py, *_lines.csv     names of the lines
        control_models/                model ladder of each behavior, fitted to the pooled controls
        line_tests/                    each line against its control: Tier 1, Tier 2, count readouts, clutch variability
        side_analysis/                 studies the pipeline does not depend on (power, sample sizes, Marques et al. data)
        kernel_demo/                   interactive page of the model ladders
    pose_estimation_model/             training-set tools for the pose model
```

`preprocessing` and `point_process` do not import each other; `screen` imports `point_process`, never `preprocessing`.

## Configuration

What a user edits lives in six files; nothing else is meant to be changed to adapt the screen.

| File | Holds | Edit it when | Then run again |
|---|---|---|---|
| `preprocessing/rig.yaml` | left and right of bouts, saccade classes and stimuli in the camera image; well and eye-crop sizes; shader clock period | the setup changes | preprocessing from the bouts and saccades steps, `merge_screen`, `check_orientation` |
| `filters.yaml` | the events kept, one list for bouts and one for saccades, applied by the models and the plots alike | an exclusion rule changes | the plots; the control models and everything after them |
| `preprocessing/plots/heatmaps.yaml` | layout of the heatmaps and PSTHs: the epochs of each row and their time bins | a figure should look different | the plots |
| `machines.yaml` | per machine (matched by the start of its host name): the folder of the merged tables, whether jobs go to Slurm (and their log folder) or run here, worker processes | the repository moves to a new computer or cluster | nothing |
| `screen/screen_lines.csv`, `screen/facility_lines.csv` | one row per data folder (plot label, parents) and one per facility line (type, IDs) | a line is added or renamed | the figures |

The rules of the two YAML filter files share one syntax, `column: {operator: value}` with `<`, `<=`, `>`, `>=`, `==`, `!=`, `in`
and `not_in`; an event or trial with a missing value fails the rule. Fish are excluded by quality control (`qc.csv`), not here.

The model ladders are code, not configuration: `screen/control_models/model_config.py`.

## Usage

Run every command from the repository root, with the BehaviorScreen environment active. The pipeline has four stages:

```
python -m BehaviorScreen                  # the stages
python -m BehaviorScreen STAGE -h         # the options of one
```

Each stage runs where `machines.yaml` says for this computer: submitted to Slurm, one job per folder or behavior with the
resources it needs, or run here one job after another (`--local` forces that on a Slurm machine, for a quick test).
A machine without an entry stops with a message saying to add one.

### 1. Preprocessing

```
python -m BehaviorScreen preprocess /path/to/screen/line/condition
```

One job per `<line>/<condition>` folder; the path can also be a line folder or the screen root, for every folder under it.
`--steps` selects among export, pose, eye_crop, eye_pose, qc, bouts, saccades and heatmaps (default: all, in that order):

```
python -m BehaviorScreen preprocess /path/to/screen --steps qc heatmaps
```

On Slurm, jobs with pose, eye_pose or bouts get a GPU. `python -m BehaviorScreen.preprocessing FOLDER [--steps ...]`
runs the pipeline on one folder directly.

### 2. Screen-wide tables

```
python -m BehaviorScreen merge [/path/to/screen]
```

writes `bouts_all.csv`, `saccades_augmented_all.csv`, `valid_trials_all.csv` and `qc_all.csv` in the screen folder
(default: this machine's `data_root`), with line and condition taken from the folder names. The merge needs every
folder of the screen, so it runs where all the data are; copy the four tables to the cluster's `data_root` before
fitting there.

### 3. Control models

```
python -m BehaviorScreen fit-controls [experiment ...] [--modes diagnostics fit]
```

one job per behavior of `experiments.txt` (or of those named) and mode (default: fit), then

```
python -m BehaviorScreen.screen.control_models.run_analysis --mode consolidate -d /path/to/screen -o ./control_fits
```

### 4. Line tests

```
python -m BehaviorScreen test-lines [behavior ...]       # Tier 1: one job per behavior
python -m BehaviorScreen test-lines --combine            # once they are done: merge the saved behaviors into tables and maps
python -m BehaviorScreen.screen.line_tests.clutch_variability ./line_tests
python -m BehaviorScreen.screen.line_tests.count_readouts
```

A resubmitted behavior skips the cells it has already saved. The screen analyses read the merged tables from this
machine's `data_root` (machines.yaml) unless given `--data-root`, and the control fits from `./control_fits` unless given
`--fits-root`; they write their results in the folder they run in (`BehaviorScreen/screen/paths.py`).

### A new week of recordings

1. `python -m BehaviorScreen preprocess /path/to/screen/line/condition` on the cluster, for each new folder.
2. The folders' tables to the computer that holds the whole screen, then `python -m BehaviorScreen merge` there.
3. The four `*_all.csv` tables to the cluster's `data_root`.
4. `fit-controls` when the controls changed, then `test-lines`.

### Browsing the plots

```
python -m BehaviorScreen slideshow [/path/to/screen]
```

writes `slideshow.html` in the screen folder: a page per line, its arms (and a mutant's reference arm) with
their number of fish per epoch, one view per plot (bouts_classic, bouts, saccades_classic, saccades), and a last view of the
line's confocal images: every `.jpg` and `.png` in `<line>/confocal/`, where `confocal-mip` copies its montages. Arrow keys turn
pages and switch views; the wheel zooms all the arms of a page together. A first page shows the vehicle arms of all
lines pooled, once their plots are made from the merged tables (`python -m BehaviorScreen.preprocessing.pooled_heatmaps` after
`merge`, into `pooled/vehicle/plots/`). `--embed` writes `slideshow_embedded.html` instead, with the images inside: one file to send.

```
python -m BehaviorScreen confocal-mip /path/to/ablation_confocal [--output DIR]
```

projects every `.czi` stack along Z: `<stack>_mip.png` and a montage per line folder (controls over ablated fish,
one intensity scale per folder), next to the stacks, and copies the montages to the screen folder of each line whose
`confocal` column in `screen_lines.csv` lists that folder (into `<line>/confocal/`; `--screen` for another screen folder). Needs `czifile` (2019.7.2, the last release for Python 3.11) and `imagecodecs`.

### Checks

```
python -m BehaviorScreen.screen.check_orientation /path/to/screen       # left and right of bouts, saccades and stimuli agree (rig.yaml)
python -m BehaviorScreen.screen.line_tests.exclusions                    # what QC and the filters remove, per arm
python -m BehaviorScreen.screen.line_tests.plate_variance                # plate effect on the count readouts
python -m BehaviorScreen.screen.kernel_demo -o ./figures                 # then open screen/kernel_demo/index.html
```

### Tests

```
pytest -m "not slow"
```
