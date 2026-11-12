# BehaviorScreen

## Installation

### Create conda environments

```
conda env create -f BehaviorScreen.yml
conda env create -f LightningPose.yml
```

### Download pose estimation model

```
conda activate BehaviorScreen
python -m BehaviorScreen.preprocessing.download_model
```

## Layout

```
BehaviorScreen/
    core.py, filters.py, screen.yaml   shared by every layer: stimulus and laterality codes, saccade classes, filter rules
    preprocessing/                     recordings -> per-fish tables (bouts, saccades, valid trials, QC) and descriptive plots
        rig.yaml                       the recording setup: left and right in the image, well and eye-crop sizes
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

## Usage

Run every command from the repository root.

### 1. Preprocessing

One line and condition (`<screen>/<line>/<condition>`) at a time:

```
python -m BehaviorScreen.preprocessing /path/to/screen/line/condition
```

`--steps` selects among export, pose, eye_crop, eye_pose, qc, bouts, saccades and heatmaps (default: all, in that order).
Every folder of a screen, on the cluster (one job per folder) or locally:

```
bash BehaviorScreen/preprocessing/submit.sh /path/to/screen [--steps qc heatmaps]
ls -d /path/to/screen/*/*/ | xargs -P 4 -I{} python -m BehaviorScreen.preprocessing {} --steps qc heatmaps
```

### 2. Screen-wide tables

```
python -m BehaviorScreen.screen.merge_screen /path/to/screen
```

writes `bouts_all.csv`, `saccades_augmented_all.csv`, `valid_trials_all.csv` and `qc_all.csv` in the screen folder,
with line and condition taken from the folder names.

### 3. Control models

```
bash BehaviorScreen/screen/control_models/submit_fits.sh
```

submits one job per behavior of `experiments.txt` (`run_analysis --mode fit`), then

```
python -m BehaviorScreen.screen.control_models.run_analysis --mode consolidate -d /path/to/screen -o ./figures
```

### 4. Line tests

```
sbatch BehaviorScreen/screen/line_tests/tier1.sbatch              # one task per behavior
sbatch --array=1 BehaviorScreen/screen/line_tests/tier1.sbatch --combine
python -m BehaviorScreen.screen.line_tests.clutch_variability ./tier1_results
python -m BehaviorScreen.screen.line_tests.count_readouts
```

`count_readouts`, `plate_variance` and `arm_absolute` take no data argument: they read the data folder from
`SCREEN_DATA_ROOT` and the control fits from `SCREEN_FITS_ROOT` (default `./figures`).

### Checks

```
python -m BehaviorScreen.screen.check_orientation /path/to/screen       # left and right of bouts, saccades and stimuli agree (rig.yaml)
python -m BehaviorScreen.screen.line_tests.exclusions /path/to/screen    # what QC and the filters remove, per arm
python -m BehaviorScreen.screen.line_tests.plate_variance                # plate effect on the count readouts
python -m BehaviorScreen.screen.kernel_demo -o ./figures                 # then open screen/kernel_demo/index.html
```

### Tests

```
pytest -m "not slow"
```
