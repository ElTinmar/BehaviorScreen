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
python -m BehaviorScreen.preprocessing /path/to/data_folder /path/to/model_folder
```

`--steps` selects among export, pose, eye_crop, eye_pose, qc, bouts, heatmaps and saccades;
`scripts/Slurm/full_pipeline.sbatch` runs it on the cluster.

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
python -m BehaviorScreen.screen.line_tests.exclusions /path/to/screen    # what QC and the filters remove, per arm
python -m BehaviorScreen.screen.line_tests.plate_variance                # plate effect on the count readouts
python -m BehaviorScreen.screen.kernel_demo -o ./figures                 # then open screen/kernel_demo/index.html
```

### Tests

```
pytest -m "not slow"
```

## TODO

### plots

- plot trajectories of bouts (loomings, phototaxis)
- plot habituation (O-bends, loomings, prey capture, ...)
    % of fish vs trial
    % of trial vs time
- try to export average video in egocentric coords (might be a mess, but who knows) for: bout category / stimulus onset
- MAE
- freezing after loomings
- look at the first ramp down to prey capture: some lines seem more agitated towards the end
- Try clustering & PCA,UMAP,Isomap,TSNE embedding of the full response vector and color scatter plot by line / condition
    (some indiv not subjected to whole protocol, maybe need to do per stim type)
- Try LDA
- plot distributions (bout interval, interbout interval, speed, distance, xy heatmap, radial distance/thigmotaxis)
- Try to correlate behavioral deficit to line expression voxels in the atlas (intersection/combination)
- Try to see if there is a correlation structure in the deficits (S2 deficit always correlate with RT deficits for instance, or PTX deficit correlate with Looming deficits)


data challenges:
- pseudoreplication / correlation structure
- non stationarity (time, trials)
- count (rate) distributions / zero inflation


Distribution shape / Zero inflation:
- permutation test stat and effect size: choose better metrics than t-stat / cohen-d (variance estimates might be meaningless with that many zeros). Maybe log2 fold change 
- check the proportion of fish that engaged in the behavior (non-zero)
- check the mean/median after excluding zero counts



### pre-processing

- stats : permutation tests, internal comparison (dark) vs external (vehicle, WT)
- estimate variability

### QC

- handle fish missing
- filter fish that do not move (at the beginning)?
- filter fish with erratic tracking

### refactoring

- clean dead code in process.py
- maybe decouple circle detection from megabouts / run circle detection as a separate step
- "foreground_color": [0.10000000000000002, 0.10000000000000002, 0.0, 1.0] in json. Make sure to normalize columns
- multiprocessing: send a single folder to multiple Slurm nodes?
