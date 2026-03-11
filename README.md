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
python -m BehaviorScreen.download_model
```

Modify data_dir in config.yaml to point to the right location

## Usage

```
python -m BehaviorScreen /path/to/data_folder /path/to/model_folder
```

## TODO

### plots

- plot trajectories of bouts (loomings, phototaxis)
- plot habituation (O-bends, loomings, prey capture, ...)
    % of fish vs trial
    % of trial vs time
- try to export average video in egocentric coords (might be a mess, but who knows)
- MAE
- freezing after loomings

### pre-processing

- stats : permutation tests, internal comparison (dark) vs external (vehicle, WT)
- estimate variability

### QC

- handle fish missing
- filter fish that do not move at the beginning?

### refactoring

- clean dead code in process.py
- maybe decouple circle detection from megabouts / run circle detection as a separate step
- "foreground_color": [0.10000000000000002, 0.10000000000000002, 0.0, 1.0] in json. Make sure to normalize columns
- multiprocessing: send a single folder to multiple Slurm nodes?
