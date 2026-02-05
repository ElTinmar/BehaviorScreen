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
python -m BehaviorScreen.lightning_pose
```

Modify data_dir in config.yaml to point to the right location

## Usage

```
python -m BehaviorScreen /path/to/data_folder /path/to/model_folder
```

## TODO

### plots

- plot trajectories of bouts (loomings, phototaxis)
- plot habituation (O-bends, loomings, prey capture, ...): % of fish doing it vs trial #
- try to export average video in egocentric coords (might be a mess, but who knows)
- MAE? can I add to plot?
- plot eye traces, prey capture: eye convergence, okr: saccades

### pre-processing

- eye tracking: imbalance in the training data? might need to annotate more during OKR? might need another model altogether
- stats : permutation tests, internal comparison (dark) vs external (vehicle, WT)
- estimate variability

### QC

- handle fish missing 
- filter fish that do not move at the beginning?

### refactoring 

- clean dead code in process.py
