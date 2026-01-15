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

## Usage

```
python -m BehaviorScreen /path/to/data_folder /path/to/model_folder
```

## TODO

- overlay bout classification on video
- overlay reconstructed stimulus on video

- plot trajectories of bouts (loomings, phototaxis)

- eye tracking: imbalance in the training data? might need to annotate more during OKR? might need another model altogether
- prey capture: eye convergence, okr: saccades

- clean dead code in process.py
