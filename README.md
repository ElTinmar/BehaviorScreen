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