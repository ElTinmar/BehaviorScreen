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
