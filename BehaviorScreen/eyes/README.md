# Saccade detection and classification

This repository detects binocular rapid eye movements and transfers
saccade-class labels from the tethered reference dataset published by
Dowell et al. (2024).

## References

Dowell et al., Current Biology (2024).
DOI: 10.1016/j.cub.2024.08.008

Reference data:
https://data.mendeley.com/datasets/vd5zdfwc37/1

## Pipeline

1. Download the required reference files:
   - `AgMetrics_05_06_2022_updated.mat`
   - `nmapIdx220606.mat`
2. Extract the published standardized features, labels, and UMAP
   coordinates.
3. Fit a Python `umap-learn` transform to the 213,462 tethered,
   non-swimming reference events.
4. Detect rapid eye movements in the new free-swimming recordings.
5. Calculate the nine published oculomotor metrics.
6. Winsorize each metric at the 0.5th and 99.5th percentiles within
   each fish and z-score within fish.
7. Transform events into the Python reference UMAP.
8. Assign the modal label among the 100 nearest reference events.
9. Reject events beyond the calibrated median-neighbor distance.
10. Optionally apply biphasic-convergent reassignment.

## Published labels

| ID | Class |
|---:|---|
| -1 | Rejected/unassigned by this pipeline |
| 0 | Unclassified |
| 1 | Conjugate left |
| 2 | Conjugate right |
| 3 | Miniature convergent |
| 4 | Convergent |
| 5 | Non-saccadic |
| 6 | Divergent |
| 7 | Biphasic convergent right |
| 8 | Biphasic convergent left |

## Features

The reference feature order is:

1. `Amp_L`
2. `Amp_R`
3. `Vergence`
4. `MaxMedAmp_L`
5. `MaxMedAmp_R`
6. `Vel_ccw_L`
7. `Vel_cw_L`
8. `Vel_ccw_R`
9. `Vel_cw_R`

Angles are in degrees and velocities are in degrees per second.

## Run analysis 

This needs to be run only once to create the reference UMAP space
using data from the Dowell paper

```
./make_ref.sh
```

To detect saccades, run

```
python -m BehaviorScreen.eyes.detect_saccades /media/martin/DATA_18TB/Screen/WT/vehicle --output saccades.csv
```

To classify the saccades, run

```
python -m BehaviorScreen.eyes.classify_saccades \
   --model BehaviorScreen/eyes/paper_data/paper_reference.joblib \
   --events /media/martin/DATA_18TB/Screen/WT/vehicle/saccades.csv \
   --output /media/martin/DATA_18TB/Screen/WT/vehicle/classified_saccades.csv
```

To plot the clusters:

```
python -m BehaviorScreen.eyes.plot_clusters \
   --npz /media/martin/DATA_18TB/Screen/WT/vehicle/saccades.npz \
   --baseline-correct \
   /media/martin/DATA_18TB/Screen/WT/vehicle/classified_saccades.csv
```