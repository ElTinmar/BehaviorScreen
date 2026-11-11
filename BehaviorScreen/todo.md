# Ideas

- plot distribution of fish excluded by QC (health status, etc.)

- "Sided models classify every non-ipsilateral event as contralateral" in SidedPoissonProcess

- add 1s before the start of the stimulus. Might be nice both for plotting and for the modelling
of the response onset?

- try to transform (or rather add on top) bout and saccade frequency heamtaps into one global series of PSTHs per bout/saccade category x stimulus x time 
showing trials on top of each other (stacked? maybe color coded? TBD). Maybe also add stimulus
related variables when they make sense (e.g. phase of the stimulus for prey capture). 
Would be nice to be able to include 1s pre-stimulus onset 

- look at correlation between tail bouts and saccades

- once happy with the pipeline try renewal point process

- Make use of the marks to predict next bouts (bout chain structure per behavior)

- can the whole point process pipeline be implemented with a GLM

- make figures for bouts and saccades classification (bouts and saccades examples, average of some sort?)

- Can pooling across vehicles be justified? Record more WT, check if vehicle pool has same variability as WT across clutches to justify pooling? 


# Plots

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

# Statistics

data challenges:
- pseudoreplication / correlation structure
- non stationarity (time, trials)
- count (rate) distributions / zero inflation

Distribution shape / Zero inflation:
- permutation test stat and effect size: choose better metrics than t-stat / cohen-d (variance estimates might be meaningless with that many zeros). Maybe log2 fold change 
- check the proportion of fish that engaged in the behavior (non-zero)
- check the mean/median after excluding zero counts

- stats : permutation tests, internal comparison (dark) vs external (vehicle, WT)
- estimate variability

# QC

- handle fish missing
- filter fish that do not move (at the beginning)?
- filter fish with erratic tracking

# Refactoring

- clean dead code in process.py
- maybe decouple circle detection from megabouts / run circle detection as a separate step
- "foreground_color": [0.10000000000000002, 0.10000000000000002, 0.0, 1.0] in json. Make sure to normalize columns
- multiprocessing: send a single folder to multiple Slurm nodes?


# Figures

## Main figures

### Figure 1. Setup, protocol and pre-processing
A. Setup (reuse render on Github?)
B. Protocol
C. Pose estimation
D. Bout detection and classification
E. Saccade detection and classification

### Figure 2. Heatmaps/PSTHs for bout/saccade categories x stimulus presented on the control
pool

Figure 2b (decide to include or not). Maybe extra descriptive statistics
on the control set?
    - bout categories x stimulius
    - interbouts distribution x stimulius
    - speed distribution x stimulius
    - markov chain bout sequence x stimulus


### Figure 3. Screen heatmap

### Figure 4. Parameters with bootstrap

### Figure 5/6/7 example of hits? PSTH for the affected behavior for hit vs control?


## Fig supp. 

### Figure S1. Confocal stacks to confirm ablation (for the hits only?)
