# Ideas

- make a GUI to preprocess the data so that someone else than me can do it? Be able to visualize the results from each step in the GUI?

- check stimulus.get_shader_trial_time() rollover calculation

- Should I use unwrap_angle in extract_raw_eye_angles()?

- plot distribution of fish excluded by QC (health status, etc.)

- "Sided models classify every non-ipsilateral event as contralateral" in SidedPoissonProcess

- generate overlay videos at different scales (whole field for stimulus, zoomed on fish for tail and eyes pose) that show everything the pre-processing extracted:
    - stimulus reconstruction (RN in regular python, can it be accelerated?)
    - online tracking
    - post-hoc pose estimation
    - bout segmentation and classification
    - sacaddes segmentation and classification
    - quality check 
    - well outline and center
    - tail timeseries
    - eye timeseries
    - accumulating (x,y) trajectories

- add 1s before the start of the stimulus. Might be nice both for plotting and for the modelling
of the response onset?

- try to transform (or rather add on top) bout and saccade frequency heamtaps into one global series of PSTHs per bout/saccade category x stimulus x time 
showing trials on top of each other (stacked? maybe color coded? TBD). Maybe also add stimulus
related variables when they make sense (e.g. phase of the stimulus for prey capture). 
Would be nice to be able to include 1s pre-stimulus onset 

- look at correlation between tail bouts and saccades

- once happy with the pipeline try renewal point process

- can the whole point process pipeline be implemented with a GLM

- make figures for bouts and saccades classification (bouts and saccades examples, average of some sort?)

- Can pooling across vehicles be justified? Record more WT, check if vehicle pool has same variability as WT across clutches to justify pooling? 

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
