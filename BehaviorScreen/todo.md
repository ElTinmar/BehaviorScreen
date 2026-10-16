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

- try to transform bout and saccade frequency heamtaps into one global series of PSTHs per bout/saccade category x stimulus x time 
showing trials on top of each other (stacked? maybe color coded? TBD)

- look at correlation between tail bouts and saccades

- once happy with the pipeline try renewal point process

- can the whole point process pipeline be implemented with a GLM

- make figures for bouts and saccades classification (bouts and saccades examples, average of some sort?)

- Can pooling across vehicles be justified? Record more WT, check if vehicle pool has same variability as WT across clutches to justify pooling? 

Figure 1. Setup, protocol and pre-processing
A. Setup (reuse render on Github?)
B. Protocol
C. Pose estimation
D. Bout detection and classification
E. Saccade detection and classification

Figure 2. Heatmaps/PSTHs for bout/saccade categories x stimulus presented on the control
pool

Figure 3. Screen heatmap

Figure 4. Parameters with bootstrap

Figure 5/6/7 example of hits? PSTH for the affected behavior for hit vs control?