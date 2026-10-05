#DATA_ROOT=/media/martin/DATA_18TB/Screen
DATA_ROOT=/media/martin/datastore_baier_group/_Projects/Martin_Privat/DATA/Behavioral_screen/DATA/Screen

python -m BehaviorScreen.merge_csv $DATA_ROOT \
-o saccades_augmented_control.csv \
ccka/vehicle/saccades_augmented.csv \
lhx9/vehicle/saccades_augmented.csv \
y532/vehicle/saccades_augmented.csv \
pth2/vehicle/saccades_augmented.csv \
itpr1b/vehicle/saccades_augmented.csv \
chata/vehicle/saccades_augmented.csv \
1026/vehicle/saccades_augmented.csv \
cart2/vehicle/saccades_augmented.csv \
opn4b/vehicle/saccades_augmented.csv \
WT/danieau/saccades_augmented.csv \
WT/ronidazole/saccades_augmented.csv \
WT/vehicle/saccades_augmented.csv \
tbr1b_run3/vehicle/saccades_augmented.csv \
gbx/vehicle/saccades_augmented.csv \
mpn310/vehicle/saccades_augmented.csv \
tbr1b/vehicle/saccades_augmented.csv \
pmchl/vehicle/saccades_augmented.csv \
rspo1/vehicle/saccades_augmented.csv \
gfra/vehicle/saccades_augmented.csv \
drd2a/vehicle/saccades_augmented.csv \
isl1/vehicle/saccades_augmented.csv \
atf5b/vehicle/saccades_augmented.csv \
mafaa-switchNTR-Huc-Cre/vehicle/saccades_augmented.csv \
mpn206/vehicle/saccades_augmented.csv \
th/vehicle/saccades_augmented.csv \
mpn302/vehicle/saccades_augmented.csv \
mafaa/vehicle/saccades_augmented.csv \
242A/vehicle/saccades_augmented.csv \
pmch2/vehicle/saccades_augmented.csv \
1010/vehicle/saccades_augmented.csv \
y359/vehicle/saccades_augmented.csv \
mafaa-switchNTR-ath5-Cre/vehicle/saccades_augmented.csv \
insm2/vehicle/saccades_augmented.csv \
id2b/vehicle/saccades_augmented.csv \
pcbp3/vehicle/saccades_augmented.csv \
cort/vehicle/saccades_augmented.csv \
lhx2b/vehicle/saccades_augmented.csv \
mpn318/vehicle/saccades_augmented.csv --no-header-check


python -m BehaviorScreen.merge_csv $DATA_ROOT \
-o qc.csv \
ccka/vehicle/qc.csv \
lhx9/vehicle/qc.csv \
y532/vehicle/qc.csv \
pth2/vehicle/qc.csv \
itpr1b/vehicle/qc.csv \
chata/vehicle/qc.csv \
1026/vehicle/qc.csv \
cart2/vehicle/qc.csv \
opn4b/vehicle/qc.csv \
WT/danieau/qc.csv \
WT/ronidazole/qc.csv \
WT/vehicle/qc.csv \
tbr1b_run3/vehicle/qc.csv \
gbx/vehicle/qc.csv \
mpn310/vehicle/qc.csv \
tbr1b/vehicle/qc.csv \
pmchl/vehicle/qc.csv \
rspo1/vehicle/qc.csv \
gfra/vehicle/qc.csv \
drd2a/vehicle/qc.csv \
isl1/vehicle/qc.csv \
atf5b/vehicle/qc.csv \
mafaa-switchNTR-Huc-Cre/vehicle/qc.csv \
mpn206/vehicle/qc.csv \
th/vehicle/qc.csv \
mpn302/vehicle/qc.csv \
mafaa/vehicle/qc.csv \
242A/vehicle/qc.csv \
pmch2/vehicle/qc.csv \
1010/vehicle/qc.csv \
y359/vehicle/qc.csv \
mafaa-switchNTR-ath5-Cre/vehicle/qc.csv \
insm2/vehicle/qc.csv \
id2b/vehicle/qc.csv \
pcbp3/vehicle/qc.csv \
cort/vehicle/qc.csv \
lhx2b/vehicle/qc.csv \
mpn318/vehicle/qc.csv 

python -m BehaviorScreen.merge_csv $DATA_ROOT \
-o valid_trials.csv \
ccka/vehicle/valid_trials.csv \
lhx9/vehicle/valid_trials.csv \
y532/vehicle/valid_trials.csv \
pth2/vehicle/valid_trials.csv \
itpr1b/vehicle/valid_trials.csv \
chata/vehicle/valid_trials.csv \
1026/vehicle/valid_trials.csv \
cart2/vehicle/valid_trials.csv \
opn4b/vehicle/valid_trials.csv \
WT/danieau/valid_trials.csv \
WT/ronidazole/valid_trials.csv \
WT/vehicle/valid_trials.csv \
tbr1b_run3/vehicle/valid_trials.csv \
gbx/vehicle/valid_trials.csv \
mpn310/vehicle/valid_trials.csv \
tbr1b/vehicle/valid_trials.csv \
pmchl/vehicle/valid_trials.csv \
rspo1/vehicle/valid_trials.csv \
gfra/vehicle/valid_trials.csv \
drd2a/vehicle/valid_trials.csv \
isl1/vehicle/valid_trials.csv \
atf5b/vehicle/valid_trials.csv \
mafaa-switchNTR-Huc-Cre/vehicle/valid_trials.csv \
mpn206/vehicle/valid_trials.csv \
th/vehicle/valid_trials.csv \
mpn302/vehicle/valid_trials.csv \
mafaa/vehicle/valid_trials.csv \
242A/vehicle/valid_trials.csv \
pmch2/vehicle/valid_trials.csv \
1010/vehicle/valid_trials.csv \
y359/vehicle/valid_trials.csv \
mafaa-switchNTR-ath5-Cre/vehicle/valid_trials.csv \
insm2/vehicle/valid_trials.csv \
id2b/vehicle/valid_trials.csv \
pcbp3/vehicle/valid_trials.csv \
cort/vehicle/valid_trials.csv \
lhx2b/vehicle/valid_trials.csv \
mpn318/vehicle/valid_trials.csv 
