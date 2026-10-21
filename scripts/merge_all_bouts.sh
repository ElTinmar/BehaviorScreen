#ROOT=/media/martin/DATA_18TB/Screen
ROOT=/media/martin/datastore_baier_group/_Projects/Martin_Privat/DATA/Behavioral_screen/DATA/Screen

find $ROOT -name bouts.csv  | xargs python -m BehaviorScreen.merge_csv $ROOT -o bouts_all.csv
find $ROOT -name saccades_augmented.csv  | xargs python -m BehaviorScreen.merge_csv $ROOT -o saccades_augmented_all.csv
find $ROOT -name qc.csv  | xargs python -m BehaviorScreen.merge_csv $ROOT -o qc_all.csv
find $ROOT -name valid_trials.csv  | xargs python -m BehaviorScreen.merge_csv $ROOT -o valid_trials_all.csv