from pathlib import Path
import shutil 

root = Path('/ptmp/mapri/Screen')

for file in root.rglob('**/*eyes.mp4'):
    dest_file = file.with_name("eyes_" + file.stem.replace("_eyes", "") + file.suffix)
    print(f"{file.relative_to(root)} -> {dest_file.relative_to(root)}")
    shutil.move(file, dest_file)

for file in root.rglob('**/*eyes.csv'):
    dest_file = file.with_name("eyes_" + file.stem.replace("_eyes", "") + file.suffix)
    print(f"{file.relative_to(root)} -> {dest_file.relative_to(root)}")
    shutil.move(file, dest_file)

for file in root.rglob('**/*eyes_temporal_norm.csv'):
    dest_file = file.with_name("eyes_" + file.stem.replace("_eyes", "") + file.suffix)
    print(f"{file.relative_to(root)} -> {dest_file.relative_to(root)}")
    shutil.move(file, dest_file)

for file in root.rglob('**/*eyes_labeled.mp4'):
    dest_file = file.with_name("eyes_" + file.stem.replace("_eyes", "") + file.suffix)
    print(f"{file.relative_to(root)} -> {dest_file.relative_to(root)}")
    shutil.move(file, dest_file)
