from pathlib import Path

ROOT = Path('/media/martin/DATA_18TB/Screen')
CONFIG_YAML = Path('BehaviorScreen/screen.yaml')

def list_subdirectories_depth(root_dir, depth=2):
    root = Path(root_dir)
    subdirs = [
        p for p in root.rglob('*') 
        if p.is_dir() and len(p.relative_to(root).parts) == depth
    ]
    
    return subdirs

FOLDERS = list_subdirectories_depth(ROOT, 2)