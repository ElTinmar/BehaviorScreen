"""
Maximum-intensity projections of confocal stacks (.czi) along Z, as 8-bit PNGs and a montage per folder.
    python -m BehaviorScreen confocal-mip FOLDER [--output DIR] [--percentile 99.9] [--screen SCREEN_FOLDER]
Each subfolder of FOLDER holding .czi files (one line's stacks) gets, in OUTPUT/<subfolder>/ (default: next to its stacks):
<stack>_mip.png per stack and <subfolder>_mip_montage.png, a row of controls over a row of ablated fish (by file name:
cont, cntr or control; abla). A stack named neither is projected but left out of the montage and its scale, and listed.
All the projections of a subfolder share one intensity scale (0 to the PERCENTILE of the montage's), so an ablation shows
as a loss of signal; stacks with several channels give one image per
channel (_c0, _c1, ...). The montages are copied to SCREEN_FOLDER/<line>/confocal/ for the slideshow, <line> being the
screen folder whose confocal column in screen_lines.csv lists the subfolder (default SCREEN_FOLDER: this machine's
data_root in machines.yaml).
"""
import argparse
import re
import shutil
import warnings
from pathlib import Path
from typing import Dict, List

import czifile
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from BehaviorScreen.screen.line_names import SCREEN
from BehaviorScreen.screen.paths import data_root

ROWS = {"control": re.compile(r"cont|cntr", re.I), "ablated": re.compile(r"abla", re.I)}


def projection(path: Path) -> np.ndarray:
    """Maximum along Z of the stack's first scene: (channel, y, x)."""
    with czifile.CziFile(path) as czi:
        data, dims = czi.asarray(), list(czi.axes)
    if "S" in dims and data.shape[dims.index("S")] > 1:
        warnings.warn(f"{path.name}: {data.shape[dims.index('S')]} scenes, only the first is projected")
        data = data.take([0], axis=dims.index("S"))
    if extra := {d: n for d, n in zip(dims, data.shape) if d not in "CZYX" and n > 1}:
        raise ValueError(f"{path}: no projection for the axes {extra} (dimensions {dims})")
    data = data.squeeze(tuple(i for i, d in enumerate(dims) if d not in "CZYX"))
    dims = [d for d in dims if d in "CZYX"]
    for axis in "CZ":
        if axis not in dims:
            data, dims = data[None], [axis, *dims]
    return data.transpose([dims.index(d) for d in "CZYX"]).max(axis=1)


def to_8bit(image: np.ndarray, top: float) -> np.ndarray:
    return (np.clip(image / top, 0, 1) * 255).astype(np.uint8)


def montage(images: Dict[str, np.ndarray]) -> Image.Image:
    """Controls, then ablated fish, a row each, every image labelled with its stack's name."""
    rows = [row for row in ([name for name in images if pattern.search(name)] for pattern in ROWS.values()) if row]
    height, width = (max(image.shape[i] for image in images.values()) for i in (0, 1))
    sheet = Image.new("L", (width * max(map(len, rows)), height * len(rows)))
    draw, font = ImageDraw.Draw(sheet), ImageFont.load_default(size=max(12, height // 25))
    for r, row in enumerate(rows):
        for c, name in enumerate(row):
            sheet.paste(Image.fromarray(images[name]), (c * width, r * height))
            draw.text((c * width + 8, r * height + 8), name, fill=255, font=font, stroke_width=2, stroke_fill=0)
    return sheet


def project_folder(folder: Path, output: Path, percentile: float) -> List[Path]:
    """Its projections and montages written in output; the montages."""
    stacks = sorted(folder.glob("*.czi"), key=lambda p: p.name.lower())
    projections = {stack.stem: projection(stack) for stack in stacks}
    shown = [name for name in projections if any(pattern.search(name) for pattern in ROWS.values())]
    if flagged := [name for name in projections if name not in shown]:
        print(f"{folder.name}: neither control nor ablated by name, left out of the montage: {', '.join(flagged)}")
    output.mkdir(parents=True, exist_ok=True)
    montages = []
    channels = max(p.shape[0] for p in projections.values())
    for channel in range(channels):
        suffix = f"_c{channel}" if channels > 1 else ""
        planes = {name: p[channel] for name, p in projections.items() if channel < p.shape[0]}
        scaled = [plane.ravel() for name, plane in planes.items() if name in shown] or [plane.ravel() for plane in planes.values()]
        top = max(np.percentile(np.concatenate(scaled), percentile), 1)
        images = {name: to_8bit(plane, top) for name, plane in planes.items()}
        for name, image in images.items():
            Image.fromarray(image).save(output / f"{name}_mip{suffix}.png")
        if montaged := {name: image for name, image in images.items() if name in shown}:
            montages.append(output / f"{folder.name}_mip_montage{suffix}.png")
            montage(montaged).save(montages[-1])
    print(f"{folder.name}: {len(stacks)} stacks -> {output}")
    return montages


def copy_to_screen(montages: List[Path], folder: str, screen: Path) -> None:
    """Into screen/<line>/confocal/, <line> listing `folder` in screen_lines.csv's confocal column."""
    lines = [line for line, names in SCREEN["confocal"].items() if folder in names.split(" / ")]
    if not lines:
        print(f"{folder}: in no confocal column of screen_lines.csv, not copied")
    for line in lines:
        if not (screen / line).is_dir():
            print(f"{folder}: no folder {screen / line}, not copied")
            continue
        (screen / line / "confocal").mkdir(exist_ok=True)
        for path in montages:
            shutil.copy2(path, screen / line / "confocal")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="Folder of one subfolder of .czi stacks per line.")
    parser.add_argument("--output", type=Path, help="Where the subfolders' images go (default: next to their stacks).")
    parser.add_argument("--percentile", type=float, default=99.9, help="Intensity shown white, as a percentile of a subfolder's projections.")
    parser.add_argument("--screen", type=Path, help="Screen folder the montages are copied to (default: this machine's data_root).")
    args = parser.parse_args()
    folders: List[Path] = sorted({stack.parent for stack in args.folder.rglob("*.czi")})
    if not folders:
        parser.error(f"no .czi under {args.folder}")
    screen = data_root(args.screen)
    for folder in folders:
        montages = project_folder(folder, (args.output or args.folder) / folder.relative_to(args.folder), args.percentile)
        copy_to_screen(montages, folder.name, screen)


if __name__ == "__main__":
    main()
