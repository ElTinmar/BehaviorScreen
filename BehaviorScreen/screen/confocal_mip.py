"""
Maximum-intensity projections of confocal stacks (.czi) along Z, as 8-bit PNGs and a montage per folder.
    python -m BehaviorScreen confocal-mip FOLDER [--output DIR] [--percentile 99.9]
Each subfolder of FOLDER holding .czi files (one line's stacks) gets, in OUTPUT/<subfolder>/ (default: next to its stacks):
<stack>_mip.png per stack and <subfolder>_mip_montage.png, a row of controls over a row of ablated fish (by file name:
cont, cntr or control; abla), other stacks in a last row. All the projections of a subfolder share one intensity scale
(0 to their PERCENTILE), so an ablation shows as a loss of signal; stacks with several channels give one image per
channel (_c0, _c1, ...). Copy the montage into the screen's <line>/confocal/ for the slideshow.
"""
import argparse
import re
import warnings
from pathlib import Path
from typing import Dict, List

import czifile
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROWS = {"control": re.compile(r"cont|cntr", re.I), "ablated": re.compile(r"abla", re.I)}


def projection(path: Path) -> np.ndarray:
    """Maximum along Z of the stack's first scene: (channel, y, x)."""
    with czifile.CziFile(path) as czi:
        scenes = list(czi.scenes.values())
        if len(scenes) > 1:
            warnings.warn(f"{path.name}: {len(scenes)} scenes, only the first is projected")
        data, dims = scenes[0].asarray(), list(scenes[0].dims)
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
    """Controls, then ablated fish, then the others, a row each, every image labelled with its stack's name."""
    groups = [[name for name in images if pattern.search(name)] for pattern in ROWS.values()]
    others = [name for name in images if not any(name in group for group in groups)]
    rows = [group for group in groups + [others] if group]
    height, width = (max(image.shape[i] for image in images.values()) for i in (0, 1))
    sheet = Image.new("L", (width * max(map(len, rows)), height * len(rows)))
    draw, font = ImageDraw.Draw(sheet), ImageFont.load_default(size=max(12, height // 25))
    for r, row in enumerate(rows):
        for c, name in enumerate(row):
            sheet.paste(Image.fromarray(images[name]), (c * width, r * height))
            draw.text((c * width + 8, r * height + 8), name, fill=255, font=font, stroke_width=2, stroke_fill=0)
    return sheet


def project_folder(folder: Path, output: Path, percentile: float) -> None:
    stacks = sorted(folder.glob("*.czi"), key=lambda p: p.name.lower())
    projections = {stack.stem: projection(stack) for stack in stacks}
    output.mkdir(parents=True, exist_ok=True)
    channels = max(p.shape[0] for p in projections.values())
    for channel in range(channels):
        suffix = f"_c{channel}" if channels > 1 else ""
        planes = {name: p[channel] for name, p in projections.items() if channel < p.shape[0]}
        top = max(np.percentile(np.concatenate([plane.ravel() for plane in planes.values()]), percentile), 1)
        images = {name: to_8bit(plane, top) for name, plane in planes.items()}
        for name, image in images.items():
            Image.fromarray(image).save(output / f"{name}_mip{suffix}.png")
        montage(images).save(output / f"{folder.name}_mip_montage{suffix}.png")
    print(f"{folder.name}: {len(stacks)} stacks -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="Folder of one subfolder of .czi stacks per line.")
    parser.add_argument("--output", type=Path, help="Where the subfolders' images go (default: next to their stacks).")
    parser.add_argument("--percentile", type=float, default=99.9, help="Intensity shown white, as a percentile of a subfolder's projections.")
    args = parser.parse_args()
    folders: List[Path] = sorted({stack.parent for stack in args.folder.rglob("*.czi")})
    if not folders:
        parser.error(f"no .czi under {args.folder}")
    for folder in folders:
        project_folder(folder, (args.output or args.folder) / folder.relative_to(args.folder), args.percentile)


if __name__ == "__main__":
    main()
