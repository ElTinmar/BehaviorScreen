"""
The diagrams of docs/pipeline.md as standalone SVG files, one per stage and one of the whole pipeline:

    python docs/export_pipeline.py [--output docs/svg]

Needs Node.js (npx): it runs the Mermaid command line (@mermaid-js/mermaid-cli), which fetches a headless Chrome
the first time. Labels are SVG text, editable in Inkscape or Illustrator, wrapped at 300 px.
"""
import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

from stitch_pipeline import BLOCK, PIPELINE

CONFIG = {"htmlLabels": False, "flowchart": {"htmlLabels": False, "wrappingWidth": 300}}
MMDC = ["npx", "--yes", "--package", "@mermaid-js/mermaid-cli@11", "mmdc"]


def file_name(number: int, block: str) -> str:
    title = re.search(r"%% stage: (.+)", block)
    return f"{number}_{re.sub(r'[^a-z0-9]+', '_', title[1].lower()).strip('_')}" if title else "whole_pipeline"


def standalone(svg: str) -> str:
    """A fixed size (Mermaid's fills its page), and the spaces between words kept by renderers other than browsers."""
    width, height = map(float, re.search(r'viewBox="\S+ \S+ ([\d.]+) ([\d.]+)"', svg).groups())
    svg = re.sub(r'width="100%"', f'width="{width:.0f}" height="{height:.0f}"', svg, count=1)
    svg = re.sub(r"max-width: [\d.]+px;\s*", "", svg, count=1)
    return svg.replace("<svg ", '<svg xml:space="preserve" ', 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=PIPELINE.with_name("svg"), help="Folder of the SVG files (default: docs/svg).")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        config, source = Path(tmp) / "config.json", Path(tmp) / "diagram.mmd"
        config.write_text(json.dumps(CONFIG))
        for number, block in enumerate(BLOCK.findall(PIPELINE.read_text()), 1):
            svg = args.output / f"{file_name(number, block)}.svg"
            source.write_text(block)
            subprocess.run([*MMDC, "--quiet", "-i", str(source), "-o", str(svg), "-c", str(config)], check=True)
            svg.write_text(standalone(svg.read_text()))
            print(svg)


if __name__ == "__main__":
    main()
