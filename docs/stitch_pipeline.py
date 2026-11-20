"""
The whole-pipeline diagram of docs/pipeline.md, stitched from its stage diagrams through the nodes they share:

    python docs/stitch_pipeline.py [--check]

A stage diagram is a mermaid block starting with `%% stage: TITLE` and `%% shared: NODE ...`. Each stage becomes a
subgraph; a shared node goes, once, into the stage that writes it, and an edge between two stages outside them.
A node is declared on a line of its own, an edge is one per line. --check: exit 1 if the stitched diagram is out of date.
"""
import argparse
import re
from pathlib import Path
from typing import Dict, List

PIPELINE = Path(__file__).with_name("pipeline.md")
BLOCK = re.compile(r"```mermaid\n(.*?)```", re.S)
BEGIN, END = "<!-- stitched: begin -->", "<!-- stitched: end -->"
EDGE = re.compile(r"^(\w+) +-[-.]*-> *(?:\|[^|]*\| *)?(\w+)$")
NODE = re.compile(r"^(\w+)[\[(]")


def stitched(text: str) -> str:
    """The mermaid source of the whole pipeline."""
    stages = [lines for block in BLOCK.findall(text) if (lines := [l.strip() for l in block.splitlines() if l.strip()])[0].startswith("%% stage:")]
    classes: List[str] = []
    declared: Dict[str, str] = {}
    first: Dict[str, int] = {}  # stage naming each node first
    writer: Dict[str, int] = {}  # stage with the first edge into each shared node
    parsed = []
    for number, lines in enumerate(stages):
        shared, body = set(lines[1].split(":", 1)[1].split()), []
        for line in lines[2:]:
            if line.startswith("classDef"):
                classes += [line] if line not in classes else []
                continue
            if line.startswith("flowchart"):
                continue
            node, edge = NODE.match(line), EDGE.match(line)
            if edge and edge[2] in shared:
                writer.setdefault(edge[2], number)
            elif node and not edge:
                if declared.setdefault(node[1], line) != line:
                    raise ValueError(f"{node[1]} is declared differently in two stages (or twice): {line}")
                if node[1] in first and node[1] not in shared:
                    raise ValueError(f"{node[1]} is in two stages but not shared")
                first.setdefault(node[1], number)
                if node[1] in shared and number != first[node[1]]:
                    continue  # declared once, in the stage it belongs to
            body.append(line)
        parsed.append((lines[0].split(":", 1)[1].strip(), body))
    owner = first | writer
    outside: List[str] = []
    subgraphs: List[str] = []
    for number, (title, body) in enumerate(parsed):
        inside = []
        for line in body:
            node, edge = NODE.match(line), EDGE.match(line)
            if edge and not owner[edge[1]] == owner[edge[2]] == number:
                outside.append(line)
            elif node and not edge and owner[node[1]] != number:
                parsed[owner[node[1]]][1].insert(0, line)  # named here first, written by a later stage
            else:
                inside.append(line)
        subgraphs += [f'subgraph stage_{number + 1}["{title}"]', *("  " + line for line in inside), "end"]
    return "\n".join(["flowchart TD", *classes, *subgraphs, *outside]) + "\n"


def updated(text: str) -> str:
    """text with its stitched diagram rewritten between BEGIN and END."""
    start, stop = text.index(BEGIN) + len(BEGIN), text.index(END)
    return text[:start] + f"\n```mermaid\n{stitched(text)}```\n" + text[stop:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="Exit 1 if the stitched diagram is out of date.")
    args = parser.parse_args()
    text = PIPELINE.read_text()
    if args.check:
        raise SystemExit(0 if updated(text) == text else f"{PIPELINE} is out of date: run python docs/stitch_pipeline.py")
    PIPELINE.write_text(updated(text))


if __name__ == "__main__":
    main()
