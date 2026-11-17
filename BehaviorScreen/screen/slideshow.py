"""
Browse the screen's plots as an HTML slideshow, a page per line.
Each page holds the line's conditions (vehicle, then ronidazole, then any other), a straight mutant's reference arm
first (MUTANT_REFERENCES, as in the screen's tests), each with its number of fish per epoch the heatmaps show (with
a usable trial in valid_trials.csv, not excluded by qc.csv). A first page holds the vehicle arms of all lines pooled, once
python -m BehaviorScreen.preprocessing.pooled_heatmaps has made their plots (SCREEN_FOLDER/pooled/vehicle/plots/).
One view per plot (default: bouts_classic, bouts, saccades_classic, saccades), its arms stacked for the plots of STACKED (wide ones),
side by side for the others: ← and → turn pages, ↑ and ↓ (or 1, 2, ...) switch views,
the wheel zooms every plot of the page together (drag to pan, double-click to reset); the zoom carries over to the
next line, and a new view starts unzoomed.
    python -m BehaviorScreen slideshow [SCREEN_FOLDER] [--plots bouts_classic.png bouts.png ...] [--embed]
SCREEN_FOLDER: by default, this machine's data_root in BehaviorScreen/machines.yaml.
Written to SCREEN_FOLDER/slideshow.html, which links to the plots in each <line>/<condition>/plots/ (by paths relative
to the screen folder: it opens wherever the screen folder is copied, as long as it stays in it), or with --embed to
SCREEN_FOLDER/slideshow_embedded.html, which holds them: one file to send.
"""
import argparse
import base64
import html
import json
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import pandas as pd

from BehaviorScreen.filters import load_yaml_config
from BehaviorScreen.jobs import this_machine
from BehaviorScreen.screen.control_models.model_config import MUTANT_REFERENCES
from BehaviorScreen.screen.line_names import PARENTS, SCREEN, is_italic

CONDITION_ORDER = ["vehicle", "ronidazole"]
PLOTS = ["bouts_classic.png", "bouts.png", "saccades_classic.png", "saccades.png"]
STACKED = {"bouts_classic", "saccades_classic"}
LABELS = {"bouts_classic": "bouts (trial avg)", "saccades_classic": "saccades (trial avg)"}  # view buttons; others: the file name
HEATMAPS = Path(__file__).parents[1] / "preprocessing" / "plots" / "heatmaps.yaml"


@lru_cache
def plotted_epochs() -> tuple:
    """The epochs the heatmaps show: the epoch_name rules of heatmaps.yaml (no adaptation, ramps or breaks they skip)."""
    epochs = []
    for stimulus in load_yaml_config(HEATMAPS)["stimuli"]:
        for rules in stimulus.get("parameters", []):
            rule = rules.get("epoch_name", {})
            epochs += [rule["=="]] if "==" in rule else rule.get("in", [])
    return tuple(epochs)


def label(folder: str) -> str:
    """The line's plot label as HTML: gene names italic where its plots set them so, ^ an allele, & an intersection."""
    if folder not in SCREEN.index or not SCREEN.at[folder, "short"]:
        return html.escape(folder)
    italic = is_italic(folder)

    def piece(part: str) -> str:
        gene, colon, construct = html.escape(part.strip()).partition(":")
        gene, caret, allele = gene.partition("^")
        gene = gene + (f"<sup>{allele}</sup>" if caret else "")
        return (f"<i>{gene}</i>" if italic else gene) + colon + construct
    return " ∩ ".join(piece(part) for part in SCREEN.at[folder, "short"].split("&"))


def source(png: Path, page: Path, embed: bool) -> str:
    if embed:
        return "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode()
    return Path(os.path.relpath(png, page.parent)).as_posix()


def sample_size(folders: List[Path], epochs: tuple) -> dict:
    """
    Per epoch of `epochs`, in protocol order, the fish of `folders` with a usable trial (presented and tracked) and not
    excluded by qc.csv, and their range ('n = 12–40 fish'): the fish the plots and the models use; empty before qc has run.
    """
    folders = [f for f in folders if (f / "valid_trials.csv").exists()]
    if not folders:
        return {"text": "", "epochs": []}
    trials = pd.concat(pd.read_csv(f / "valid_trials.csv", usecols=["file", "epoch_name", "presented", "tracking_ok"]) for f in folders)
    trials = trials[trials["epoch_name"].isin(epochs)]
    excluded = set().union(*(pd.read_csv(f / "qc.csv", usecols=["file"])["file"] for f in folders))
    usable = trials[trials["presented"] & trials["tracking_ok"] & ~trials["file"].isin(excluded)]
    n = usable.groupby("epoch_name")["file"].nunique().reindex(trials["epoch_name"].unique(), fill_value=0)
    low, high = n.min(), n.max()
    excluded_here = len(excluded & set(trials["file"]))
    text = f"n = {low}{'' if low == high else f'–{high}'} fish" + (f", {excluded_here} excluded by QC" if excluded_here else "")
    return {"text": text, "epochs": [[epoch, int(count)] for epoch, count in n.items()]}


def panel(folder: Path, caption: str, plots: List[str], page: Path, embed: bool, fish: Optional[List[Path]] = None) -> dict:
    """The plots of `folder`, and the number of fish of `fish` (default: of `folder`)."""
    pngs = {Path(plot).stem: folder / "plots" / plot for plot in plots}
    return {"caption": caption, "n": sample_size(fish or [folder], plotted_epochs()), "src": {view: source(png, page, embed) if png.exists() else None for view, png in pngs.items()}}


def pages(screen: Path, plots: List[str], page: Path, embed: bool) -> List[dict]:
    result = []
    pooled = screen / "pooled" / "vehicle"
    if (pooled / "plots").is_dir():
        vehicles = sorted(d for d in screen.glob("*/vehicle") if (d / "results").is_dir())
        result.append({"title": "All vehicles", "folder": "All vehicles", "cross": f"{len(vehicles)} lines pooled",
                       "panels": [panel(pooled, "Vehicle", plots, page, embed, fish=vehicles)]})
    for line in sorted((d for d in screen.iterdir() if d.is_dir()), key=lambda d: d.name.lower()):
        conditions = sorted((c for c in line.iterdir() if (c / "results").is_dir()),
                            key=lambda c: (CONDITION_ORDER.index(c.name) if c.name in CONDITION_ORDER else len(CONDITION_ORDER), c.name))
        if not conditions:
            continue
        if line.name in MUTANT_REFERENCES:
            reference, condition = MUTANT_REFERENCES[line.name]
            panels = [panel(screen / reference / condition, f"Reference: {label(reference)} {html.escape(condition)}", plots, page, embed)]
            panels += [panel(c, f"{label(line.name)} {html.escape(c.name)}", plots, page, embed) for c in conditions]
        else:
            panels = [panel(c, html.escape(c.name.capitalize()), plots, page, embed) for c in conditions]
        cross = " × ".join(p for p in SCREEN.loc[line.name, PARENTS] if p) if line.name in SCREEN.index else ""
        result.append({"title": label(line.name), "folder": line.name, "cross": "" if cross == line.name else cross, "panels": panels})
    return result


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Screen slideshow</title>
<style>
  :root { --bg: #fafaf9; --fg: #1c1917; --muted: #78716c; --line: #e7e5e4; --panel: #ffffff; }
  @media (prefers-color-scheme: dark) { :root { --bg: #1c1917; --fg: #f5f5f4; --muted: #a8a29e; --line: #44403c; --panel: #292524; } }
  * { box-sizing: border-box; }
  body { margin: 0; height: 100vh; display: flex; flex-direction: column; background: var(--bg); color: var(--fg);
         font: 15px/1.4 system-ui, sans-serif; }
  header { display: flex; align-items: baseline; gap: 1rem; padding: .6rem 1rem; border-bottom: 1px solid var(--line); }
  h1 { margin: 0; font-size: 1.3rem; font-weight: 600; }
  .muted { color: var(--muted); }
  .spacer { flex: 1; }
  .views { display: flex; }
  .views button { border-radius: 0; margin-left: -1px; }
  .views button:first-child { border-radius: 6px 0 0 6px; }
  .views button:last-child { border-radius: 0 6px 6px 0; }
  .views button[aria-pressed="true"] { background: var(--fg); color: var(--bg); border-color: var(--fg); }
  button, select { font: inherit; color: inherit; background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: .2rem .6rem; }
  main { flex: 1; min-height: 0; display: flex; flex-direction: column; gap: .4rem 1rem; padding: .5rem 1rem 1rem; }
  main.side { flex-direction: row; }
  figure { flex: 1; min-height: 0; min-width: 0; margin: 0; display: flex; flex-direction: column; }
  figcaption { font-weight: 600; }
  .n { font-weight: 400; position: relative; }
  .n summary { cursor: pointer; display: inline; }
  .n table { position: absolute; z-index: 1; left: 0; top: 1.5em; background: var(--panel); border: 1px solid var(--line);
             border-radius: 6px; padding: .3rem .6rem; border-spacing: 1rem 0; white-space: nowrap; color: var(--fg); }
  .n td:last-child { text-align: right; font-variant-numeric: tabular-nums; }
  .n details { display: inline; }
  .view { flex: 1; min-height: 0; overflow: hidden; touch-action: none; }
  .view img { display: block; width: 100%; height: 100%; object-fit: contain; object-position: left top; transform-origin: 0 0;
              user-select: none; -webkit-user-drag: none; }
  main.zoomed .view { cursor: grab; }
  .missing { flex: 1; display: grid; place-items: center; border: 1px dashed var(--line); color: var(--muted); }
</style>
</head>
<body>
<header>
  <h1 id="title"></h1><span id="folder" class="muted"></span><span id="cross" class="muted"></span>
  <span class="spacer"></span>
  <span id="views" class="views" role="group" aria-label="View"></span>
  <button id="prev" aria-label="Previous line">←</button><select id="jump" aria-label="Line"></select><button id="next" aria-label="Next line">→</button>
  <span id="count" class="muted"></span><span class="muted">generated __GENERATED__</span>
</header>
<main id="panels"></main>
<script>
const PAGES = __PAGES__, VIEWS = __VIEWS__, STACKED = __STACKED__, LABELS = __LABELS__;
const label = v => LABELS[v] || v;
const $ = id => document.getElementById(id);
PAGES.forEach((p, i) => $("jump").add(new Option(p.folder, i)));
VIEWS.forEach((v, k) => { const b = document.createElement("button"); b.textContent = label(v); b.onclick = () => current = show(current, k); $("views").append(b); });
let view = 0, scale = 1, x = 0, y = 0, drag = null;
function show(i, k = view) {
  i = (i + PAGES.length) % PAGES.length;
  k = (k + VIEWS.length) % VIEWS.length;
  if (k !== view) [scale, x, y] = [1, 0, 0];
  view = k;
  [...$("views").children].forEach((b, j) => b.setAttribute("aria-pressed", j === view));
  const p = PAGES[i];
  $("title").innerHTML = p.title;
  $("folder").textContent = $("title").textContent === p.folder ? "" : p.folder;
  $("cross").textContent = p.cross;
  $("count").textContent = `${i + 1} / ${PAGES.length}`;
  $("jump").value = i;
  const counts = n => n.text ? ` <span class="muted n">· <details><summary>${n.text}</summary><table>` +
    n.epochs.map(([epoch, count]) => `<tr><td>${epoch}</td><td>${count}</td></tr>`).join("") + `</table></details></span>` : "";
  $("panels").classList.toggle("side", !STACKED.includes(VIEWS[view]));
  $("panels").innerHTML = p.panels.map(c => `<figure><figcaption>${c.caption}${counts(c.n)}</figcaption>` +
    (c.src[VIEWS[view]] ? `<div class="view"><img src="${c.src[VIEWS[view]]}" alt=""></div>` : `<div class="missing">no ${label(VIEWS[view])} plot</div>`) +
    `</figure>`).join("");
  place();
  history.replaceState(null, "", "#" + encodeURIComponent(p.folder) + "/" + VIEWS[view]);
  return i;
}
const [hashLine, hashView] = decodeURIComponent(location.hash.slice(1)).split("/");
let current = show(Math.max(0, PAGES.findIndex(p => p.folder === hashLine)), Math.max(0, VIEWS.indexOf(hashView)));
$("prev").onclick = () => current = show(current - 1);
$("next").onclick = () => current = show(current + 1);
$("jump").onchange = e => current = show(+e.target.value);
function place() {
  $("panels").classList.toggle("zoomed", scale > 1);
  for (const img of $("panels").querySelectorAll(".view img")) {
    img.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
    img.style.imageRendering = scale > 1.5 ? "pixelated" : "auto";
  }
}
$("panels").addEventListener("wheel", e => {
  const box = e.target.closest(".view");
  if (!box) return;
  e.preventDefault();
  const r = box.getBoundingClientRect(), cx = e.clientX - r.left, cy = e.clientY - r.top;
  const next = Math.max(1, scale * Math.exp(-e.deltaY / 600)), f = next / scale;
  [scale, x, y] = next === 1 ? [1, 0, 0] : [next, cx - (cx - x) * f, cy - (cy - y) * f];
  place();
}, { passive: false });
$("panels").addEventListener("pointerdown", e => {
  const box = e.target.closest(".view");
  if (!box || scale === 1) return;
  drag = [e.clientX - x, e.clientY - y];
  box.setPointerCapture(e.pointerId);
});
$("panels").addEventListener("pointermove", e => { if (drag) { [x, y] = [e.clientX - drag[0], e.clientY - drag[1]]; place(); } });
$("panels").addEventListener("pointerup", () => drag = null);
$("panels").addEventListener("dblclick", e => { if (e.target.closest(".view")) { [scale, x, y] = [1, 0, 0]; place(); } });
document.addEventListener("keydown", e => {
  if (e.target.tagName === "SELECT") return;
  if (e.key === "ArrowLeft") current = show(current - 1);
  if (e.key === "ArrowRight") current = show(current + 1);
  if (e.key === "ArrowUp") { e.preventDefault(); current = show(current, view - 1); }
  if (e.key === "ArrowDown") { e.preventDefault(); current = show(current, view + 1); }
  if (/^[1-9]$/.test(e.key) && +e.key <= VIEWS.length) current = show(current, +e.key - 1);
});
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("screen", type=Path, nargs="?", help="Screen folder: the one holding the <line>/<condition> folders.")
    parser.add_argument("--plots", nargs="+", default=PLOTS, help=f"File names of the plots in each condition's plots/, one view each (default: {' '.join(PLOTS)}).")
    parser.add_argument("--embed", action="store_true", help="Put the images in the page instead of linking to them.")
    args = parser.parse_args()
    screen = args.screen or this_machine().data_root
    page = screen / ("slideshow_embedded.html" if args.embed else "slideshow.html")
    content = pages(screen, args.plots, page, args.embed)
    views = [Path(plot).stem for plot in args.plots]
    page.write_text(PAGE.replace("__PAGES__", json.dumps(content)).replace("__VIEWS__", json.dumps(views)).replace("__LABELS__", json.dumps(LABELS))
                    .replace("__STACKED__", json.dumps(sorted(STACKED))).replace("__GENERATED__", f"{datetime.now():%Y-%m-%d %H:%M}"), encoding="utf-8")
    print(f"{len(content)} lines, views {', '.join(views)} -> {page}")


if __name__ == "__main__":
    main()
