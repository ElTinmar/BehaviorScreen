"""
line_tests/tier1.html: the Tier 1 maps of every model variant merged (test-lines --combine) and of the model-free
screen (--model-free), rewritten by both. Buttons (or ↑ and ↓) switch variant, others the map; clicking a cell shows its
line's arm surface grid for that behavior in that variant (← map or Escape: back); the wheel zooms, dragging pans.
It links to the maps and grids below it: the line_tests folder can be copied whole.
"""
import html
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List

from BehaviorScreen.point_process.io import slugify
from BehaviorScreen.screen.line_names import line_html
from BehaviorScreen.screen.paths import COUNT_RESULTS, LINE_TESTS, VARIANTS, line_tests_dir
from BehaviorScreen.screen.zoom import ZOOM_CSS, ZOOM_JS

from .tier1_config import MAP_COLUMNS

MAPS = {"tier1_map": "log2 ratio", "tier1_map_d": "Glass's Δ"}  # save_tier1_maps' maps, a button each


def variants(root: Path) -> List[dict]:
    """Each variant under root with a map, its maps' cells (cell_boxes) and each cell's arm surface grid (or None)."""
    folders = {v: line_tests_dir(v, root) for v in VARIANTS} | {"model-free": root / COUNT_RESULTS.relative_to(LINE_TESTS)}
    link = lambda path: Path(os.path.relpath(path, root)).as_posix()
    result = []
    for name, folder in folders.items():
        grids = {g.stem: link(g) for g in folder.glob("arm_surface_grids/*/*.png")}
        column = lambda c: html.escape(MAP_COLUMNS.get(c["behavior"], c["behavior"]))
        cell = lambda c: {**c, "caption": f"{line_html(c['line'])} · {column(c)}",
                          "tip": html.escape(f"{c['line']} · {MAP_COLUMNS.get(c['behavior'], c['behavior'])} · {c['parameter']}"),
                          "grid": grids.get(slugify(f"{c['line']}_{c['behavior']}"))}
        maps = {view: {"src": link(folder / f"{view}.png"), "cells": [cell(c) for c in json.loads((folder / f"{view}.json").read_text())["cells"]]}
                for view in MAPS if (folder / f"{view}.png").exists() and (folder / f"{view}.json").exists()}
        if maps:
            result.append({"name": name, "maps": maps})
    return result


def write_browser(root: Path) -> Path:
    page = root / "tier1.html"
    page.write_text(PAGE.replace("__VARIANTS__", json.dumps(variants(root))).replace("__MAPS__", json.dumps(MAPS))
                    .replace("__ZOOM_CSS__", ZOOM_CSS).replace("__ZOOM_JS__", ZOOM_JS)
                    .replace("__GENERATED__", f"{datetime.now():%Y-%m-%d %H:%M}"), encoding="utf-8")
    return page


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tier 1</title>
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
  .group { display: flex; }
  .group button { border-radius: 0; margin-left: -1px; }
  .group button:first-child { border-radius: 6px 0 0 6px; }
  .group button:last-child { border-radius: 0 6px 6px 0; }
  .group button[aria-pressed="true"] { background: var(--fg); color: var(--bg); border-color: var(--fg); }
  button { font: inherit; color: inherit; background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: .2rem .6rem; }
  main { flex: 1; min-height: 0; display: flex; flex-direction: column; gap: .4rem; padding: .5rem 1rem 1rem; }
  .cells { position: absolute; left: 0; top: 0; transform-origin: 0 0; }
  .cell { position: absolute; cursor: pointer; }
  .cell:hover { outline: 2px solid #808080; background: rgba(128, 128, 128, 0.35); }
  .missing { flex: 1; display: grid; place-items: center; border: 1px dashed var(--line); color: var(--muted); }
__ZOOM_CSS__
</style>
</head>
<body>
<header>
  <h1>Tier 1</h1><span id="caption"></span>
  <span class="spacer"></span>
  <span id="variants" class="group" role="group" aria-label="Model"></span><span id="maps" class="group" role="group" aria-label="Map"></span>
  <span class="muted">generated __GENERATED__</span>
</header>
<main id="panel"></main>
<script>
__ZOOM_JS__
const VARIANTS = __VARIANTS__, MAPS = __MAPS__;
const $ = id => document.getElementById(id), zoom = Zoom($("panel"));
let variant = 0, map = 0, cell = null, shown = null;
function buttons(group, names, active, pick) {
  $(group).replaceChildren(...names.map((name, j) => {
    const b = document.createElement("button");
    b.textContent = name; b.setAttribute("aria-pressed", j === active); b.onclick = () => pick(j);
    return b;
  }));
}
function show() {
  if (!VARIANTS.length) { $("panel").innerHTML = `<div class="missing">no Tier 1 map yet: test-lines --combine</div>`; return; }
  variant = (variant + VARIANTS.length) % VARIANTS.length;
  const v = VARIANTS[variant], names = Object.keys(v.maps), m = names[Math.min(map, names.length - 1)];
  buttons("variants", VARIANTS.map(w => w.name), variant, j => { [variant, cell] = [j, null]; show(); });
  buttons("maps", names.map(n => MAPS[n]), names.indexOf(m), j => { [map, cell] = [j, null]; show(); });
  shown = v.maps[m];
  zoom.reset();
  if (cell) {
    $("caption").innerHTML = `<button id="back">← map</button> ${cell.caption}`;
    $("back").onclick = () => { cell = null; show(); };
    $("panel").innerHTML = cell.grid ? `<div class="view"><img src="${cell.grid}" alt=""></div>` : `<div class="missing">no arm surface grid for this cell</div>`;
  } else {
    const pct = f => `${(100 * f).toFixed(3)}%`;
    $("caption").innerHTML = `<span class="muted">click a cell for its arm surface grid</span>`;
    $("panel").innerHTML = `<div class="view"><img src="${shown.src}" alt=""><div class="cells">` + shown.cells.map((c, j) =>
      `<a class="cell" data-j="${j}" title="${c.tip}" style="left:${pct(c.box[0])};top:${pct(c.box[1])};width:${pct(c.box[2] - c.box[0])};height:${pct(c.box[3] - c.box[1])}"></a>`).join("") +
      `</div></div>`;
    $("panel").querySelector("img").onload = fitCells;
  }
  history.replaceState(null, "", "#" + encodeURIComponent(v.name) + "/" + m);
}
function fitCells() {
  const box = $("panel").querySelector(".view"), cells = box && box.querySelector(".cells"), img = box && box.querySelector("img");
  if (!cells || !img.naturalWidth) return;
  const s = Math.min(box.clientWidth / img.naturalWidth, box.clientHeight / img.naturalHeight);
  [cells.style.width, cells.style.height] = [`${img.naturalWidth * s}px`, `${img.naturalHeight * s}px`];
}
window.addEventListener("resize", fitCells);
$("panel").addEventListener("click", e => {
  const hit = document.elementsFromPoint(e.clientX, e.clientY).find(el => el.classList.contains("cell"));
  if (!hit || zoom.moved || cell) return;
  cell = shown.cells[+hit.dataset.j];
  show();
});
document.addEventListener("keydown", e => {
  if (e.key === "Escape") { if (cell) { cell = null; show(); } else zoom.reset(); }
  if (e.key === "ArrowUp" || e.key === "ArrowDown") { e.preventDefault(); [variant, cell] = [variant + (e.key === "ArrowUp" ? -1 : 1), null]; show(); }
});
const [hashVariant, hashMap] = decodeURIComponent(location.hash.slice(1)).split("/");
variant = Math.max(0, VARIANTS.findIndex(v => v.name === hashVariant));
map = Math.max(0, Object.keys(MAPS).indexOf(hashMap));
show();
</script>
</body>
</html>
"""
