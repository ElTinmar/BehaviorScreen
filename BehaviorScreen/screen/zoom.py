"""
The zoom shared by the HTML pages (slideshow, line_tests/tier1.html): the wheel zooms every image of a page's .view
boxes together (and the clickable .cells laid over them), dragging pans, double-click or zoom.reset() restores.
"""

ZOOM_CSS = """
  .view { flex: 1; min-height: 0; overflow: hidden; touch-action: none; user-select: none; position: relative; }
  .view img { display: block; width: 100%; height: 100%; object-fit: contain; object-position: left top; transform-origin: 0 0;
              pointer-events: none; }
  .zoomed .view { cursor: grab; }
"""

ZOOM_JS = """
function Zoom(root) {
  let scale = 1, x = 0, y = 0, drag = null;
  const zoom = { moved: false };
  zoom.place = () => {
    root.classList.toggle("zoomed", scale > 1);
    for (const el of root.querySelectorAll(".view img, .view .cells")) {
      el.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
      el.style.imageRendering = scale > 1.5 ? "pixelated" : "auto";
    }
  };
  zoom.reset = () => { [scale, x, y] = [1, 0, 0]; zoom.place(); };
  root.addEventListener("wheel", e => {
    const box = e.target.closest(".view");
    if (!box) return;
    e.preventDefault();
    const r = box.getBoundingClientRect(), cx = e.clientX - r.left, cy = e.clientY - r.top;
    const next = Math.max(1, scale * Math.exp(-e.deltaY / 600)), f = next / scale;
    [scale, x, y] = next === 1 ? [1, 0, 0] : [next, cx - (cx - x) * f, cy - (cy - y) * f];
    zoom.place();
  }, { passive: false });
  root.addEventListener("pointerdown", e => {
    const box = e.target.closest(".view");
    zoom.moved = false;
    if (!box || scale === 1) return;
    drag = [e.clientX - x, e.clientY - y];
    box.setPointerCapture(e.pointerId);
  });
  root.addEventListener("pointermove", e => {
    if (drag) { [x, y, zoom.moved] = [e.clientX - drag[0], e.clientY - drag[1], true]; zoom.place(); }
  });
  for (const end of ["pointerup", "pointercancel"]) root.addEventListener(end, () => drag = null);
  root.addEventListener("dblclick", e => { if (e.target.closest(".view")) zoom.reset(); });
  return zoom;
}
"""
