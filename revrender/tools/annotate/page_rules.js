// The rules of the revision annotation page that do not touch the screen: the strokes of a question, how they are joined and grouped
// into revisions, and the region of writing each revision voids.  page_editor.html inlines this file; tests/test_page_editor.py runs it
// with node.
//
// A question's state is {strokes, groups, next}:
//   stroke  {id, kind: "line" | "blot", path: [[x, y], ...]}    a struck line is a polyline drawn along the stroke (it stops where the
//           stroke crosses writing and starts again after it); a blackout is a polygon round the blackened ink
//   group   {id, strokes: [stroke ids], region: [[x, y], ...] | null, region_auto: true | false}
//           one revision: the strokes that together void some writing, and the region of that writing (computed from the ink the
//           strokes touch until the annotator moves it, then kept)
// Every stroke is in exactly one group.
"use strict";

const LINE_REACH = 2.5;     // ink this close to a struck line's path is where the stroke is
const WINDOW_X = 14, WINDOW_Y = 28;   // the ink a revision voids is looked for this far round its strokes (pixels)
const MARGIN = 3;           // a computed region is this much larger than the ink it holds

function distanceToSegment(p, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1], l = dx * dx + dy * dy;
  const t = l ? Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l)) : 0;
  return Math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy);
}

function distanceToPath(p, path, closed) {
  let best = Infinity;
  const last = closed ? path.length : path.length - 1;
  for (let i = 0; i < last; i++) best = Math.min(best, distanceToSegment(p, path[i], path[(i + 1) % path.length]));
  if (path.length === 1) best = Math.hypot(p[0] - path[0][0], p[1] - path[0][1]);
  return best;
}

function pointInPolygon(point, polygon) {
  const [x, y] = point;
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i], [xj, yj] = polygon[j];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function bounds(points) {
  const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

function joinPaths(a, b) {
  // One polyline from two: the pair of ends that lie closest is joined, so the line reads in order (paper 1's editor, key C).
  const d = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1]);
  const ra = a.slice().reverse(), rb = b.slice().reverse();
  const options = [[a.concat(b), d(a[a.length - 1], b[0])], [a.concat(rb), d(a[a.length - 1], b[b.length - 1])],
                   [ra.concat(b), d(a[0], b[0])], [b.concat(a), d(b[b.length - 1], a[0])]];
  options.sort((x, y) => x[1] - y[1]);
  return options[0][0].map(p => [p[0], p[1]]);
}

function autoRegion(strokes, ink, width, height) {
  // The writing a revision voids: the ink its strokes lie on, grown through touching ink inside a window round the strokes, as a box.
  // ink is a Uint8Array of width * height (1 = ink).  Null when the strokes touch no ink.
  const all = strokes.flatMap(s => s.path);
  if (!all.length) return null;
  const [l, t, r, b] = bounds(all);
  const x0 = Math.max(0, Math.floor(l - WINDOW_X)), x1 = Math.min(width - 1, Math.ceil(r + WINDOW_X));
  const y0 = Math.max(0, Math.floor(t - WINDOW_Y)), y1 = Math.min(height - 1, Math.ceil(b + WINDOW_Y));
  const seen = new Uint8Array((x1 - x0 + 1) * (y1 - y0 + 1)), queue = [];
  const at = (x, y) => (y - y0) * (x1 - x0 + 1) + (x - x0);
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
    if (!ink[y * width + x]) continue;
    const p = [x + .5, y + .5];
    const on = strokes.some(s => s.kind === "blot" ? pointInPolygon(p, s.path) : distanceToPath(p, s.path, false) <= LINE_REACH);
    if (on) { seen[at(x, y)] = 1; queue.push(x, y); }
  }
  if (!queue.length) return null;
  let bl = Infinity, bt = Infinity, br = -Infinity, bb = -Infinity;
  for (let k = 0; k < queue.length; k += 2) {
    const x = queue[k], y = queue[k + 1];
    bl = Math.min(bl, x); bt = Math.min(bt, y); br = Math.max(br, x); bb = Math.max(bb, y);
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const nx = x + dx, ny = y + dy;
      if (nx < x0 || nx > x1 || ny < y0 || ny > y1 || !ink[ny * width + nx] || seen[at(nx, ny)]) continue;
      seen[at(nx, ny)] = 1; queue.push(nx, ny);
    }
  }
  const L = Math.max(0, Math.min(bl, l) - MARGIN), T = Math.max(0, Math.min(bt, t) - MARGIN);
  const R = Math.min(width, Math.max(br + 1, r) + MARGIN), B = Math.min(height, Math.max(bb + 1, b) + MARGIN);
  return [[L, T], [R, T], [R, B], [L, B]].map(p => [Math.round(p[0]), Math.round(p[1])]);
}

// ---------- state changes (each returns nothing; it changes the state it is given) ----------
function groupOf(state, strokeId) { return state.groups.find(g => g.strokes.includes(strokeId)); }
function strokesOf(state, group) { return group.strokes.map(id => state.strokes.find(s => s.id === id)).filter(Boolean); }

function addStroke(state, kind, path) {
  // A new stroke in a new group of its own; returns the stroke.
  state.next = (state.next || 0) + 1;
  const stroke = { id: "s" + state.next, kind, path: path.map(p => [Math.round(p[0]), Math.round(p[1])]) };
  state.strokes.push(stroke);
  state.next += 1;
  state.groups.push({ id: "g" + state.next, strokes: [stroke.id], region: null, region_auto: true });
  return stroke;
}

function dropEmpty(state) { state.groups = state.groups.filter(g => g.strokes.length); }

function join(state, a, b) {
  // Two struck lines into one polyline (in a's group, which takes b's group's other strokes too).  False when either is a blackout.
  const sa = state.strokes.find(s => s.id === a), sb = state.strokes.find(s => s.id === b);
  if (!sa || !sb || sa.kind !== "line" || sb.kind !== "line" || a === b) return false;
  sa.path = joinPaths(sa.path, sb.path);
  const ga = groupOf(state, a), gb = groupOf(state, b);
  state.strokes = state.strokes.filter(s => s.id !== b);
  gb.strokes = gb.strokes.filter(id => id !== b);
  if (gb !== ga) { ga.strokes.push(...gb.strokes); gb.strokes = []; ga.region_auto = ga.region_auto && gb.region_auto; }
  dropEmpty(state);
  return true;
}

function group(state, ids) {
  // The groups of the given strokes become one (the first stroke's), its region computed again.
  const groups = [...new Set(ids.map(id => groupOf(state, id)).filter(Boolean))];
  if (groups.length < 2) return false;
  const [keep, ...rest] = groups;
  for (const g of rest) { keep.strokes.push(...g.strokes); g.strokes = []; }
  keep.region_auto = true;
  dropEmpty(state);
  return true;
}

function ungroup(state, ids) {
  // Each given stroke leaves its group for a group of its own.
  let changed = false;
  for (const id of ids) {
    const g = groupOf(state, id);
    if (!g || g.strokes.length < 2) continue;
    g.strokes = g.strokes.filter(s => s !== id);
    state.next = (state.next || 0) + 1;
    state.groups.push({ id: "g" + state.next, strokes: [id], region: null, region_auto: true });
    changed = true;
  }
  return changed;
}

function remove(state, ids) {
  state.strokes = state.strokes.filter(s => !ids.includes(s.id));
  for (const g of state.groups) {
    const before = g.strokes.length;
    g.strokes = g.strokes.filter(id => !ids.includes(id));
    if (g.strokes.length !== before && g.region_auto) g.region = null;
  }
  dropEmpty(state);
}

function refreshRegions(state, ink, width, height) {
  // Every group whose region is the computed one gets it computed again (null regions first of all).
  for (const g of state.groups) if (g.region_auto || !g.region) { g.region = autoRegion(strokesOf(state, g), ink, width, height); if (!g.region) g.region_auto = true; }
}

if (typeof module !== "undefined") module.exports = { distanceToSegment, distanceToPath, pointInPolygon, bounds, joinPaths, autoRegion, groupOf, strokesOf,
                                                      addStroke, join, group, ungroup, remove, refreshRegions };
