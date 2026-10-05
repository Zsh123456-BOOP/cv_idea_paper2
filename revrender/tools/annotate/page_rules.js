// The rules of the revision annotation page that do not touch the screen.  page_editor.html inlines this file; tests/test_page_editor.py
// runs it with node.
//
// A question's state is {revisions, next}:
//   revision {id, kind: "line" | "blot", region: [[x, y], ...], text, unreadable}
//            one place where writing was voided: the region of that writing (a polygon, four corners to start with), whether it was
//            struck through or blacked out, and what the voided writing says (? for a character that cannot be read; unreadable when
//            none of it can, the text then kept but not counted)
"use strict";

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

function polygonArea(points) {
  let twice = 0;
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) twice += (points[j][0] + points[i][0]) * (points[j][1] - points[i][1]);
  return Math.abs(twice) / 2;
}

// ---------- state changes (each changes the state it is given) ----------
function addRevision(state, kind, region) {
  // A new revision after the others; returns it.  A blackout starts unreadable.
  state.next = (state.next || 0) + 1;
  const revision = { id: "r" + state.next, kind, region: region.map(p => [Math.round(p[0]), Math.round(p[1])]), text: "", unreadable: kind === "blot" };
  state.revisions.push(revision);
  return revision;
}

function removeRevision(state, id) {
  const before = state.revisions.length;
  state.revisions = state.revisions.filter(r => r.id !== id);
  return state.revisions.length !== before;
}

function setKind(state, id, kind) {
  // A revision becomes struck through or blacked out; a blackout made from a struck line becomes unreadable unless it has text.
  const r = state.revisions.find(r => r.id === id);
  if (!r || r.kind === kind) return false;
  r.kind = kind;
  if (kind === "blot" && !r.text.trim()) r.unreadable = true;
  return true;
}

function revisionAt(state, p, tolerance) {
  // The revision whose border passes within tolerance of p, else the smallest whose region holds p; null when none.
  let border = null, borderD = Infinity, inside = null, insideA = Infinity;
  for (const r of state.revisions) {
    const d = distanceToPath(p, r.region, true);
    if (d <= tolerance && d < borderD) { border = r; borderD = d; }
    const a = polygonArea(r.region);
    if (pointInPolygon(p, r.region) && a < insideA) { inside = r; insideA = a; }
  }
  return border || inside;
}

// ---------- the text ----------
function mathParts(text) {
  // [{math, text}] of a text for rendering: what is between $ $ (or $$ $$) is a formula, the rest is words; a text without $ that holds
  // a LaTeX command, ^ or _ is one formula.
  const parts = [], re = /\$\$([\s\S]+?)\$\$|\$([^$]+?)\$/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) parts.push({ math: false, text: text.slice(last, m.index) });
    parts.push({ math: true, text: m[1] !== undefined ? m[1] : m[2] });
    last = re.lastIndex;
  }
  if (last < text.length) parts.push({ math: false, text: text.slice(last) });
  if (parts.length === 1 && !parts[0].math && !text.includes("$") && /\\[a-zA-Z]+|[\^_]/.test(text)) parts[0].math = true;
  return parts.filter(p => p.text.trim());
}

// ---------- time ----------
const IDLE = 60;            // seconds: a gap between two actions longer than this counts as this long (the annotator was away)
function activeSeconds(gaps) { return gaps.reduce((sum, g) => sum + Math.min(Math.max(0, g), IDLE), 0); }

if (typeof module !== "undefined") module.exports = { distanceToSegment, distanceToPath, pointInPolygon, bounds, polygonArea, addRevision, removeRevision,
                                                      setKind, revisionAt, mathParts, activeSeconds, IDLE };
