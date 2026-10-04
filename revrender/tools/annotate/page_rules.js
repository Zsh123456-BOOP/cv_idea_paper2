// The rules of the page annotation editor that do not touch the screen: how a line's text is read, which line a revision mark
// belongs to, and what the editor warns about.  page_editor.html inlines this file; tests/test_page_editor.py runs it with node.
//
// A line's text is everything the student wrote on it, with what the student struck out between [[ and ]]:
//   x=[[3]]5      the final text is x=5, the struck text is 3
//   [[2?]]        a struck stretch partly read: ? stands for a character that cannot be read
//   [[?]]         a struck stretch that cannot be read at all
//   [[旧]]新       written over: the old text is struck, the new one is final
"use strict";

const OPEN = "[[", CLOSE = "]]";

function parseLine(text) {
  // {final, struck: [{text, legible: "yes" | "partly" | "no"}], error}: error is null or a word on what is wrong with the brackets.
  const struck = [];
  let final = "", rest = String(text || ""), error = null;
  while (rest.length) {
    const open = rest.indexOf(OPEN), close = rest.indexOf(CLOSE);
    if (close !== -1 && (open === -1 || close < open)) { error = "多了 ]]"; final += rest; break; }
    if (open === -1) { final += rest; break; }
    final += rest.slice(0, open);
    const after = rest.slice(open + OPEN.length), end = after.indexOf(CLOSE), nested = after.indexOf(OPEN);
    if (end === -1) { error = "[[ 没有配对的 ]]"; final += rest.slice(open); break; }
    if (nested !== -1 && nested < end) { error = "[[ 里面又有 [["; final += rest.slice(open); break; }
    const inside = after.slice(0, end);
    const plain = inside.replace(/\s+/g, "");
    if (!plain.length) error = error || "空的 [[ ]]";
    struck.push({ text: inside, legible: /^\?+$/.test(plain) ? "no" : plain.includes("?") ? "partly" : "yes" });
    rest = after.slice(end + CLOSE.length);
  }
  return { final, struck, error };
}

function pointInPolygon(point, polygon) {
  // Even-odd rule; polygon is [[x, y], ...].
  const [x, y] = point;
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i], [xj, yj] = polygon[j];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function markPoints(mark) {
  if (mark.shape === "box") return [[mark.x, mark.y], [mark.x + mark.w, mark.y], [mark.x + mark.w, mark.y + mark.h], [mark.x, mark.y + mark.h]];
  return mark.path.map(p => [p[0], p[1]]);
}

function centre(points) {
  const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
  return [(Math.min(...xs) + Math.max(...xs)) / 2, (Math.min(...ys) + Math.max(...ys)) / 2];
}

function distanceToPolygon(point, polygon) {
  let best = Infinity;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [ax, ay] = polygon[j], [bx, by] = polygon[i];
    const dx = bx - ax, dy = by - ay, length = dx * dx + dy * dy;
    const t = length ? Math.max(0, Math.min(1, ((point[0] - ax) * dx + (point[1] - ay) * dy) / length)) : 0;
    best = Math.min(best, Math.hypot(point[0] - ax - t * dx, point[1] - ay - t * dy));
  }
  return best;
}

function assignMarks(lines, marks) {
  // {line: [mark indexes per line, left to right], outside: [mark indexes in no line]}.  A mark is in the line whose polygon holds the
  // centre of the mark (the nearest such line's centre when several do); a mark whose centre no line holds is outside.  Lines that are
  // "not writing" take no mark.
  const line = lines.map(() => []), outside = [];
  marks.forEach((mark, m) => {
    const c = centre(markPoints(mark));
    let best = -1, bestDistance = Infinity;
    lines.forEach((l, k) => {
      if (l.not_writing || !pointInPolygon(c, l.poly)) return;
      const d = Math.hypot(c[0] - centre(l.poly)[0], c[1] - centre(l.poly)[1]);
      if (d < bestDistance) { best = k; bestDistance = d; }
    });
    if (best === -1) outside.push(m); else line[best].push(m);
  });
  line.forEach(list => list.sort((a, b) => centre(markPoints(marks[a]))[0] - centre(markPoints(marks[b]))[0]));
  return { line, outside };
}

function checks(lines, marks) {
  // {line: [warning or "" per line], page: [warnings]}: brackets that do not pair, a different number of struck stretches and marks.
  const assigned = assignMarks(lines, marks);
  const line = lines.map((l, k) => {
    if (l.not_writing) return "";
    if (l.unreadable) return "";
    const parsed = parseLine(l.text);
    if (parsed.error) return parsed.error;
    const n = assigned.line[k].length, s = parsed.struck.length;
    if (n !== s) return `涂改区域 ${n} 处，[[ ]] ${s} 处`;
    return "";
  });
  const page = [];
  if (assigned.outside.length) page.push(`${assigned.outside.length} 处涂改区域不在任何行框里`);
  return { line, page, assigned };
}

if (typeof module !== "undefined") module.exports = { parseLine, pointInPolygon, markPoints, centre, distanceToPolygon, assignMarks, checks };
