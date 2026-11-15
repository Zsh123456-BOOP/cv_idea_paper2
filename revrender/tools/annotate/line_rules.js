/* The state rules of the line page (line_review.py): which line each struck region belongs to, the line polygons, the whole-line
   mark, the corrected text, and the region polygons and strike pixels where the annotator changed them.  No screen here; line_editor.js
   draws it and tests/test_line_review.py runs this file with node.

   store {schema, pack, regions: {id: state}, lines: {id: {page, polygon, adjusted, before_whole}}, next, added}
   state {text, readable, line, whole_line, seen} and, once changed, region (its polygon) and mask ({box, runs}: the strike pixels on a
   box that only grows, so a polygon shrunk and grown again gets its pixels back); a region the annotator added also has added, page
   and mode, one taken out has deleted.  A whole-line region is alone on its line and the line's polygon is the region's own;
   before_whole keeps the polygon to go back to. */
'use strict';
(function(root, factory) {
  if (typeof module !== 'undefined') module.exports = factory(require('./page_rules.js'), require('./pixel_rules.js'));
  else root.LineRules = factory({bounds, polygonArea, pointInPolygon}, root.PixelReview);
})(typeof window !== 'undefined' ? window : this, function(G, P) {
  const SCHEMA = 'paper2-line-review-v1', MARGIN = 6;
  const clone = x => JSON.parse(JSON.stringify(x));
  function index(pack) {
    const regions = {}, lines = {};
    pack.pages.forEach((p, pi) => { p.regions.forEach(r => regions[r.id] = {...r, page: pi}); p.lines.forEach(l => lines[l.id] = {...l, page: pi}); });
    return {regions, lines};
  }
  const cache = new WeakMap();
  const packRows = pack => { if (!cache.has(pack)) cache.set(pack, index(pack).regions); return cache.get(pack); };
  function row(pack, store, id) {
    // What the page was given about a region (for one the annotator added: nothing to start from).
    const s = store.regions[id];
    return packRows(pack)[id] || {id, page: s.page, mode: s.mode || 'erase', region: null, box: null, runs: [], text: '', readable: true, readings: {}, added: true};
  }
  const polygonOf = (pack, store, id) => store.regions[id].region || packRows(pack)[id].region;
  const maskOf = (pack, store, id) => store.regions[id].mask || {box: packRows(pack)[id].box, runs: packRows(pack)[id].runs};
  function ids(pack, store, page = null) {
    // The regions in page order: those of the pack, then those added, by the order they were added.
    const out = [];
    pack.pages.forEach((p, pi) => {
      if (page !== null && pi !== page) return;
      out.push(...p.regions.map(r => r.id), ...Object.keys(store.regions).filter(id => store.regions[id].added && store.regions[id].page === pi));
    });
    return out;
  }
  function initial(pack) {
    const store = {schema: SCHEMA, pack: pack.id, regions: {}, lines: {}, next: {}, added: {}};
    pack.pages.forEach((p, pi) => {
      p.lines.forEach(l => {
        store.lines[l.id] = {page: pi, polygon: clone(l.polygon), adjusted: false, ...(l.whole ? {before_whole: clone(l.estimate)} : {})};
        l.regions.forEach(id => store.regions[id] = {line: l.id, whole_line: !!l.whole});
      });
      p.regions.forEach(r => Object.assign(store.regions[r.id], {text: r.text, readable: r.readable, seen: false}));
    });
    return store;
  }
  function members(store, line) { return Object.keys(store.regions).filter(id => store.regions[id].line === line && !store.regions[id].deleted); }
  function validPolygon(polygon, size) {
    if (!Array.isArray(polygon) || polygon.length < 3 || polygon.length > 200 || polygon.some(p => !Array.isArray(p) || p.length !== 2 ||
        p.some((v, i) => !Number.isFinite(v) || v < 0 || v > size[i])) || G.polygonArea(polygon) < 1) throw Error('框至少需要三个有效顶点');
    return polygon.map(p => p.map(Math.round));
  }
  function check(pack, store) {
    // Throws unless the store belongs to this pack and names every region once, each on a line of its page.
    if (store?.schema !== SCHEMA || store.pack !== pack.id) throw Error('这份进度不属于这个页面');
    store.next ||= {}; store.added ||= {};
    const rows = packRows(pack);
    for (const id of Object.keys(rows)) if (!store.regions[id]) throw Error('缺少涂改 ' + id);
    for (const [id, s] of Object.entries(store.regions)) {
      if (!rows[id] && !(s.added && Number.isInteger(s.page) && pack.pages[s.page])) throw Error('多出的涂改 ' + id);
      if (typeof s.text !== 'string' || typeof s.readable !== 'boolean') throw Error('涂改 ' + id + ' 缺少答案');
      const page = row(pack, store, id).page, size = pack.pages[page].size;
      validPolygon(polygonOf(pack, store, id), size);
      if (s.deleted) continue;
      const line = store.lines[s.line];
      if (!line || line.page !== page) throw Error('涂改 ' + id + ' 的行不在本题');
      validPolygon(line.polygon, size);
    }
    return store;
  }
  function newLine(pack, store, page, polygon) {
    store.next[page] = (store.next[page] || 0) + 1;
    const id = `${pack.pages[page].id}#N${store.next[page]}`;
    store.lines[id] = {page, polygon: polygon.map(p => p.map(Math.round)), adjusted: true};
    return id;
  }
  function around(pack, page, polygon) {
    const [l, t, r, b] = G.bounds(polygon), [w, h] = pack.pages[page].size;
    return [[Math.max(0, l - 4), Math.max(0, t - 4)], [Math.min(w, r + 4), Math.max(0, t - 4)], [Math.min(w, r + 4), Math.min(h, b + 4)], [Math.max(0, l - 4), Math.min(h, b + 4)]];
  }
  function lineFor(pack, store, page, polygon) {
    // The line a new region goes to: the smallest line of its page holding its centre that is not a whole struck line, else a new one.
    const [l, t, r, b] = G.bounds(polygon), centre = [(l + r) / 2, (t + b) / 2];
    const holding = Object.keys(store.lines).filter(id => store.lines[id].page === page && G.pointInPolygon(centre, store.lines[id].polygon) &&
                                                         !members(store, id).some(m => store.regions[m].whole_line))
      .sort((a, b) => G.polygonArea(store.lines[a].polygon) - G.polygonArea(store.lines[b].polygon));
    return holding[0] || newLine(pack, store, page, around(pack, page, polygon));
  }
  function assign(store, region, line) {
    const s = store.regions[region];
    if (s.whole_line) throw Error('整行被划的涂改单独一行；要并到别的行，先取消「整行被划」');
    if (members(store, line).some(id => store.regions[id].whole_line)) throw Error('那一行是整行被划的涂改，不能再放别的涂改');
    s.line = line;
  }
  function setWhole(pack, store, region, on) {
    const s = store.regions[region], polygon = polygonOf(pack, store, region);
    if (!!s.whole_line === on) return;
    if (on) {
      if (members(store, s.line).length > 1) {
        const old = store.lines[s.line];
        s.line = newLine(pack, store, old.page, polygon);
        store.lines[s.line].before_whole = clone(old.polygon);
      } else {
        const line = store.lines[s.line];
        line.before_whole = clone(line.polygon); line.polygon = clone(polygon);
      }
      s.whole_line = true;
    } else {
      const line = store.lines[s.line], [l, t, r, b] = G.bounds(polygon);
      line.polygon = line.before_whole || [[l, t], [r, t], [r, b], [l, b]]; delete line.before_whole;
      s.whole_line = false;
    }
  }
  function reshape(pack, store, line, polygon) {
    if (members(store, line).some(id => store.regions[id].whole_line)) throw Error('整行被划的行框就是涂改框；要改行框，先取消「整行被划」，或改涂改框');
    store.lines[line].polygon = validPolygon(polygon, pack.pages[store.lines[line].page].size);
    store.lines[line].adjusted = true;
  }
  function removeLine(store, line) {
    if (members(store, line).length) throw Error('这一行还有涂改，先把它们归到别的行');
    delete store.lines[line];
  }
  // ---------- the regions themselves ----------
  const union = (a, b) => [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[2], b[2]), Math.max(a[3], b[3])];
  function window(pack, store, id) {
    // The readers' window of a region (data.struck_pack.window): its polygon's box and six pixels, inside the page.
    const [l, t, r, b] = G.bounds(polygonOf(pack, store, id)), [w, h] = pack.pages[row(pack, store, id).page].size;
    return [Math.max(0, l - MARGIN), Math.max(0, t - MARGIN), Math.min(w, r + MARGIN), Math.min(h, b + MARGIN)];
  }
  function reshapeRegion(pack, store, id, polygon) {
    const s = store.regions[id], size = pack.pages[row(pack, store, id).page].size;
    polygon = validPolygon(polygon, size);
    const box = P.cropBox(polygon, size), mask = maskOf(pack, store, id), all = union(mask.box, box);
    s.mask = {box: all, runs: P.remap(mask.runs, mask.box, all)}; s.region = polygon;
    if (s.whole_line && !s.deleted) store.lines[s.line].polygon = clone(polygon);
  }
  function maskIn(pack, store, id, box) {
    const mask = maskOf(pack, store, id);
    return P.decode(P.remap(mask.runs, mask.box, box), (box[2] - box[0]) * (box[3] - box[1]));
  }
  function setMask(pack, store, id, box, pixels) {
    // The strike pixels of box (a window) replaced by pixels; the rest of the mask kept.
    const mask = maskOf(pack, store, id), all = union(mask.box, box), width = all[2] - all[0];
    const out = P.decode(P.remap(mask.runs, mask.box, all), width * (all[3] - all[1])), w = box[2] - box[0];
    for (let y = box[1]; y < box[3]; y++) for (let x = box[0]; x < box[2]; x++) out[(y - all[1]) * width + x - all[0]] = pixels[(y - box[1]) * w + x - box[0]];
    store.regions[id].mask = {box: all, runs: P.encode(out)}; store.regions[id].painted = true;
  }
  function addRegion(pack, store, page, polygon) {
    polygon = validPolygon(polygon, pack.pages[page].size);
    store.added[page] = (store.added[page] || 0) + 1;
    const id = `${pack.pages[page].id}#add${store.added[page]}`;
    store.regions[id] = {added: true, page, mode: 'erase', region: polygon, mask: {box: P.cropBox(polygon, pack.pages[page].size), runs: []},
                         text: '', readable: true, seen: false, whole_line: false};
    store.regions[id].line = lineFor(pack, store, page, polygon);
    return id;
  }
  function removeRegion(store, id) { store.regions[id].deleted = true; }
  function restoreRegion(pack, store, id) {
    const s = store.regions[id], page = row(pack, store, id).page; delete s.deleted;
    if (!store.lines[s.line] || (!s.whole_line && members(store, s.line).some(m => m !== id && store.regions[m].whole_line))) {
      s.line = s.whole_line ? newLine(pack, store, page, polygonOf(pack, store, id)) : lineFor(pack, store, page, polygonOf(pack, store, id));
    }
    if (s.whole_line) {
      if (members(store, s.line).length > 1) s.line = newLine(pack, store, page, polygonOf(pack, store, id));
      store.lines[s.line].polygon = clone(polygonOf(pack, store, id));
    }
  }
  function changes(pack, store, id) {
    const s = store.regions[id], o = row(pack, store, id);
    if (o.added) return {added: true, text: true, region: true, pixels: true};
    return {added: false, text: s.text !== o.text || s.readable !== o.readable, region: JSON.stringify(polygonOf(pack, store, id)) !== JSON.stringify(o.region),
            pixels: !!s.painted};
  }
  function strike(pack, store, id, box) {
    // The strike pixels on the region's box, inside its polygon, as runs.
    const pixels = maskIn(pack, store, id, box), inside = P.regionPixels(polygonOf(pack, store, id), box);
    for (let i = 0; i < pixels.length; i++) pixels[i] &= inside[i];
    return P.encode(pixels);
  }
  function exported(pack, store) {
    const regionIds = ids(pack, store).filter(id => !(store.regions[id].added && store.regions[id].deleted));
    const lineIds = Object.keys(store.lines).filter(id => members(store, id).length);
    return {
      schema: SCHEMA, pack: pack.id, exported_at: new Date().toISOString(), saved_at: store.savedAt || null,
      regions: regionIds.map(id => {
        const s = store.regions[id], o = row(pack, store, id), c = changes(pack, store, id);
        const polygon = polygonOf(pack, store, id), box = P.cropBox(polygon, pack.pages[o.page].size);
        return {id, page: pack.pages[o.page].id, text: s.text, readable: s.readable, original_text: o.added ? null : o.text,
                original_readable: o.added ? null : o.readable, corrected: c.text, line: s.deleted ? null : s.line, whole_line: !s.deleted && !!s.whole_line,
                seen: !!s.seen, deleted: !!s.deleted, added: c.added, region_changed: c.region, pixels_changed: c.pixels, mode: o.mode,
                region: polygon, original_region: o.region, box, runs: strike(pack, store, id, box)};
      }),
      lines: lineIds.map(id => {
        const l = store.lines[id], inside = members(store, id);
        return {id, page: pack.pages[l.page].id, polygon: l.polygon, regions: inside, whole_line: inside.some(r => store.regions[r].whole_line),
                adjusted: !!l.adjusted};
      }),
      draft: store,
    };
  }
  function imported(pack, data, other = false) {
    // other: a draft of an earlier build of this page (another pack id) taken over, as long as its regions fit this pack.
    if (data?.schema !== SCHEMA || !data.draft) throw Error('不是本页面导出的文件');
    const draft = clone(data.draft);
    if (draft.pack !== pack.id) { if (!other) throw Error('这份进度不属于这个页面'); draft.pack = pack.id; }
    return check(pack, draft);
  }
  return {SCHEMA, MARGIN, index, row, ids, polygonOf, maskOf, initial, members, check, newLine, lineFor, assign, setWhole, reshape, removeLine,
          window, reshapeRegion, maskIn, setMask, addRegion, removeRegion, restoreRegion, changes, strike, exported, imported, validPolygon};
});
