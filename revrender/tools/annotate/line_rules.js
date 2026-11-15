/* The state rules of the line page (line_review.py): which line each struck region belongs to, the line polygons, the whole-line
   mark and the corrected text.  No screen here; line_editor.js draws it and tests/test_line_review.py runs this file with node.

   store {schema, pack, regions: {id: {text, readable, line, whole_line, seen}}, lines: {id: {page, polygon, adjusted, before_whole}}, next}
   A whole-line region is alone on its line and the line's polygon is the region's own; before_whole keeps the polygon to go back to. */
'use strict';
(function(root, factory) {
  if (typeof module !== 'undefined') module.exports = factory(require('./page_rules.js'));
  else root.LineRules = factory({bounds, polygonArea});
})(typeof window !== 'undefined' ? window : this, function(G) {
  const SCHEMA = 'paper2-line-review-v1';
  const clone = x => JSON.parse(JSON.stringify(x));
  function index(pack) {
    const regions = {}, lines = {};
    pack.pages.forEach((p, pi) => { p.regions.forEach(r => regions[r.id] = {...r, page: pi}); p.lines.forEach(l => lines[l.id] = {...l, page: pi}); });
    return {regions, lines};
  }
  function initial(pack) {
    const store = {schema: SCHEMA, pack: pack.id, regions: {}, lines: {}, next: {}};
    pack.pages.forEach((p, pi) => {
      p.lines.forEach(l => {
        store.lines[l.id] = {page: pi, polygon: clone(l.polygon), adjusted: false, ...(l.whole ? {before_whole: clone(l.estimate)} : {})};
        l.regions.forEach(id => store.regions[id] = {line: l.id, whole_line: !!l.whole});
      });
      p.regions.forEach(r => Object.assign(store.regions[r.id], {text: r.text, readable: r.readable, seen: false}));
    });
    return store;
  }
  function members(store, line) { return Object.keys(store.regions).filter(id => store.regions[id].line === line); }
  function validPolygon(polygon, size) {
    if (!Array.isArray(polygon) || polygon.length < 3 || polygon.length > 200 || polygon.some(p => !Array.isArray(p) || p.length !== 2 ||
        p.some((v, i) => !Number.isFinite(v) || v < 0 || v > size[i])) || G.polygonArea(polygon) < 1) throw Error('行框至少需要三个有效顶点');
    return polygon.map(p => p.map(Math.round));
  }
  function check(pack, store) {
    // Throws unless the store belongs to this pack and names every region once, each on a line of its page.
    if (store?.schema !== SCHEMA || store.pack !== pack.id) throw Error('这份进度不属于这个页面');
    const {regions} = index(pack);
    for (const id of Object.keys(regions)) {
      const s = store.regions[id];
      if (!s || typeof s.text !== 'string' || typeof s.readable !== 'boolean') throw Error('缺少涂改 ' + id);
      const line = store.lines[s.line];
      if (!line || line.page !== regions[id].page) throw Error('涂改 ' + id + ' 的行不在本题');
      validPolygon(line.polygon, pack.pages[line.page].size);
    }
    for (const id of Object.keys(store.regions)) if (!regions[id]) throw Error('多出的涂改 ' + id);
    return store;
  }
  function newLine(pack, store, page, polygon) {
    store.next[page] = (store.next[page] || 0) + 1;
    const id = `${pack.pages[page].id}#N${store.next[page]}`;
    store.lines[id] = {page, polygon: polygon.map(p => p.map(Math.round)), adjusted: true};
    return id;
  }
  function assign(store, region, line) {
    const s = store.regions[region];
    if (s.whole_line) throw Error('整行被划的涂改单独一行；要并到别的行，先取消「整行被划」');
    if (members(store, line).some(id => store.regions[id].whole_line)) throw Error('那一行是整行被划的涂改，不能再放别的涂改');
    s.line = line;
  }
  function setWhole(pack, store, region, on) {
    const s = store.regions[region], {regions} = index(pack);
    if (!!s.whole_line === on) return;
    if (on) {
      if (members(store, s.line).length > 1) {
        const old = store.lines[s.line];
        s.line = newLine(pack, store, old.page, regions[region].region);
        store.lines[s.line].before_whole = clone(old.polygon);
      } else {
        const line = store.lines[s.line];
        line.before_whole = clone(line.polygon); line.polygon = clone(regions[region].region);
      }
      s.whole_line = true;
    } else {
      const line = store.lines[s.line], [l, t, r, b] = G.bounds(regions[region].region);
      line.polygon = line.before_whole || [[l, t], [r, t], [r, b], [l, b]]; delete line.before_whole;
      s.whole_line = false;
    }
  }
  function reshape(pack, store, line, polygon) {
    if (members(store, line).some(id => store.regions[id].whole_line)) throw Error('整行被划的行框就是涂改框；要改行框，先取消「整行被划」');
    store.lines[line].polygon = validPolygon(polygon, pack.pages[store.lines[line].page].size);
    store.lines[line].adjusted = true;
  }
  function removeLine(store, line) {
    if (members(store, line).length) throw Error('这一行还有涂改，先把它们归到别的行');
    delete store.lines[line];
  }
  function exported(pack, store) {
    const {regions} = index(pack);
    const lineIds = Object.keys(store.lines).filter(id => members(store, id).length);
    return {
      schema: SCHEMA, pack: pack.id, exported_at: new Date().toISOString(), saved_at: store.savedAt || null,
      regions: Object.keys(regions).map(id => {
        const s = store.regions[id], o = regions[id];
        return {id, page: pack.pages[o.page].id, text: s.text, readable: s.readable, original_text: o.text, original_readable: o.readable,
                corrected: s.text !== o.text || s.readable !== o.readable, line: s.line, whole_line: !!s.whole_line, seen: !!s.seen};
      }),
      lines: lineIds.map(id => {
        const l = store.lines[id], inside = members(store, id);
        return {id, page: pack.pages[l.page].id, polygon: l.polygon, regions: inside, whole_line: inside.some(r => store.regions[r].whole_line),
                adjusted: !!l.adjusted};
      }),
      draft: store,
    };
  }
  function imported(pack, data) {
    if (data?.schema !== SCHEMA || !data.draft) throw Error('不是本页面导出的文件');
    return check(pack, clone(data.draft));
  }
  return {SCHEMA, index, initial, members, check, newLine, assign, setWhole, reshape, removeLine, exported, imported, validPolygon};
});
