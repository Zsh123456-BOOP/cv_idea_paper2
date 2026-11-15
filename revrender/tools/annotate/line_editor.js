/* The line page (line_review.py): draw each struck region's whole line on the question, correct the region polygons, strike pixels and
   struck text, mark what was seen.  The state rules are line_rules.js; the polygon handles are pixel_overview.js; saving is IndexedDB
   (library_store.js) and localStorage, each read back, plus a bound backup file. */
'use strict';
const $ = id => document.getElementById(id), L = LineRules;
const pack = JSON.parse($('line-pack').textContent), KEY = 'line-review-' + pack.id;
const LETTERS = 'abcdefghijklmnopqrstuvwxyz';
const HINTS = {
  lines: '点行框选中；一整行都被划掉就按 <kbd>Del</kbd> 删掉外面的行框；拖白点改形状，拖边上小点加点，<kbd>Alt</kbd>+点白点删点；空白处拖出矩形 = 新建一行（中心落在里面的涂改归它）；<kbd>Shift</kbd>/右键拖动平移，滚轮缩放。红虚线是涂改框，点它在右边打开。',
  regions: '点涂改框选中；拖白点改形状，拖边上小点加点，<kbd>Alt</kbd>+点白点删点；空白处拖出矩形 = 补一处漏标的涂改（再在右边涂划线像素、写被划文字）；<kbd>Shift</kbd>/右键拖动平移，滚轮缩放。紫虚线是行框。',
};
let store, page = 0, current = null, mode = 'lines', redraw = false, history = [], textMark = null, saveFailed = false;
let backupHandle = null, backupPending = null, backupWriting = false, down = null, tool = 'brush', stroke = null, carried = null, choosing = null;
const images = new Map(), greys = new Map();

// ---------- saving ----------
function save() {
  store.savedAt = Date.now(); const stamp = store.savedAt;
  let local = false; try { localStorage.setItem(KEY, JSON.stringify(store)); local = true; } catch (e) {}
  const shown = db => {
    if (store.savedAt !== stamp) return;
    const where = [db && '浏览器数据库', local && '浏览器本地'].filter(Boolean); saveFailed = !where.length;
    $('saveState').textContent = where.length ? '已自动保存 · ' + new Date(stamp).toLocaleTimeString() + ' · ' + where.join(' + ') + (backupHandle ? ' · 备份文件' : '')
                                              : '保存失败！请立即导出 JSON';
    $('saveState').classList.toggle('error', saveFailed);
  };
  ReviewLibraryStore.put(KEY, {draft: store}).then(() => ReviewLibraryStore.get(KEY)).then(v => shown(v?.draft?.savedAt === stamp)).catch(() => shown(false));
  if (backupHandle) { backupPending = JSON.stringify(L.exported(pack, store)); flushBackup(); }
}
async function flushBackup() {
  if (backupWriting) return; backupWriting = true;
  try { while (backupPending) { const value = backupPending; backupPending = null; const w = await backupHandle.createWritable(); await w.write(value); await w.close(); } }
  catch (e) { backupHandle = null; saveFailed = true; $('saveState').textContent = '备份文件写入失败：' + e.message + '；请导出 JSON'; $('saveState').classList.add('error'); }
  finally { backupWriting = false; }
}
window.addEventListener('beforeunload', e => { if (saveFailed) { e.preventDefault(); e.returnValue = ''; } });
async function load() {
  const found = [];
  try { const v = await ReviewLibraryStore.get(KEY); if (v?.draft) found.push(v.draft); } catch (e) {}
  try { const v = JSON.parse(localStorage.getItem(KEY) || 'null'); if (v) found.push(v); } catch (e) {}
  found.sort((a, b) => (b.savedAt || 0) - (a.savedAt || 0));
  for (const s of found) { try { return L.check(pack, s); } catch (e) { status('浏览器里的进度读不进来：' + e.message, true); } }
  for (const old of pack.carry || []) {
    // Nothing saved by this build yet: take over what an earlier build of this page saved in this browser.
    const drafts = [];
    try { const v = await ReviewLibraryStore.get('line-review-' + old); if (v?.draft) drafts.push(v.draft); } catch (e) {}
    try { const v = JSON.parse(localStorage.getItem('line-review-' + old) || 'null'); if (v) drafts.push(v); } catch (e) {}
    drafts.sort((a, b) => (b.savedAt || 0) - (a.savedAt || 0));
    for (const d of drafts) {
      try { const s = L.imported(pack, {schema: L.SCHEMA, draft: d}, true); carried = old; return s; }
      catch (e) { status('旧版页面的进度接不过来：' + e.message + '；请在旧页面导出后导入', true); }
    }
  }
  return L.initial(pack);
}

// ---------- changes ----------
function status(text, error = false) { $('status').textContent = text; $('status').className = error ? 'error' : ''; }
function remember() { history.push(JSON.stringify(store)); if (history.length > 80) history.shift(); }
function act(change, message) {
  remember();
  try { const said = change(); save(); render(); status(said || message); return true; }
  catch (e) { store = JSON.parse(history.pop()); render(); status(e.message, true); return false; }
}
function undo() {
  if (!history.length) return status('没有可撤销的修改');
  store = JSON.parse(history.pop()); textMark = null;
  if (!store.regions[current]) current = L.ids(pack, store, page)[0] || L.ids(pack, store)[0];
  save(); render(); status('已撤销');
}

// ---------- what is shown ----------
const state = id => store.regions[id], info = id => L.row(pack, store, id), polygon = id => L.polygonOf(pack, store, id);
const live = id => !state(id).deleted;
const order = () => L.ids(pack, store);
const bare = id => !!state(id).mask && !L.strike(pack, store, id, PixelReview.cropBox(polygon(id), pack.pages[info(id).page].size)).length;
const corrected = id => L.changes(pack, store, id).text && !info(id).added;
function matches(id) {
  const f = $('filter').value, s = state(id), c = L.changes(pack, store, id);
  if (f === 'deleted') return !!s.deleted;
  if (f === 'nopixels') return !s.deleted && bare(id);
  if (s.deleted) return f === 'all';
  return f === 'all' || (f === 'unseen' && !s.seen) || (f === 'wrong' && info(id).wrong) || (f === 'whole' && s.whole_line) ||
         (f === 'changed' && (c.text || c.region || c.pixels));
}
const centre = region => { const [l, t, r, b] = bounds(region); return [(l + r) / 2, (t + b) / 2]; };
function pageLines(pi = page) {
  return Object.keys(store.lines).filter(id => store.lines[id].page === pi)
    .sort((a, b) => bounds(store.lines[a].polygon)[1] - bounds(store.lines[b].polygon)[1] || bounds(store.lines[a].polygon)[0] - bounds(store.lines[b].polygon)[0]);
}
const lineName = id => store.lines[id] ? '第 ' + (pageLines(store.lines[id].page).indexOf(id) + 1) + ' 行' : '（无）';
const regionName = id => { const i = L.ids(pack, store, info(id).page).indexOf(id); return i < LETTERS.length ? LETTERS[i] : 'z' + (i - 25); };
function progress() {
  const all = order().filter(live), seen = all.filter(id => state(id).seen).length;
  const fixed = all.filter(id => { const c = L.changes(pack, store, id); return c.text || c.region || c.pixels; }).length;
  const added = all.filter(id => info(id).added).length, gone = order().filter(id => !live(id) && !info(id).added).length;
  $('progress').textContent = `已看 ${seen} / ${all.length} 处 · 改过 ${fixed} 处` + (added ? ` · 补了 ${added} 处` : '') + (gone ? ` · 删了 ${gone} 处` : '') + ` · 共 ${pack.pages.length} 题`;
}

function renderPages() {
  const list = $('pages'); list.replaceChildren();
  pack.pages.forEach((p, pi) => {
    const ids = L.ids(pack, store, pi); if (!ids.some(matches) && pi !== page) return;
    const alive = ids.filter(live), seen = alive.filter(id => state(id).seen).length, d = document.createElement('div');
    d.className = (pi === page ? 'on ' : '') + (seen === alive.length ? 'done' : '');
    const [writer, ...rest] = p.id.split('_');
    d.innerHTML = `<span>${pi + 1}. ${writer.slice(0, 6)}…_${rest.join('_')}</span><span class="count">${seen}/${alive.length}</span>`;
    d.onclick = () => go(ids.find(matches) || ids[0]);
    list.append(d);
  });
  list.querySelector('.on')?.scrollIntoView({block: 'nearest'});
}
function renderChips() {
  const ids = L.ids(pack, store, page), alive = ids.filter(live);
  $('pageInfo').textContent = `${pack.pages[page].id} · ${alive.filter(id => state(id).seen).length}/${alive.length} 已看 · ${pageLines().filter(id => L.members(store, id).length).length} 行`;
  $('chips').replaceChildren(...ids.map(id => {
    const c = document.createElement('span');
    c.className = 'chip' + (state(id).seen ? ' seen' : '') + (info(id).wrong ? ' wrong' : '') + (id === current ? ' on' : '') + (matches(id) ? '' : ' off') + (live(id) ? '' : ' deleted');
    c.textContent = regionName(id) + (info(id).added ? '+' : ''); c.title = info(id).wrong ? '上限读错（按原答案）' : info(id).added ? '补标的' : '';
    c.onclick = () => go(id); return c;
  }));
}
function renderDetail() {
  const id = current, s = state(id), o = info(id), c = L.changes(pack, store, id);
  $('regionName').textContent = '涂改 ' + regionName(id) + (o.added ? '（补标）' : '');
  $('regionWhere').innerHTML = s.deleted ? '<span class="deletedNote">已删除（不是涂改）</span>' :
    `${lineName(s.line)} · ${o.mode === 'keep' ? '原来是模式 2（保原字）' : '模式 1（擦划线）'}` + (c.region ? ' · 框改过' : '') + (c.pixels ? ' · 像素改过' : '') + (bare(id) ? ' · <span class="deletedNote">还没涂划线像素</span>' : '');
  $('deleteRegionBtn').textContent = s.deleted ? '恢复这处' : '删除这处（不是涂改）';
  $('readings').replaceChildren(...Object.entries(o.readings).map(([label, text]) => {
    const r = document.createElement('div'); r.className = 'r';
    const bad = label === pack.check && o.wrong;
    r.innerHTML = `<small>${pack.labels[label] || label}</small><span>${SymbolPicker.rendered(text)} <small><code></code></small>${bad ? ' <span class="bad">与原答案不一致</span>' : ''}</span>`;
    r.querySelector('code').textContent = text; return r;
  }));
  if (!Object.keys(o.readings).length) $('readings').textContent = '（补标的涂改没有读数）';
  $('staleNote').textContent = c.region || c.pixels ? '；框或像素改过，读数是改之前的' : '';
  if (document.activeElement !== $('text') || textMark !== id) $('text').value = s.text;
  preview();
  $('origText').textContent = o.added ? '（补标，没有原答案）' : (o.text || '（空）'); $('origReadable').textContent = !o.added && !o.readable ? '· 原来勾了读不出' : '';
  $('changedMark').textContent = corrected(id) ? '已改' : ''; $('changedMark').className = corrected(id) ? 'changed' : ''; $('revertText').hidden = !corrected(id);
  $('unreadable').checked = !s.readable; $('wholeLine').checked = !!s.whole_line; $('seen').checked = !!s.seen;
  for (const k of ['text', 'unreadable', 'wholeLine', 'lineSelect']) $(k).disabled = !!s.deleted;
  const prefill = pack.pages[o.page].lines.find(l => l.regions.includes(id));
  $('wholeGuess').textContent = prefill?.whole ? '（程序猜是整行，请确认）' : '';
  const select = $('lineSelect'); select.replaceChildren();
  for (const line of pageLines()) select.append(new Option(`${lineName(line)}（${L.members(store, line).length} 处${line === s.line ? '，当前' : ''}）`, line));
  select.append(new Option('新建一行', 'new')); select.value = s.line; if (s.whole_line) select.disabled = true;
  renderPixels();
}
let lastGood = '';
function preview() {
  const {html, complete} = SymbolPicker.render($('text').value);
  if (complete || !lastGood) { $('preview').innerHTML = html; if (complete) lastGood = html; }
  $('previewNote').hidden = complete;
}
function renderLine() {
  const id = mode === 'lines' ? whole.selected : null, panel = $('linePanel');
  const wholeAlready = id && store.lines[id] && L.members(store, id).some(r => state(r).whole_line);
  $('redrawBtn').disabled = !id; $('deleteLineBtn').disabled = !id || !store.lines[id] || wholeAlready;
  if (choosing !== id) choosing = null;
  $('redrawBtn').textContent = redraw ? '在图上拖出新的行框…' : '重画选中行框';
  const line = mode === 'lines' ? id : state(current)?.line;
  if (!line || !store.lines[line]) { panel.textContent = mode === 'lines' ? '在图上点一个行框。' : '（这处没有行）'; return; }
  const inside = L.members(store, line), l = store.lines[line];
  panel.innerHTML = `<b>${lineName(line)}</b> · ${l.polygon.length} 个点 · ${l.adjusted ? '人工调过' : '程序预填'}${inside.some(r => state(r).whole_line) ? ' · 整行被划' : ''}<ul></ul>`;
  panel.querySelector('ul').replaceChildren(...(inside.length ? inside.map(r => {
    const li = document.createElement('li'); li.textContent = '涂改 ' + regionName(r) + (state(r).seen ? '（已看过）' : ''); li.style.cursor = 'pointer'; li.onclick = () => go(r); return li;
  }) : [Object.assign(document.createElement('li'), {textContent: '空行（导出时不算），可删除'})]));
  if (choosing === line) {
    const box = document.createElement('div'); box.className = 'choose';
    box.append('这一行有 ' + inside.length + ' 处涂改。哪一处是整行被划？它单独成行、不再有外框，其余的留在这一行：', document.createElement('br'));
    for (const r of inside) { const b = document.createElement('button'); b.textContent = '涂改 ' + regionName(r) + '：' + (state(r).text || '（空）').slice(0, 16); b.onclick = () => dropSelected(r); box.append(b); }
    const cancel = document.createElement('button'); cancel.textContent = '取消'; cancel.onclick = () => { choosing = null; renderLine(); }; box.append(cancel);
    panel.append(box);
  }
}
function dropSelected(region = null) {
  // Delete in the line mode: the selected line's outer polygon goes (line_rules dropLine); of several regions, which is asked first.
  const id = whole.selected; if (mode !== 'lines' || !id || !store.lines[id]) return status('先在图上点一个行框', true);
  const inside = L.members(store, id);
  if (inside.length > 1 && !region) { choosing = id; renderLine(); $('linePanel').scrollIntoView({block: 'nearest'}); return status('这一行有几处涂改：在右边「选中的行」里选哪一处是整行被划'); }
  choosing = null;
  act(() => {
    const did = L.dropLine(pack, store, id, region);
    if (did === 'removed') { whole.selected = null; return '已删除空行'; }
    const which = inside.length === 1 ? inside[0] : region; whole.selected = state(which).line;
    return `已删除外面的行框：涂改 ${regionName(which)} 算整行被划，行框就是它的涂改框` + (inside.length > 1 ? '；其余涂改留在原来那一行，记得把那一行的框收窄' : '');
  });
}
function render() {
  if (!state(current)) current = L.ids(pack, store, page)[0];
  progress(); renderPages(); renderChips(); renderDetail(); renderLine(); whole.draw();
  $('modeLines').classList.toggle('on', mode === 'lines'); $('modeRegions').classList.toggle('on', mode === 'regions'); $('hint').innerHTML = HINTS[mode];
}

// ---------- the pixels of a region, cut as data.struck_pack.layered cuts them ----------
async function imageOf(pi) {
  if (!images.has(pi)) { const im = new Image(); im.src = pack.pages[pi].image; await im.decode(); images.set(pi, im); }
  return images.get(pi);
}
function greyOf(pi) {
  if (!greys.has(pi)) {
    const im = images.get(pi), c = document.createElement('canvas'); c.width = im.width; c.height = im.height;
    const cx = c.getContext('2d', {willReadFrequently: true}); cx.drawImage(im, 0, 0);
    const rgba = cx.getImageData(0, 0, im.width, im.height).data, g = new Uint8Array(im.width * im.height);
    for (let i = 0; i < g.length; i++) g[i] = Math.round((rgba[4 * i] + rgba[4 * i + 1] + rgba[4 * i + 2]) / 3);
    greys.set(pi, g);
  }
  return greys.get(pi);
}
function cut(id) {
  // {box, w, h, scan, inside, strike} of a region's readers' window: the scan white outside the polygon; strike the pixels marked in it.
  const pi = info(id).page, box = L.window(pack, store, id), [l, t, r, b] = box, w = r - l, h = b - t, width = pack.pages[pi].size[0];
  const grey = greyOf(pi), inside = PixelReview.regionPixels(polygon(id), box), scan = new Uint8Array(w * h), strike = L.maskIn(pack, store, id, box);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) scan[y * w + x] = inside[y * w + x] ? grey[(y + t) * width + x + l] : 255;
  for (let i = 0; i < strike.length; i++) strike[i] &= inside[i];
  return {box, w, h, scan, inside, strike};
}
function paintCanvas(canvas, w, h, pixel, scale) {
  canvas.width = w; canvas.height = h; const cx = canvas.getContext('2d'), image = cx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) { const [r, g, b] = pixel(i); image.data.set([r, g, b, 255], 4 * i); }
  cx.putImageData(image, 0, 0); canvas.style.width = Math.round(w * scale) + 'px';
}
function renderPixels(view = null) {
  if (!images.has(info(current).page)) return;
  const v = view || cut(current), big = Math.max(1, Math.min(8, 400 / v.w, 220 / v.h)), small = Math.min(2, 200 / v.w);
  paintCanvas($('paint'), v.w, v.h, i => v.strike[i] ? [230, 40, 40] : v.inside[i] ? [v.scan[i], v.scan[i], v.scan[i]] : [228, 232, 238], big);
  paintCanvas($('cropRaw'), v.w, v.h, i => [v.scan[i], v.scan[i], v.scan[i]], small);
  paintCanvas($('cropErased'), v.w, v.h, i => v.strike[i] ? [255, 255, 255] : [v.scan[i], v.scan[i], v.scan[i]], small);
  $('brushLabel').textContent = $('brushSize').value + ' 像素';
}
function dab(event) {
  const c = $('paint'), rect = c.getBoundingClientRect(), v = stroke.view, radius = +$('brushSize').value / 2;
  const x = (event.clientX - rect.left) * v.w / rect.width, y = (event.clientY - rect.top) * v.h / rect.height;
  for (let yy = Math.floor(y - radius); yy <= Math.ceil(y + radius); yy++) for (let xx = Math.floor(x - radius); xx <= Math.ceil(x + radius); xx++) {
    if (xx < 0 || yy < 0 || xx >= v.w || yy >= v.h || Math.hypot(xx + .5 - x, yy + .5 - y) > Math.max(radius, .71)) continue;
    const i = yy * v.w + xx;
    if (tool === 'eraser') v.strike[i] = 0; else if (v.inside[i] && v.scan[i] < 245) v.strike[i] = 1;     // the brush takes ink in the polygon only
  }
  renderPixels(v);
}
$('paint').addEventListener('pointerdown', e => {
  if (state(current).deleted) return status('这处已删除；先恢复再改像素', true);
  e.preventDefault(); $('paint').setPointerCapture(e.pointerId); stroke = {view: cut(current)}; dab(e);
});
$('paint').addEventListener('pointermove', e => { if (stroke) dab(e); });
const endStroke = () => {
  if (!stroke) return; const v = stroke.view; stroke = null;
  act(() => L.setMask(pack, store, current, v.box, v.strike), (tool === 'eraser' ? '已擦掉标记' : '已涂上划线像素') + '，已自动保存');
};
$('paint').addEventListener('pointerup', endStroke); $('paint').addEventListener('pointercancel', endStroke);
function setTool(next) { tool = next; $('brushBtn').classList.toggle('on', tool === 'brush'); $('eraserBtn').classList.toggle('on', tool === 'eraser'); }

// ---------- the question image ----------
const regionRows = () => L.ids(pack, store, page).filter(live).map(id => ({id, region: polygon(id)}));
const whole = new PixelOverview($('whole'), {
  rows: () => mode === 'lines'
    ? pageLines().map(id => ({id, region: store.lines[id].polygon, adjusted: store.lines[id].adjusted,
                              empty: !L.members(store, id).length, wholeLine: L.members(store, id).some(r => state(r).whole_line)}))
    : regionRows(),
  colour: (r, selected) => mode === 'regions' ? (selected ? '#235cca' : '#c8372d') :
    selected ? '#235cca' : r.empty ? '#9aa3ad' : r.wholeLine ? '#137660' : r.adjusted ? '#7b4fc9' : '#3f8fd8',
  label: (r, i) => mode === 'regions' ? regionName(r.id) : String(i + 1),
  overlay: (cx, scale) => {
    const outline = (points, colour, width, dash) => {
      cx.beginPath(); points.forEach((p, i) => i ? cx.lineTo(...p) : cx.moveTo(...p)); cx.closePath();
      cx.setLineDash(dash.map(d => d / scale)); cx.lineWidth = width / scale; cx.strokeStyle = colour; cx.stroke();
    };
    for (const o of pack.pages[page].others) outline(o.region, '#9aa3ad', 1, [3, 3]);
    if (mode === 'regions') for (const id of pageLines()) outline(store.lines[id].polygon, '#7b4fc9', 1.2, [6, 4]);
    else for (const {id, region} of regionRows()) {
      outline(region, '#c8372d', id === current ? 2 : 1.2, [4, 3]); cx.setLineDash([]);
      const [, , right, bottom] = bounds(region); cx.font = `${12 / scale}px sans-serif`; cx.fillStyle = '#c8372d'; cx.fillText(regionName(id), right + 2 / scale, bottom);
    }
    if (current && live(current) && info(current).page === page) {
      cx.beginPath(); polygon(current).forEach((p, i) => i ? cx.lineTo(...p) : cx.moveTo(...p)); cx.closePath(); cx.fillStyle = 'rgba(255,200,0,.28)'; cx.fill();
    }
    cx.setLineDash([]);
  },
  select: r => { if (mode === 'regions' && r && r.id !== current) go(r.id); else renderLine(); },
  activity: () => {}, open: () => {},
  zoom: scale => { $('zoomLabel').textContent = Math.round(scale * 100) + '%'; },
  reshape: (id, points) => mode === 'regions'
    ? act(() => L.reshapeRegion(pack, store, id, points), `涂改 ${regionName(id)} 的框已调整` + (state(id).whole_line ? '（整行被划，行框跟着变）' : '') + '；框变大了就在右边补涂划线像素')
    : act(() => L.reshape(pack, store, id, points), lineName(id) + '的行框已调整'),
  create: points => {
    if (mode === 'regions') {
      let made = null;
      if (act(() => { made = L.addRegion(pack, store, page, points); return `补了一处涂改 ${regionName(made)}，归到${lineName(state(made).line)}；在右边涂划线像素、写被划文字`; })) go(made);
      return;
    }
    if (redraw && whole.selected) { const id = whole.selected; redraw = false; return act(() => L.reshape(pack, store, id, points), lineName(id) + '已重画'); }
    let made = null;
    act(() => {
      made = L.newLine(pack, store, page, points); const moved = [];
      for (const {id, region} of regionRows()) if (pointInPolygon(centre(region), points) && !state(id).whole_line) { state(id).line = made; moved.push(regionName(id)); }
      return `新建了${lineName(made)}` + (moved.length ? `，涂改 ${moved.join('、')} 归到这一行` : '（空行：在右边「所属行」里选它，或删掉）');
    });
    if (made && store.lines[made]) whole.select(made);
  },
});
$('whole').addEventListener('pointerdown', e => { down = [e.clientX, e.clientY]; });
$('whole').addEventListener('pointerup', e => {
  // In the line mode a click (not a drag) inside a region opens it on the right, and selects its line.
  if (mode !== 'lines' || !down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4 || e.button !== 0 || e.shiftKey) return;
  const p = whole.point(e), hit = regionRows().filter(r => pointInPolygon(p, r.region)).sort((a, b) => polygonArea(a.region) - polygonArea(b.region))[0];
  if (hit) go(hit.id);
});
function setMode(next) {
  mode = next; redraw = false;
  whole.selected = mode === 'regions' ? (live(current) ? current : null) : state(current).line;
  render(); status(mode === 'regions' ? '编辑涂改框：拖点改框，空白处拖矩形补一处' : '编辑行框');
}

// ---------- moving about ----------
async function go(id) {
  if (!id || !state(id)) return; textMark = null;
  const pi = info(id).page, turned = pi !== page || !whole.image; current = id; page = pi;
  const target = mode === 'regions' ? (live(id) ? id : null) : state(id).line;
  if (turned) { const im = await imageOf(pi); if (current !== id) return; whole.setImage(im, target, true); }
  else whole.select(target);
  render();
}
function step(direction) {
  const list = order().filter(id => matches(id) || id === current), at = list.indexOf(current);
  const next = list[at + direction];
  if (next) go(next); else status(direction > 0 ? '这已经是筛选后的最后一处' : '这已经是第一处');
}

// ---------- controls ----------
$('text').addEventListener('input', () => {
  if (textMark !== current) { remember(); textMark = current; }        // one undo step per stay in the box
  state(current).text = $('text').value; preview(); save();
  $('changedMark').textContent = corrected(current) ? '已改' : ''; $('changedMark').className = corrected(current) ? 'changed' : ''; $('revertText').hidden = !corrected(current);
  renderChips(); progress();
});
$('text').addEventListener('blur', () => { textMark = null; });
const picker = SymbolPicker.attach($('text'), $('symbolList'), $('symbolBar'), value => {
  if ($('text').value !== value) $('text').value = value;
  $('text').dispatchEvent(new Event('input'));
});
$('text').addEventListener('keydown', e => { if (picker.keys(e)) e.stopPropagation(); });
$('revertText').onclick = () => act(() => { state(current).text = info(current).text; state(current).readable = info(current).readable; }, '已恢复原来的答案');
$('unreadable').onchange = e => act(() => { state(current).readable = !e.target.checked; }, e.target.checked ? '已标为读不出' : '已取消读不出');
$('wholeLine').onchange = e => act(() => { L.setWhole(pack, store, current, e.target.checked); if (mode === 'lines') whole.selected = state(current).line; },
                                   e.target.checked ? '已标为整行被划：行框就是涂改框' : '已取消整行被划：行框换回原来的');
$('lineSelect').onchange = e => act(() => {
  let line = e.target.value;
  if (line === 'new') { const [l, t, r, b] = bounds(polygon(current)); line = L.newLine(pack, store, page, [[l - 4, t - 4], [r + 4, t - 4], [r + 4, b + 4], [l - 4, b + 4]].map(([x, y]) => [Math.max(0, x), Math.max(0, y)])); }
  L.assign(store, current, line); if (mode === 'lines') whole.selected = line; return `涂改 ${regionName(current)} 归到${lineName(line)}`;
});
$('seen').onchange = e => act(() => { state(current).seen = e.target.checked; }, e.target.checked ? '已标为看过' : '已取消看过');
$('deleteRegionBtn').onclick = () => {
  const id = current;
  if (state(id).deleted) act(() => { L.restoreRegion(pack, store, id); }, `已恢复涂改 ${regionName(id)}`);
  else act(() => { L.removeRegion(store, id); if (mode === 'regions') whole.selected = null; }, `已删除涂改 ${regionName(id)}（可点「恢复这处」或撤销）`);
};
$('nextBtn').onclick = () => { if (!state(current).seen) act(() => { state(current).seen = true; }, '已标为看过'); step(1); };
$('prevBtn').onclick = () => step(-1);
$('filter').onchange = () => { render(); if (!matches(current)) { const next = order().find(matches); if (next) go(next); else status('没有符合筛选的涂改'); } };
$('fitBtn').onclick = () => whole.fit();
$('modeLines').onclick = () => setMode('lines'); $('modeRegions').onclick = () => setMode('regions');
$('brushBtn').onclick = () => setTool('brush'); $('eraserBtn').onclick = () => setTool('eraser');
$('brushSize').oninput = () => { $('brushLabel').textContent = $('brushSize').value + ' 像素'; };
$('redrawBtn').onclick = () => { redraw = !redraw; renderLine(); status(redraw ? '在图上拖出一个矩形，替换选中的行框' : '已取消重画'); };
$('deleteLineBtn').onclick = () => dropSelected();
$('undoBtn').onclick = undo;
document.addEventListener('keydown', e => {
  const el = document.activeElement, typing = ['TEXTAREA', 'SELECT'].includes(el?.tagName) || (el?.tagName === 'INPUT' && !['checkbox', 'range'].includes(el.type));
  if (typing || e.isComposing) return;
  const k = e.key.toLowerCase();
  if ((e.metaKey || e.ctrlKey) && k === 'z') { e.preventDefault(); undo(); }
  else if (e.metaKey || e.ctrlKey || e.altKey) return;
  else if (e.key === 'Enter') { e.preventDefault(); $('nextBtn').click(); }
  else if ((e.key === 'Delete' || e.key === 'Backspace') && mode === 'lines') { e.preventDefault(); dropSelected(); }
  else if (k === 'b') setTool('brush'); else if (k === 'e') setTool('eraser');
  else if (k === 'l') setMode('lines'); else if (k === 'r') setMode('regions');
  else if (e.key === '[' || e.key === ']') { $('brushSize').value = +$('brushSize').value + (e.key === ']' ? 1 : -1); $('brushSize').oninput(); }
  else if (e.key === ' ') { e.preventDefault(); whole.space = true; }
});
document.addEventListener('keyup', e => { if (e.key === ' ') whole.space = false; });
function download(value, name) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 1)], {type: 'application/json'})), a = document.createElement('a');
  a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$('exportBtn').onclick = () => { download(L.exported(pack, store), `整行框选_导出_${new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-')}.json`); status('已导出（下载文件夹）'); };
$('importBtn').onclick = () => $('importFile').click();
$('importFile').onchange = async e => {
  const file = e.target.files[0]; e.target.value = ''; if (!file) return;
  try {
    const data = JSON.parse(await file.text()), other = !!data?.draft?.pack && data.draft.pack !== pack.id;
    if (other && !confirm(`这份导出来自另一版页面（${data.draft.pack}）。涂改都对得上就接着用，导入吗？`)) return;
    const next = L.imported(pack, data, other); remember(); store = next; save(); render(); status('已导入 ' + file.name + '（导入前的状态可撤销）');
  } catch (error) { status('导入失败：' + error.message, true); }
};
$('bindBackup').hidden = !window.showSaveFilePicker;
$('bindBackup').onclick = async () => {
  try { backupHandle = await window.showSaveFilePicker({suggestedName: '整行框选_自动备份.json', types: [{description: 'JSON', accept: {'application/json': ['.json']}}]}); save(); }
  catch (e) { if (e.name !== 'AbortError') status('绑定备份文件失败：' + e.message, true); }
};

(async () => {
  store = await load(); save();
  await go(order().find(id => live(id) && !state(id).seen) || order()[0]);
  whole.resize(true);
  if (carried) status(`已接过旧版页面（${carried}）在这个浏览器里保存的进度`);
})();
