/* The line page (line_review.py): draw each struck region's whole line on the question, correct the struck text, mark what was seen.
   The state rules are line_rules.js; the polygon handles are pixel_overview.js; saving is IndexedDB (library_store.js) and
   localStorage, each read back, plus a bound backup file. */
'use strict';
const $ = id => document.getElementById(id), L = LineRules;
const pack = JSON.parse($('line-pack').textContent), KEY = 'line-review-' + pack.id;
const ROWS = L.index(pack).regions, ORDER = pack.pages.flatMap(p => p.regions.map(r => r.id));
const LETTERS = 'abcdefghijklmnopqrstuvwxyz';
let store, page = 0, current = ORDER[0], redraw = false, history = [], textMark = null, saveFailed = false;
let backupHandle = null, backupPending = null, backupWriting = false, down = null;
const images = new Map();

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
  return L.initial(pack);
}

// ---------- changes ----------
function status(text, error = false) { $('status').textContent = text; $('status').className = error ? 'error' : ''; }
function remember() { history.push(JSON.stringify(store)); if (history.length > 200) history.shift(); }
function act(change, message) {
  remember();
  try { const said = change(); save(); render(); status(said || message); }
  catch (e) { store = JSON.parse(history.pop()); render(); status(e.message, true); }
}
function undo() {
  if (!history.length) return status('没有可撤销的修改');
  store = JSON.parse(history.pop()); textMark = null; save(); render(); status('已撤销');
}

// ---------- what is shown ----------
const state = id => store.regions[id];
const corrected = id => state(id).text !== ROWS[id].text || state(id).readable !== ROWS[id].readable;
function matches(id) {
  const f = $('filter').value;
  return f === 'all' || (f === 'unseen' && !state(id).seen) || (f === 'wrong' && ROWS[id].wrong) || (f === 'whole' && state(id).whole_line) || (f === 'changed' && corrected(id));
}
const centre = region => { const [l, t, r, b] = bounds(region); return [(l + r) / 2, (t + b) / 2]; };
function pageLines(pi = page) {
  return Object.keys(store.lines).filter(id => store.lines[id].page === pi)
    .sort((a, b) => bounds(store.lines[a].polygon)[1] - bounds(store.lines[b].polygon)[1] || bounds(store.lines[a].polygon)[0] - bounds(store.lines[b].polygon)[0]);
}
const lineName = id => '第 ' + (pageLines(store.lines[id].page).indexOf(id) + 1) + ' 行';
const regionName = id => LETTERS[pack.pages[ROWS[id].page].regions.findIndex(r => r.id === id)] || '?';

function renderPages() {
  const list = $('pages'); list.replaceChildren();
  pack.pages.forEach((p, pi) => {
    const ids = p.regions.map(r => r.id); if (!ids.some(matches) && pi !== page) return;
    const seen = ids.filter(id => state(id).seen).length, d = document.createElement('div');
    d.className = (pi === page ? 'on ' : '') + (seen === ids.length ? 'done' : '');
    const [writer, ...rest] = p.id.split('_');
    d.innerHTML = `<span>${pi + 1}. ${writer.slice(0, 6)}…_${rest.join('_')}</span><span class="count">${seen}/${ids.length}</span>`;
    d.onclick = () => go(ids.find(matches) || ids[0]);
    list.append(d);
  });
  list.querySelector('.on')?.scrollIntoView({block: 'nearest'});
}
function renderChips() {
  const ids = pack.pages[page].regions.map(r => r.id);
  $('pageInfo').textContent = `${pack.pages[page].id} · ${ids.filter(id => state(id).seen).length}/${ids.length} 已看 · ${pageLines().filter(id => L.members(store, id).length).length} 行`;
  $('chips').replaceChildren(...ids.map(id => {
    const c = document.createElement('span');
    c.className = 'chip' + (state(id).seen ? ' seen' : '') + (ROWS[id].wrong ? ' wrong' : '') + (id === current ? ' on' : '') + (matches(id) ? '' : ' off');
    c.textContent = regionName(id); c.title = ROWS[id].wrong ? '上限读错（按原答案）' : ''; c.onclick = () => go(id); return c;
  }));
}
function renderDetail() {
  const id = current, s = state(id), o = ROWS[id];
  $('regionName').textContent = '涂改 ' + regionName(id); $('regionWhere').textContent = `${lineName(s.line)} · ${o.mode === 'keep' ? '模式 2（保原字）' : '模式 1（擦划线）'}`;
  for (const [img, src] of [[$('cropRaw'), o.raw], [$('cropErased'), o.erased]]) if (img.dataset.id !== id) {
    img.dataset.id = id; img.onload = () => { img.style.width = Math.min(img.naturalWidth * 2, 200) + 'px'; }; img.src = src;
  }
  $('readings').replaceChildren(...Object.entries(o.readings).map(([label, text]) => {
    const r = document.createElement('div'); r.className = 'r';
    const bad = label === pack.check && o.wrong;
    r.innerHTML = `<small>${pack.labels[label] || label}</small><span>${SymbolPicker.rendered(text)} <small><code></code></small>${bad ? ' <span class="bad">与原答案不一致</span>' : ''}</span>`;
    r.querySelector('code').textContent = text; return r;
  }));
  if ($('text').value !== s.text || document.activeElement !== $('text')) $('text').value = s.text;
  preview();
  $('origText').textContent = o.text || '（空）'; $('origReadable').textContent = o.readable ? '' : '· 原来勾了读不出';
  $('changedMark').textContent = corrected(id) ? '已改' : ''; $('changedMark').className = corrected(id) ? 'changed' : ''; $('revertText').hidden = !corrected(id);
  $('unreadable').checked = !s.readable; $('wholeLine').checked = !!s.whole_line; $('seen').checked = !!s.seen;
  const prefill = pack.pages[o.page].lines.find(l => l.regions.includes(id));
  $('wholeGuess').textContent = prefill?.whole ? '（程序猜是整行，请确认）' : '';
  const select = $('lineSelect'); select.replaceChildren();
  for (const line of pageLines()) {
    const n = L.members(store, line).length, opt = new Option(`${lineName(line)}（${n} 处${line === s.line ? '，当前' : ''}）`, line);
    select.append(opt);
  }
  select.append(new Option('新建一行', 'new')); select.value = s.line; select.disabled = !!s.whole_line;
}
let lastGood = '';
function preview() {
  const {html, complete} = SymbolPicker.render($('text').value);
  if (complete || !lastGood) { $('preview').innerHTML = html; if (complete) lastGood = html; }
  $('previewNote').hidden = complete;
}
function renderLine() {
  const id = whole.selected, panel = $('linePanel');
  $('redrawBtn').disabled = !id; $('deleteLineBtn').disabled = !id || L.members(store, id).length > 0;
  $('redrawBtn').textContent = redraw ? '在图上拖出新的行框…' : '重画选中行框';
  if (!id || !store.lines[id]) { panel.textContent = '在图上点一个行框。'; return; }
  const inside = L.members(store, id), line = store.lines[id];
  panel.innerHTML = `<b>${lineName(id)}</b> · ${line.polygon.length} 个点 · ${line.adjusted ? '人工调过' : '程序预填'}${inside.some(r => state(r).whole_line) ? ' · 整行被划' : ''}<ul></ul>`;
  panel.querySelector('ul').replaceChildren(...(inside.length ? inside.map(r => {
    const li = document.createElement('li'); li.textContent = '涂改 ' + regionName(r) + (state(r).seen ? '（已看过）' : ''); li.style.cursor = 'pointer'; li.onclick = () => go(r); return li;
  }) : [Object.assign(document.createElement('li'), {textContent: '空行（导出时不算），可删除'})]));
}
function render() {
  const seen = ORDER.filter(id => state(id).seen).length, fixed = ORDER.filter(corrected).length;
  $('progress').textContent = `已看 ${seen} / ${ORDER.length} 处 · 改过答案 ${fixed} 处 · 共 ${pack.pages.length} 题`;
  renderPages(); renderChips(); renderDetail(); renderLine(); whole.draw();
}

// ---------- the question image ----------
async function imageOf(pi) {
  if (!images.has(pi)) { const im = new Image(); im.src = pack.pages[pi].image; await im.decode(); images.set(pi, im); }
  return images.get(pi);
}
const whole = new PixelOverview($('whole'), {
  rows: () => pageLines().map(id => ({id, region: store.lines[id].polygon, adjusted: store.lines[id].adjusted,
                                      empty: !L.members(store, id).length, wholeLine: L.members(store, id).some(r => state(r).whole_line)})),
  colour: (r, selected) => selected ? '#235cca' : r.empty ? '#9aa3ad' : r.wholeLine ? '#137660' : r.adjusted ? '#7b4fc9' : '#3f8fd8',
  overlay: (cx, scale) => {
    for (const o of pack.pages[page].others) {
      cx.beginPath(); o.region.forEach((p, i) => i ? cx.lineTo(...p) : cx.moveTo(...p)); cx.closePath();
      cx.setLineDash([3 / scale, 3 / scale]); cx.lineWidth = 1 / scale; cx.strokeStyle = '#9aa3ad'; cx.stroke();
    }
    for (const r of pack.pages[page].regions) {
      cx.beginPath(); r.region.forEach((p, i) => i ? cx.lineTo(...p) : cx.moveTo(...p)); cx.closePath();
      if (r.id === current) { cx.fillStyle = 'rgba(255,200,0,.28)'; cx.fill(); }
      cx.setLineDash([4 / scale, 3 / scale]); cx.lineWidth = (r.id === current ? 2 : 1.2) / scale; cx.strokeStyle = '#c8372d'; cx.stroke();
      const [, , right, bottom] = bounds(r.region); cx.setLineDash([]);
      cx.font = `${12 / scale}px sans-serif`; cx.fillStyle = '#c8372d'; cx.fillText(regionName(r.id), right + 2 / scale, bottom);
    }
    cx.setLineDash([]);
  },
  select: () => renderLine(), activity: () => {}, open: () => {},
  zoom: scale => { $('zoomLabel').textContent = Math.round(scale * 100) + '%'; },
  reshape: (id, polygon) => act(() => L.reshape(pack, store, id, polygon), lineName(id) + '的行框已调整'),
  create: polygon => {
    if (redraw && whole.selected) { const id = whole.selected; redraw = false; return act(() => L.reshape(pack, store, id, polygon), lineName(id) + '已重画'); }
    let made = null;
    act(() => {
      made = L.newLine(pack, store, page, polygon); const moved = [];
      for (const r of pack.pages[page].regions) if (pointInPolygon(centre(r.region), polygon) && !state(r.id).whole_line) { state(r.id).line = made; moved.push(regionName(r.id)); }
      return `新建了${lineName(made)}` + (moved.length ? `，涂改 ${moved.join('、')} 归到这一行` : '（空行：在右边「所属行」里选它，或删掉）');
    });
    if (made && store.lines[made]) whole.select(made);
  },
});
$('whole').addEventListener('pointerdown', e => { down = [e.clientX, e.clientY]; });
$('whole').addEventListener('pointerup', e => {
  // A click (not a drag) inside a region opens it on the right, and selects its line.
  if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4 || e.button !== 0 || e.shiftKey) return;
  const p = whole.point(e), hit = pack.pages[page].regions.filter(r => pointInPolygon(p, r.region)).sort((a, b) => polygonArea(a.region) - polygonArea(b.region))[0];
  if (hit) go(hit.id);
});

// ---------- moving about ----------
async function go(id) {
  if (!id) return; textMark = null;
  const pi = ROWS[id].page, turned = pi !== page || !whole.image; current = id; page = pi;
  if (turned) { const im = await imageOf(pi); if (current !== id) return; whole.setImage(im, state(id).line, true); }
  else whole.select(state(id).line);
  render();
}
function step(direction) {
  const list = ORDER.filter(id => matches(id) || id === current), at = list.indexOf(current);
  const next = list[at + direction];
  if (next) go(next); else status(direction > 0 ? '这已经是筛选后的最后一处' : '这已经是第一处');
}

// ---------- controls ----------
$('text').addEventListener('focus', () => { textMark = null; });
$('text').addEventListener('input', () => {
  if (textMark !== current) { remember(); textMark = current; }        // one undo step per stay in the box
  state(current).text = $('text').value; preview(); save();
  const id = current; $('changedMark').textContent = corrected(id) ? '已改' : ''; $('changedMark').className = corrected(id) ? 'changed' : ''; $('revertText').hidden = !corrected(id);
  renderChips(); $('progress').textContent = `已看 ${ORDER.filter(x => state(x).seen).length} / ${ORDER.length} 处 · 改过答案 ${ORDER.filter(corrected).length} 处 · 共 ${pack.pages.length} 题`;
});
const picker = SymbolPicker.attach($('text'), $('symbolList'), $('symbolBar'), value => {
  if ($('text').value !== value) $('text').value = value;
  $('text').dispatchEvent(new Event('input'));
});
$('text').addEventListener('keydown', e => { if (picker.keys(e)) e.stopPropagation(); });
$('revertText').onclick = () => act(() => { state(current).text = ROWS[current].text; state(current).readable = ROWS[current].readable; }, '已恢复原来的答案');
$('unreadable').onchange = e => act(() => { state(current).readable = !e.target.checked; }, e.target.checked ? '已标为读不出' : '已取消读不出');
$('wholeLine').onchange = e => act(() => { L.setWhole(pack, store, current, e.target.checked); whole.selected = state(current).line; },
                               e.target.checked ? '已标为整行被划：行框就是涂改框' : '已取消整行被划：行框换回原来的');
$('lineSelect').onchange = e => act(() => {
  let line = e.target.value;
  if (line === 'new') { const [l, t, r, b] = bounds(ROWS[current].region); line = L.newLine(pack, store, page, [[l - 4, t - 4], [r + 4, t - 4], [r + 4, b + 4], [l - 4, b + 4]].map(([x, y]) => [Math.max(0, x), Math.max(0, y)])); }
  L.assign(store, current, line); whole.selected = line; return `涂改 ${regionName(current)} 归到${lineName(line)}`;
});
$('seen').onchange = e => act(() => { state(current).seen = e.target.checked; }, e.target.checked ? '已标为看过' : '已取消看过');
$('nextBtn').onclick = () => { if (!state(current).seen) act(() => { state(current).seen = true; }, '已标为看过'); step(1); };
$('prevBtn').onclick = () => step(-1);
$('filter').onchange = () => { render(); if (!matches(current)) { const next = ORDER.find(matches); if (next) go(next); else status('没有符合筛选的涂改'); } };
$('fitBtn').onclick = () => whole.fit();
$('redrawBtn').onclick = () => { redraw = !redraw; renderLine(); status(redraw ? '在图上拖出一个矩形，替换选中的行框' : '已取消重画'); };
$('deleteLineBtn').onclick = () => { const id = whole.selected; act(() => { L.removeLine(store, id); whole.selected = null; }, '已删除空行'); };
$('undoBtn').onclick = undo;
document.addEventListener('keydown', e => {
  const typing = ['TEXTAREA', 'INPUT', 'SELECT'].includes(document.activeElement?.tagName);
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z' && !typing) { e.preventDefault(); undo(); }
  else if (e.key === 'Enter' && !typing && !e.isComposing) { e.preventDefault(); $('nextBtn').click(); }
  else if (e.key === ' ' && !typing) { e.preventDefault(); whole.space = true; }
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
  try { const next = L.imported(pack, JSON.parse(await file.text())); remember(); store = next; save(); render(); status('已导入 ' + file.name + '（导入前的状态可撤销）'); }
  catch (error) { status('导入失败：' + error.message, true); }
};
$('bindBackup').hidden = !window.showSaveFilePicker;
$('bindBackup').onclick = async () => {
  try { backupHandle = await window.showSaveFilePicker({suggestedName: '整行框选_自动备份.json', types: [{description: 'JSON', accept: {'application/json': ['.json']}}]}); save(); }
  catch (e) { if (e.name !== 'AbortError') status('绑定备份文件失败：' + e.message, true); }
};

(async () => {
  store = await load(); save();
  await go(ORDER.find(id => !state(id).seen) || ORDER[0]);
  whole.resize(true);
})();
