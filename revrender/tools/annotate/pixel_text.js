/* Text mode of the pixel review (pack.task === 'strike_text'): beside the pixel editor, the struck text of the region (typed, with
   SymbolPicker), the model's starting text, the exam question with its standard solution, and one confirmation for pixels and text.
   pixel_editor.js calls these hooks; they use its globals (pack, store, rows, at, state(), row(), show, save, nav, status, touch). */
'use strict';
const TextMode = (() => {
  let picker = null, notice = '';
  const on = () => pack?.task === 'strike_text';
  const judging = () => on() && !!pack.judgment_drafts;
  const FIELDS = [['judgeStruck', 'struck'], ['judgeAfter', 'after'], ['judgeType', 'type']];
  function draftOf(id) { return pack.judgment_drafts?.[id] || null; }
  function judgment(s, id) { return s.judgment || draftOf(id); }
  function showJudgment() {
    // The region's judgment as the annotator left it, or the draft; a region without a draft hides the panel.
    const s = state(), j = judgment(s, row().id);
    $('judgeSlot').hidden = !j;
    if (!j) return;
    for (const [id, key] of FIELDS) $(id).value = j[key] || '';
    $('judgeReason').value = j.reason || '';
    $('judgeState').textContent = s.judgment_confirmed ? '已确认 ✓' : s.judgment ? '已修改，未确认' : '模型初判，请核对';
    $('judgeSlot').classList.toggle('ok', !!s.judgment_confirmed);
  }
  function judgmentChanged() {
    touch(); const s = state();
    s.judgment = { ...(judgment(s, row().id) || {}), ...Object.fromEntries(FIELDS.map(([id, key]) => [key, $(id).value])), reason: $('judgeReason').value };
    s.judgment_confirmed = false; save(); nav(); showJudgment();
  }
  const escape = text => String(text).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
  function markdown(text) {
    // The question's markdown: $$ $$ and $ $ formulas, headings and bold, line breaks; an image link shown as a link.
    const out = [], re = /\$\$([\s\S]+?)\$\$|\$([^$]+?)\$|!\[[^\]]*\]\(([^)]+)\)/g;
    const words = s => escape(s).replace(/^#{1,6} +(.*)$/gm, '<b>$1</b>').replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>').replace(/\n/g, '<br>');
    let last = 0, m;
    while ((m = re.exec(text || ''))) {
      out.push(words(text.slice(last, m.index)));
      out.push(m[1] !== undefined ? SymbolPicker.math(m[1], true) : m[2] !== undefined ? SymbolPicker.math(m[2]) : `<a href="${escape(m[3])}" target="_blank" rel="noopener">[题图]</a>`);
      last = re.lastIndex;
    }
    out.push(words((text || '').slice(last)));
    return out.join('');
  }
  let lastGood = '';
  function preview() {
    // The typed answer rendered; while a formula is half written the last rendering that parsed stays, with a quiet note.
    if (!on()) return;
    const { html, complete } = SymbolPicker.render($('transcription').value);
    if (complete || !lastGood) { $('textPreview').innerHTML = html; if (complete) lastGood = html; }
    $('previewNote').hidden = complete;
  }
  function crop() {
    // The region's own crop beside the rendering, to read the two against each other.
    const r = row(), c = $('textCrop');
    try {
      const data = cropData(r.page, r.box); c.width = data.width; c.height = data.height; c.getContext('2d').putImageData(data, 0, 0); c.hidden = false;
    } catch (e) { c.hidden = true; }
  }
  function question() {
    const page = pack.pages[row().page], q = pack.questions?.[page.exam_question];
    $('questionTitle').textContent = q ? `第 ${q.printed_problem_no} 题（题目与标准解答）` : '（没有对应的题目）';
    $('questionBody').innerHTML = q ? markdown(q.problem_statement_markdown) + '<div class="answer"><b>标准解答：</b><br>' +
      markdown(q.standard_solution_markdown) + '<br><b>答案：</b>' + markdown(q.final_answer) + '</div>' : '';
  }
  async function oldProgress() {
    // What the earlier struck-text page (pack.text_source_pack) kept in this browser: the newest of its IndexedDB and localStorage copies.
    const key = 'struck-text:' + pack.text_source_pack, found = [];
    try {
      const db = await new Promise((resolve, reject) => {
        const r = indexedDB.open('paper2-struck-text', 1);
        r.onupgradeneeded = () => r.result.createObjectStore('progress');
        r.onsuccess = () => resolve(r.result); r.onerror = () => reject(r.error);
      });
      const value = await new Promise((resolve, reject) => { const g = db.transaction('progress').objectStore('progress').get(key); g.onsuccess = () => resolve(g.result); g.onerror = () => reject(g.error); });
      if (value) found.push(value);
    } catch (e) {}
    try { const v = JSON.parse(localStorage.getItem(key) || 'null'); if (v) found.push(v); } catch (e) {}
    found.sort((a, b) => String(b.savedAt || '').localeCompare(String(a.savedAt || '')));
    return found[0] || null;
  }
  function apply(regions) {
    // Records of the earlier page onto this pack's rows: text, unreadable, not a strike, note, confirmation, time, and a box the annotator
    // reshaped there (reshaping clips the pixels to it, which asks for the pixels to be confirmed again).
    let count = 0, reshaped = 0;
    for (const r of rows) {
      const old = regions[r.id];
      if (!old || !(old.done || old.edited)) continue;
      if (old.region_edited && old.region) {
        try { store.rows[r.id] = PixelReview.reshape(ensure(r), old.region, pack.pages[r.page].size); reshaped++; } catch (e) {}
      }
      const s = ensure(r);
      Object.assign(s, { text: old.text ?? s.text, unreadable: !!old.unreadable, not_revision: !!old.not_strike || s.not_revision,
                         text_confirmed: !!old.done, text_seconds: (s.text_seconds || 0) + (old.seconds || 0) });
      if (old.note) s.note = s.note ? s.note + '；' + old.note : old.note;
      count++;
    }
    return { count, reshaped };
  }
  async function setup() {
    if (!on()) return;
    document.body.classList.add('text-mode');
    $('textSide').hidden = false;
    const panel = document.querySelector('.transcription');
    panel.open = true; $('textSlot').append(panel);
    panel.querySelector('summary').firstChild.textContent = '被划掉的字 ';
    panel.querySelector('small').textContent = '只写框里被划掉的原字（改正后的字不写），学生写错也照录；看不清的字写 ?，公式写在 $ $ 里。';
    $('confirm').innerHTML = '确认像素和文字，下一处 <kbd>Enter</kbd>';
    $('appTitle').textContent = '划线精修 + 被划文字';
    $('appSubtitle').textContent = '框只包住被划掉的字；写下被划掉的原字（改正后的字不写），像素有错直接改。';
    picker = SymbolPicker.attach($('transcription'), $('symbolList'), $('symbolBar'), value => {
      if ($('transcription').value !== value) $('transcription').value = value;   // assigning would clear Ctrl+Z's history
      textChanged(); preview(); });
    $('transcription').addEventListener('input', preview);
    $('transcription').addEventListener('keydown', ev => {
      if (picker.keys(ev)) return;
      if (ev.key !== 'Enter' || ev.shiftKey || ev.isComposing) return;
      const area = $('transcription'), at = area.selectionStart;
      ev.preventDefault();
      // Inside a brace (\begin{cases}) Enter starts its next row; Ctrl / ⌘ + Enter confirms from anywhere.
      if (!ev.ctrlKey && !ev.metaKey && SymbolPicker.insideCases(area.value.slice(0, at))) {
        SymbolPicker.put(area, at, area.selectionEnd, '\\\\ '); area.setSelectionRange(at + 3, at + 3); textChanged(); preview(); return;
      }
      $('confirm').click();
    });
    if (judging()) {
      for (const [id, key] of FIELDS) {
        $(id).replaceChildren(...(pack.judgment_values?.[key] || []).map(v => { const o = document.createElement('option'); o.value = o.textContent = v; return o; }));
        $(id).onchange = judgmentChanged;
      }
      $('judgeReason').oninput = judgmentChanged;
      $('judgeReason').addEventListener('keydown', ev => { if (ev.key === 'Enter' && !ev.shiftKey && !ev.isComposing) { ev.preventDefault(); $('confirm').click(); } });
      $('confirm').innerHTML = '确认像素、文字和对错，下一处 <kbd>Enter</kbd>';
    }
    window.katexLoaded = () => { picker.paintBar(); preview(); if (row()) question(); };
    if (window.katex) window.katexLoaded();
    // A new starting text reaches a region the annotator has neither confirmed nor changed (its text is still an earlier starting text).
    let redrafted = 0;
    for (const r of rows) {
      const s = ensure(r), earlier = r.previous_prefill || [];
      if (!s.text_confirmed && s.text !== r.prefill && earlier.includes(s.text || '')) { s.text = r.prefill; redrafted++; }
    }
    if (redrafted) { save(); notice = `已用新的模型初稿替换 ${redrafted} 处还没改过的文字。`; }
    if (!store.textMigrated && pack.text_source_pack) {
      const old = await oldProgress();
      const result = old ? apply(old.regions || {}) : { count: 0, reshaped: 0 };
      store.textMigrated = { from: pack.text_source_pack, at: new Date().toISOString(), ...result };
      save();
      if (result.count) notice = `已接入旧「补被划文字」页的 ${result.count} 处进度` + (result.reshaped ? `；其中 ${result.reshaped} 处按你调过的框裁切，像素需要再确认一次。` : '。');
    }
  }
  function onShow() {
    if (!on()) return;
    const r = row();
    $('prefillLine').innerHTML = r.prefill ? '模型初稿：<code>' + escape(r.prefill) + '</code>' : '模型初稿：（空）';
    lastGood = ''; preview(); crop(); question(); if (judging()) showJudgment();
    const area = $('transcription'); area.focus(); area.setSelectionRange(area.value.length, area.value.length);
    if (notice) { const text = notice; notice = ''; setTimeout(() => status(text), 1200); }
  }
  function confirmText() {
    // true when the region may be confirmed: struck text written, or unreadable, or not a strike at all.
    if (!on()) return true;
    const s = state();
    if (!s.text?.trim() && !s.unreadable && !s.not_revision) { status('请先写被划掉的字，或勾选“整处无法辨认”。', true); $('transcription').focus(); return false; }
    s.text_confirmed = true;
    const j = judging() && judgment(s, row().id);
    if (j) { s.judgment = { ...j }; s.judgment_confirmed = true; }
    return true;
  }
  // The region list filtered by judgment (any field 不确定, a type, a value of 被划内容 / 改后, not yet confirmed); remembered per browser.
  const FILTER_KEY = 'paper2-judgment-filter';
  let filter = '', box = null;
  try { filter = localStorage.getItem(FILTER_KEY) || ''; } catch (e) {}
  function matches(r, f = filter) {
    if (!f || !judging()) return true;
    const s = ensure(r), j = judgment(s, r.id);
    if (!j) return false;
    if (f === 'uncertain') return FIELDS.some(([, key]) => j[key] === '不确定');
    if (f === 'unconfirmed') return !s.judgment_confirmed;
    const [key, value] = f.split(':');
    return j[key] === value;
  }
  const filtering = () => !!filter && judging();
  function filterBox() {
    // The <select> above the region list, its counts brought up to date.
    if (!judging()) return null;
    if (!box) {
      box = document.createElement('label'); box.className = 'judge-filter';
      box.append('按对错筛选', document.createElement('select'));
      box.lastChild.onchange = ev => { filter = ev.target.value; try { localStorage.setItem(FILTER_KEY, filter); } catch (e) {} nav(); };
    }
    const count = f => rows.filter(r => matches(r, f)).length, values = pack.judgment_values || {};
    const options = [['', `全部（${rows.length}）`], ['uncertain', `任一栏不确定（${count('uncertain')}）`], ['unconfirmed', `对错未确认（${count('unconfirmed')}）`],
      ...(values.type || []).map(v => [`type:${v}`, `类型：${v}（${count('type:' + v)}）`]),
      ...[['struck', '被划内容'], ['after', '改后']].flatMap(([key, name]) => (values[key] || []).map(v => [`${key}:${v}`, `${name}：${v}（${count(key + ':' + v)}）`]))];
    if (!options.some(([v]) => v === filter)) filter = '';
    box.lastChild.replaceChildren(...options.map(([v, text]) => { const o = document.createElement('option'); o.value = v; o.textContent = text; return o; }));
    box.lastChild.value = filter;
    return box;
  }
  function step(from, direction) {
    // The next region in that direction the filter shows; -1 when there is none.
    for (let k = from + direction; k >= 0 && k < rows.length; k += direction) if (matches(rows[k])) return k;
    return -1;
  }
  function tag(r) { const j = judging() && judgment(ensure(r), r.id); return j?.type ? ' · ' + j.type : ''; }
  // Done: pixels confirmed; in text mode the text too; with judgment drafts, the judgment of a region that has one.
  const done = (s, r) => !on() ? s.confirmed : s.confirmed && s.text_confirmed && (!judging() || !r || !judgment(s, r.id) || !!s.judgment_confirmed);
  const judged = r => { const s = ensure(r); return !!(judging() && judgment(s, r.id) && s.judgment_confirmed); };
  return { on, setup, onShow, confirmText, preview, done, judging, judged, matches, filtering, filterBox, step, tag };
})();
