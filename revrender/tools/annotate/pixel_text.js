/* Text mode of the pixel review (pack.task === 'strike_text'): beside the pixel editor, the struck text of the region (typed, with
   SymbolPicker), the model's starting text, the exam question with its standard solution, and one confirmation for pixels and text.
   pixel_editor.js calls these hooks; they use its globals (pack, store, rows, at, state(), row(), show, save, nav, status, touch). */
'use strict';
const TextMode = (() => {
  let picker = null, notice = '';
  const on = () => pack?.task === 'strike_text';
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
  function preview() { if (on()) $('textPreview').innerHTML = SymbolPicker.rendered($('transcription').value); }
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
    picker = SymbolPicker.attach($('transcription'), $('symbolList'), $('symbolBar'), value => { $('transcription').value = value; textChanged(); preview(); });
    $('transcription').addEventListener('input', preview);
    $('transcription').addEventListener('keydown', ev => {
      if (picker.keys(ev)) return;
      if (ev.key === 'Enter' && !ev.shiftKey && !ev.isComposing) { ev.preventDefault(); $('confirm').click(); }
    });
    window.katexLoaded = () => { picker.paintBar(); preview(); if (row()) question(); };
    if (window.katex) window.katexLoaded();
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
    preview(); question();
    const area = $('transcription'); area.focus(); area.setSelectionRange(area.value.length, area.value.length);
    if (notice) { const text = notice; notice = ''; setTimeout(() => status(text), 0); }
  }
  function confirmText() {
    // true when the region may be confirmed: struck text written, or unreadable, or not a strike at all.
    if (!on()) return true;
    const s = state();
    if (!s.text?.trim() && !s.unreadable && !s.not_revision) { status('请先写被划掉的字，或勾选“整处无法辨认”。', true); $('transcription').focus(); return false; }
    s.text_confirmed = true;
    return true;
  }
  const done = s => on() ? s.confirmed && s.text_confirmed : s.confirmed;
  return { on, setup, onShow, confirmText, preview, done };
})();
