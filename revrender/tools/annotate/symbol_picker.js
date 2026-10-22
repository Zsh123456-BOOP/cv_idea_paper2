/* LaTeX symbols for typing struck text: \ opens a list (command, how it looks, what it means), a bar of common ones in groups, with
   the ones this browser used last first.  Rendering uses KaTeX when it has loaded (window.katex), the command otherwise. */
'use strict';
const SymbolPicker = (() => {
  // [name, inserted (▮ marks where the caret goes; a leading § is plain text, put in \text{} inside a formula), meaning and search words]
  const CIRCLED = '①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳';
  const SYMBOLS = [
    ['because', '\\because ', '因为'], ['therefore', '\\therefore ', '所以'], ['angle', '\\angle ', '角'], ['triangle', '\\triangle ', '三角形'],
    ['parallel', '\\parallel ', '平行'], ['perp', '\\perp ', '垂直'],
    ['Rt', '\\mathrm{Rt}\\triangle ', '直角三角形（Rt△）'], ['parallelogram', '\\text{▱}', '平行四边形（▱）'],
    ['quan', '\\text{≌}', '全等（课本写法 ≌）'], ['xiangsi', '\\backsim ', '相似（课本写法 ∽）'], ['cong', '\\cong ', '全等（≅）'], ['sim', '\\sim ', '相似（∼）'],
    ['prime', '^{\\prime}', '撇（A′）'], ['S_triangle', 'S_{\\triangle ▮}', '三角形面积'], ['odot', '\\odot ', '圆（⊙）'],
    ['circ', '^{\\circ}', '度（°）'], ['frac', '\\frac{▮}{}', '分数'], ['sqrt', '\\sqrt{▮}', '根号'], ['sqrt[n]', '\\sqrt[▮]{}', 'n 次根号'],
    ['^', '^{▮}', '上标／乘方'], ['_', '_{▮}', '下标'], ['^2', '^{2}', '平方'], ['^3', '^{3}', '立方'],
    ['times', '\\times ', '乘号'], ['div', '\\div ', '除号'], ['cdot', '\\cdot ', '点乘'], ['pm', '\\pm ', '正负'], ['mp', '\\mp ', '负正'],
    ['neq', '\\neq ', '不等于'], ['approx', '\\approx ', '约等于'], ['leq', '\\leq ', '小于等于'], ['geq', '\\geq ', '大于等于'],
    ['lt', '< ', '小于'], ['gt', '> ', '大于'], ['equiv', '\\equiv ', '恒等于'],
    ['left|', '\\left|▮\\right|', '绝对值'], ['left(', '\\left(▮\\right)', '括号'], ['overline', '\\overline{▮}', '上划线／线段'],
    ['overset{\\frown}', '\\overset{\\frown}{▮}', '弧'], ['vec', '\\vec{▮}', '向量'], ['square', '\\square ', '正方形'],
    ['pi', '\\pi ', '圆周率'], ['alpha', '\\alpha ', '阿尔法'], ['beta', '\\beta ', '贝塔'], ['gamma', '\\gamma ', '伽马'], ['theta', '\\theta ', '西塔'],
    ['infty', '\\infty ', '无穷'], ['in', '\\in ', '属于'], ['notin', '\\notin ', '不属于'], ['subseteq', '\\subseteq ', '子集'],
    ['cap', '\\cap ', '交集'], ['cup', '\\cup ', '并集'], ['varnothing', '\\varnothing ', '空集'],
    ['rightarrow', '\\rightarrow ', '箭头'], ['Rightarrow', '\\Rightarrow ', '推出'], ['Leftrightarrow', '\\Leftrightarrow ', '等价'],
    ['cdots', '\\cdots ', '省略号'], ['%', '\\% ', '百分号'],
    ['cases', '\\begin{cases}▮\\end{cases}', '大括号 方程组 联立 条件（里面 Enter 换一行）'],
    ['sin', '\\sin ', '正弦'], ['cos', '\\cos ', '余弦'], ['tan', '\\tan ', '正切'], ['text', '\\text{▮}', '公式里写中文'],
    ...['SSS', 'SAS', 'ASA', 'AAS', 'HL'].map(x => [x, '§' + x, '全等判定 ' + x]),
    ...[...CIRCLED].map((c, i) => [String(i + 1), '§' + c, `带圈数字 圈${i + 1} ${c}`]),
  ];
  // The bar, in groups; '①' opens the palette of circled numbers.
  const GROUPS = [['几何', ['because', 'therefore', 'angle', 'triangle', 'Rt', 'parallelogram', 'parallel', 'perp', 'quan', 'xiangsi', 'circ', 'overline']],
                  ['关系', ['neq', 'leq', 'geq', 'approx', 'Rightarrow']], ['运算', ['times', 'div', 'pm', 'pi']],
                  ['结构', ['cases', 'frac', 'sqrt', '^', '_', 'left|']], ['编号', ['1', 'SSS', 'SAS', 'ASA', 'AAS', 'HL']]];
  const BAR = GROUPS.flatMap(([, names]) => names);
  const RECENT_KEY = 'paper2-symbol-recent', RECENT = 6;
  const escape = text => text.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
  const plain = snippet => snippet.startsWith('§');
  const sample = snippet => plain(snippet) ? snippet.slice(1) : snippet.replace('▮\\end{cases}', 'x\\\\ y\\end{cases}').replace('▮', 'x').replace(/\{\}/g, '{y}').trim();
  function math(tex, display = false) {
    return window.katex ? window.katex.renderToString(tex, { throwOnError: false, strict: 'ignore', displayMode: display, output: 'html' }) : escape(tex);
  }
  const tall = tex => /\\begin\{|\\\\/.test(tex);           // a formula of several rows: shown as a block, not squeezed into the line
  function render(text) {
    // {html, complete} of a typed answer: what is between $ $ as formulas (page_rules.js mathParts), the rest as words, line breaks kept;
    // complete is false while a formula does not parse yet (the caller keeps showing the last one that did).
    const parts = mathParts(text || '');
    if (!parts.length) return { html: '<span class="none">（还没有字）</span>', complete: true };
    let complete = true;
    const html = parts.map(p => {
      if (!p.math) return escape(p.text).replace(/\n/g, '<br>');
      if (!window.katex) return escape(p.text);
      try { return window.katex.renderToString(p.text, { throwOnError: true, strict: 'ignore', displayMode: tall(p.text), output: 'html' }); }
      catch (e) { complete = false; return '<span class="unparsed">' + escape(p.text) + '</span>'; }
    }).join('');
    return { html, complete };
  }
  function rendered(text) { return render(text).html; }
  function insideMath(text) { return (text.replace(/\\\$/g, '').match(/\$/g) || []).length % 2 === 1; }
  function insideCases(text) { return (text.match(/\\begin\{cases\}/g) || []).length > (text.match(/\\end\{cases\}/g) || []).length; }
  function put(area, from, to, body) {
    // Through the browser's own editing when it can, so Ctrl+Z takes an insertion back like typing.
    area.focus(); area.setSelectionRange(from, to);
    let done = false;
    try { done = document.execCommand && document.execCommand('insertText', false, body); } catch (e) {}
    if (!done || area.value.slice(from, from + body.length) !== body) area.value = area.value.slice(0, from) + body + area.value.slice(to);
  }
  function insert(area, snippet, from, to, onChange) {
    // The snippet at [from, to): a formula wrapped in $ $ when the caret is not already in one, plain text put in \text{} when it is;
    // the caret goes to ▮.
    const inMath = insideMath(area.value.slice(0, from));
    let body = plain(snippet) ? (inMath ? '\\text{' + snippet.slice(1) + '}' : snippet.slice(1))
                              : (inMath ? snippet : '$' + snippet.replace(/ $/, '') + '$');
    const caret = body.includes('▮') ? body.indexOf('▮') : body.length;
    body = body.replace('▮', '');
    put(area, from, to, body);
    area.setSelectionRange(from + caret, from + caret);
    remember(snippet);
    onChange(area.value);
  }
  function recent() { try { return JSON.parse(localStorage.getItem(RECENT_KEY) || '[]').filter(s => SYMBOLS.some(x => x[1] === s)); } catch (e) { return []; } }
  function remember(snippet) {
    try { localStorage.setItem(RECENT_KEY, JSON.stringify([snippet, ...recent().filter(s => s !== snippet)].slice(0, RECENT))); } catch (e) {}
  }
  function attach(area, popup, bar, onChange) {
    let open = false, items = [], index = 0, start = 0;
    const close = () => { open = false; popup.hidden = true; };
    function list() {
      popup.hidden = false;
      popup.replaceChildren(...items.map(([name, snippet, zh], i) => {
        const row = document.createElement('div'); row.className = 'ac-item' + (i === index ? ' on' : '');
        row.innerHTML = `<code>\\${escape(name)}</code><span class="ac-sym">${plain(snippet) ? escape(sample(snippet)) : math(sample(snippet))}</span><span>${escape(zh)}</span>`;
        row.onmousedown = ev => { ev.preventDefault(); index = i; choose(); };
        return row;
      }));
      popup.children[index].scrollIntoView({ block: 'nearest' });
    }
    function suggest() {
      const m = /\\([A-Za-z0-9^_|(\[\]{%一-龥]*)$/.exec(area.value.slice(0, area.selectionStart));
      if (!m) return close();
      const q = m[1], rank = ([name, , zh]) => !q ? 0 : name.startsWith(q) ? 0 : name.toLowerCase().includes(q.toLowerCase()) ? 1 : zh.includes(q) ? 2 : -1;
      items = SYMBOLS.map(x => [rank(x), x]).filter(([k]) => k >= 0).sort((a, b) => a[0] - b[0]).map(([, x]) => x);
      if (q === '{') items = SYMBOLS.filter(x => x[0] === 'cases');
      if (!items.length) return close();
      open = true; index = 0; start = m.index; list();
    }
    function choose() { const [, snippet] = items[index]; close(); insert(area, snippet, start, area.selectionStart, onChange); }
    area.addEventListener('input', suggest);
    area.addEventListener('click', suggest);
    area.addEventListener('blur', () => setTimeout(close, 150));
    const keys = ev => {
      // Called first by the textarea's keydown: true when the list took the key (Enter then inserts, it does not confirm).
      if (!open) return false;
      if (ev.key === 'ArrowDown' || ev.key === 'ArrowUp') { ev.preventDefault(); index = (index + (ev.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length; list(); return true; }
      if ((ev.key === 'Enter' || ev.key === 'Tab') && !ev.isComposing) { ev.preventDefault(); choose(); return true; }
      if (ev.key === 'Escape') { ev.preventDefault(); close(); return true; }
      return false;
    };
    const button = (snippet, title, label) => {
      const b = document.createElement('button'); b.type = 'button'; b.title = title; b.dataset.snippet = snippet;
      if (label) b.dataset.label = label;
      b.onmousedown = ev => ev.preventDefault();
      b.onclick = () => { insert(area, snippet, area.selectionStart, area.selectionEnd, onChange); palette.hidden = true; paintBar(); };
      return b;
    };
    const palette = document.createElement('div'); palette.className = 'symbol-palette'; palette.hidden = true;
    palette.append(...[...CIRCLED].map((c, i) => button('§' + c, `带圈数字 ${i + 1}（\\${i + 1}）`)));
    function paintBar() {
      const groups = [['最近', recent()], ...GROUPS.map(([g, names]) => [g, names.map(n => SYMBOLS.find(x => x[0] === n)[1])])];
      bar.replaceChildren(...groups.filter(([, list]) => list.length).map(([g, list]) => {
        const box = document.createElement('span'); box.className = 'symbol-group';
        const label = document.createElement('small'); label.textContent = g; box.append(label);
        for (const snippet of list) {
          const [name, , zh] = SYMBOLS.find(x => x[1] === snippet);
          if (name === '1' && g !== '最近') {
            const b = document.createElement('button'); b.type = 'button'; b.title = '带圈数字 ①…⑳（也可打 \\1、\\圈3）'; b.textContent = '①…';
            b.onmousedown = ev => ev.preventDefault(); b.onclick = () => { palette.hidden = !palette.hidden; };
            box.append(b); continue;
          }
          box.append(button(snippet, `${zh}（\\${name}）`));
        }
        return box;
      }), palette);
      for (const b of bar.querySelectorAll('button[data-snippet]')) b.innerHTML = plain(b.dataset.snippet) ? escape(sample(b.dataset.snippet)) : math(sample(b.dataset.snippet));
    }
    paintBar();
    return { keys, paintBar, close };
  }
  return { SYMBOLS, BAR, CIRCLED, attach, insert, put, render, rendered, math, insideMath, insideCases };
})();
if (typeof module !== 'undefined') module.exports = SymbolPicker;
