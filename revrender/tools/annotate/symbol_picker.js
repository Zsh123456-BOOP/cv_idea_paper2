/* LaTeX symbols for typing struck text: \ opens a list (command, how it looks, what it means), a bar of common ones.
   Rendering uses KaTeX when it has loaded (window.katex), the command otherwise. */
'use strict';
const SymbolPicker = (() => {
  // [name, LaTeX inserted (▮ marks where the caret goes), meaning]
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
    ['cdots', '\\cdots ', '省略号'], ['%', '\\% ', '百分号'], ['cases', '\\begin{cases}▮\\\\ \\end{cases}', '方程组（\\\\ 换行）'],
    ['sin', '\\sin ', '正弦'], ['cos', '\\cos ', '余弦'], ['tan', '\\tan ', '正切'], ['text', '\\text{▮}', '公式里写中文'],
  ];
  const BAR = ['because', 'therefore', 'angle', 'triangle', 'Rt', 'parallelogram', 'parallel', 'perp', 'quan', 'xiangsi', 'circ', 'times', 'div', 'pm', 'neq', 'leq', 'geq', 'frac', 'sqrt', '^', 'pi'];
  const escape = text => text.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
  const sample = snippet => snippet.replace('▮', 'x').replace(/\{\}/g, '{y}').trim();
  function math(tex, display = false) {
    return window.katex ? window.katex.renderToString(tex, { throwOnError: false, strict: 'ignore', displayMode: display, output: 'html' }) : escape(tex);
  }
  function rendered(text) {
    // A typed answer for display: what is between $ $ as formulas (page_rules.js mathParts), the rest as words.
    const parts = mathParts(text);
    if (!parts.length) return '<span class="none">（还没有字）</span>';
    return parts.map(p => p.math ? math(p.text) : escape(p.text)).join('');
  }
  function insideMath(text) { return (text.replace(/\\\$/g, '').match(/\$/g) || []).length % 2 === 1; }
  function insert(area, snippet, from, to, onChange) {
    // The snippet at [from, to), wrapped in $ $ when the caret is not already in a formula; the caret goes to ▮.
    const value = area.value, wrap = !insideMath(value.slice(0, from));
    let body = wrap ? '$' + snippet.replace(/ $/, '') + '$' : snippet;
    const caret = body.includes('▮') ? body.indexOf('▮') : body.length;
    body = body.replace('▮', '');
    area.value = value.slice(0, from) + body + value.slice(to);
    area.focus(); area.setSelectionRange(from + caret, from + caret);
    onChange(area.value);
  }
  function attach(area, popup, bar, onChange) {
    let open = false, items = [], index = 0, start = 0;
    const close = () => { open = false; popup.hidden = true; };
    function render() {
      popup.hidden = false;
      popup.replaceChildren(...items.map(([name, snippet, zh], i) => {
        const row = document.createElement('div'); row.className = 'ac-item' + (i === index ? ' on' : '');
        row.innerHTML = `<code>\\${escape(name)}</code><span class="ac-sym">${math(sample(snippet))}</span><span>${escape(zh)}</span>`;
        row.onmousedown = ev => { ev.preventDefault(); index = i; choose(); };
        return row;
      }));
      popup.children[index].scrollIntoView({ block: 'nearest' });
    }
    function suggest() {
      const m = /\\([A-Za-z^_|(\[\]%一-龥]*)$/.exec(area.value.slice(0, area.selectionStart));
      if (!m) return close();
      const q = m[1], rank = ([name, , zh]) => !q ? 0 : name.startsWith(q) ? 0 : name.toLowerCase().includes(q.toLowerCase()) ? 1 : zh.includes(q) ? 2 : -1;
      items = SYMBOLS.map(x => [rank(x), x]).filter(([k]) => k >= 0).sort((a, b) => a[0] - b[0]).map(([, x]) => x);
      if (!items.length) return close();
      open = true; index = 0; start = m.index; render();
    }
    function choose() { const [, snippet] = items[index]; close(); insert(area, snippet, start, area.selectionStart, onChange); }
    area.addEventListener('input', suggest);
    area.addEventListener('click', suggest);
    area.addEventListener('blur', () => setTimeout(close, 150));
    const keys = ev => {
      // Called first by the textarea's keydown: true when the list took the key (Enter then inserts, it does not confirm).
      if (!open) return false;
      if (ev.key === 'ArrowDown' || ev.key === 'ArrowUp') { ev.preventDefault(); index = (index + (ev.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length; render(); return true; }
      if ((ev.key === 'Enter' || ev.key === 'Tab') && !ev.isComposing) { ev.preventDefault(); choose(); return true; }
      if (ev.key === 'Escape') { ev.preventDefault(); close(); return true; }
      return false;
    };
    bar.replaceChildren(...BAR.map(name => {
      const [, snippet, zh] = SYMBOLS.find(x => x[0] === name), b = document.createElement('button');
      b.type = 'button'; b.title = `${zh}（\\${name}）`; b.dataset.snippet = snippet;
      b.onmousedown = ev => ev.preventDefault();
      b.onclick = () => insert(area, snippet, area.selectionStart, area.selectionEnd, onChange);
      return b;
    }));
    const paintBar = () => { for (const b of bar.children) b.innerHTML = math(sample(b.dataset.snippet)); };
    paintBar();
    return { keys, paintBar, close };
  }
  return { SYMBOLS, BAR, attach, insert, rendered, math, insideMath };
})();
if (typeof module !== 'undefined') module.exports = SymbolPicker;
