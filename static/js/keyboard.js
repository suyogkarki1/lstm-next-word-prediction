const textEl = document.getElementById('text');
const ghostEl = document.getElementById('ghost');
const chatEl = document.getElementById('chat');
const keysEl = document.getElementById('keys');
const suggBtns = [...document.querySelectorAll('.sugg')];
const inputWrap = document.querySelector('.input-wrap');

let shift = false, manualShift = false, numbers = false;
let current = { partial: '', suggestions: [] }, reqId = 0, timer = null;

const LAYOUTS = {
  letters: [
    ['q','w','e','r','t','y','u','i','o','p'],
    ['a','s','d','f','g','h','j','k','l'],
    ['SHIFT','z','x','c','v','b','n','m','BACK'],
    ['123','SPACE','RETURN'],
  ],
  numbers: [
    ['1','2','3','4','5','6','7','8','9','0'],
    ['-','/',':',';','(',')','$','&','@','"'],
    ['#+=','.',',','?','!',"'",'%','BACK'],
    ['ABC','SPACE','RETURN'],
  ],
};

const ICONS = {
  shift: '<svg width="20" height="19" viewBox="0 0 20 19"><path d="M10 1.5 1.5 10H6v7h8v-7h4.5z" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/></svg>',
  shiftOn: '<svg width="20" height="19" viewBox="0 0 20 19"><path d="M10 1.5 1.5 10H6v7h8v-7h4.5z" fill="currentColor"/></svg>',
  back: '<svg width="25" height="18" viewBox="0 0 25 18"><path d="M8 1.5h14a1.5 1.5 0 0 1 1.5 1.5v12a1.5 1.5 0 0 1-1.5 1.5H8L1.5 9z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/><path d="m11.5 5.5 7 7m0-7-7 7" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>',
};

/* ---------------- text helpers ---------------- */

const beforeCaret = () => textEl.value.slice(0, textEl.selectionStart);
const caretAtEnd = () => textEl.selectionStart === textEl.value.length;
const atSentenceStart = s => !s.trim() || /[.!?]\s+$/.test(s);

// the model works in lowercase; show words the way a phone keyboard would
function display(word, contextBefore) {
  if (word === 'i' || /^i'/.test(word)) return 'I' + word.slice(1);
  return atSentenceStart(contextBefore) ? word.charAt(0).toUpperCase() + word.slice(1) : word;
}

// text before the word being typed, and the partial word itself
function splitPartial(text) {
  const m = /\S+$/.exec(text);
  return m && !/\s$/.test(text) ? [text.slice(0, m.index), m[0]] : [text, ''];
}

/* ---------------- keyboard ---------------- */

function renderKeys() {
  keysEl.innerHTML = '';
  LAYOUTS[numbers ? 'numbers' : 'letters'].forEach((row, ri) => {
    const r = document.createElement('div');
    r.className = 'row' + (!numbers && ri === 1 ? ' mid' : '');
    for (const k of row) {
      const b = document.createElement('button');
      b.className = 'key';
      b.dataset.k = k;
      if (k === 'SHIFT') { b.classList.add('mod', 'gap-r'); if (shift) b.classList.add('on'); b.innerHTML = shift ? ICONS.shiftOn : ICONS.shift; }
      else if (k === '#+=') { b.classList.add('mod', 'gap-r'); b.textContent = k; b.style.fontSize = '15px'; }
      else if (k === 'BACK') { b.classList.add('mod', 'gap-l'); b.innerHTML = ICONS.back; }
      else if (k === '123' || k === 'ABC') { b.classList.add('mod', 'wide'); b.textContent = k; }
      else if (k === 'SPACE') { b.classList.add('space'); b.textContent = 'space'; }
      else if (k === 'RETURN') { b.classList.add('ret'); b.textContent = 'send'; }
      else {
        const label = shift ? k.toUpperCase() : k;
        b.textContent = label;
        const pop = document.createElement('span');
        pop.className = 'pop'; pop.textContent = label;
        b.appendChild(pop);
      }
      r.appendChild(b);
    }
    keysEl.appendChild(r);
  });
}

function setShift(on) {
  if (shift === on) return;
  shift = on;
  if (!numbers) renderKeys();
}

// iOS-style auto-capitalisation at the start of a sentence
function autoShift() {
  if (!manualShift) setShift(atSentenceStart(beforeCaret()));
}

function insert(str) {
  const s = textEl.selectionStart, e = textEl.selectionEnd, v = textEl.value;
  textEl.value = v.slice(0, s) + str + v.slice(e);
  textEl.selectionStart = textEl.selectionEnd = s + str.length;
  onChange();
}

function backspace() {
  const s = textEl.selectionStart, e = textEl.selectionEnd, v = textEl.value;
  if (s !== e) { textEl.value = v.slice(0, s) + v.slice(e); textEl.selectionStart = textEl.selectionEnd = s; }
  else if (s > 0) { textEl.value = v.slice(0, s - 1) + v.slice(s); textEl.selectionStart = textEl.selectionEnd = s - 1; }
  onChange();
}

function pressKey(k, btn) {
  if (btn && btn.querySelector('.pop')) {
    btn.classList.add('pressed');
    setTimeout(() => btn.classList.remove('pressed'), 110);
  }
  switch (k) {
    case 'SHIFT': manualShift = !shift; shift = !shift; renderKeys(); return;
    case 'BACK': backspace(); return;
    case '123': numbers = true; renderKeys(); return;
    case 'ABC': numbers = false; renderKeys(); return;
    case '#+=': return;
    case 'SPACE': insert(' '); return;
    case 'RETURN': send(); return;
    default:
      manualShift = false;
      insert(shift ? k.toUpperCase() : k);
  }
}

keysEl.addEventListener('pointerdown', e => {
  const b = e.target.closest('.key');
  if (!b) return;
  e.preventDefault();          // keep focus/caret in the textarea
  pressKey(b.dataset.k, b);
});

/* ---------------- input box ---------------- */

function autosize() {
  textEl.style.height = 'auto';
  textEl.style.height = Math.min(textEl.scrollHeight, 108) + 'px';
  inputWrap.classList.toggle('has-text', textEl.value.trim().length > 0);
  // keep the caret visible once the box hits its max height and starts scrolling
  if (caretAtEnd()) textEl.scrollTop = textEl.scrollHeight;
}

function onChange() {
  autosize();
  autoShift();
  renderGhost(null);           // hide stale ghost text until fresh suggestions arrive
  clearTimeout(timer);
  timer = setTimeout(fetchSuggestions, 60);
}
textEl.addEventListener('input', () => { manualShift = false; onChange(); });
textEl.addEventListener('scroll', () => { ghostEl.scrollTop = textEl.scrollTop; });

// grey inline completion after the caret, like iOS predictive text
function renderGhost(top) {
  const text = textEl.value;
  let rest = '';
  if (top && caretAtEnd() && text.trim()) {
    const [ctx, partial] = splitPartial(text);
    const word = display(top.word, ctx);
    if (partial) rest = word.toLowerCase().startsWith(partial.toLowerCase()) ? word.slice(partial.length) : '';
    else rest = word;
  }
  ghostEl.innerHTML = rest ? `<span class="typed">${escapeHtml(text)}</span><span class="rest">${escapeHtml(rest)}</span>` : '';
  ghostEl.scrollTop = textEl.scrollTop;
}

/* ---------------- suggestions ---------------- */

async function fetchSuggestions() {
  const id = ++reqId;
  const text = beforeCaret();
  if (!text.trim()) { showSuggestions({ partial: '', suggestions: [] }); return; }
  try {
    const res = await fetch('/predict', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    if (id === reqId) showSuggestions(data);
  } catch (err) { console.error(err); }
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function showSuggestions(data) {
  current = data;
  const { suggestions } = data;
  const [ctx, partial] = splitPartial(beforeCaret());
  const max = suggestions.length ? suggestions[0].score : 1;
  for (const btn of suggBtns) {
    const s = suggestions[+btn.dataset.i];
    const w = btn.querySelector('.w'), bar = btn.querySelector('.bar i');
    btn.classList.toggle('empty', !s);
    btn.classList.remove('fade'); void btn.offsetWidth; btn.classList.add('fade');
    if (!s) { w.innerHTML = ''; bar.style.width = '0'; btn.title = ''; continue; }
    const word = display(s.word, ctx);
    const p = partial && word.toLowerCase().startsWith(partial.toLowerCase()) ? partial.length : 0;
    w.innerHTML = `<b>${escapeHtml(word.slice(0, p))}</b>${escapeHtml(word.slice(p))}`;
    bar.style.width = Math.max(8, (s.score / max) * 100) + '%';
    btn.title = `${(s.score * 100).toFixed(1)}% confidence`;
  }
  renderGhost(suggestions[0]);
}

function accept(i) {
  const s = current.suggestions[i];
  if (!s) return;
  const after = textEl.value.slice(textEl.selectionStart);
  const [ctx] = splitPartial(beforeCaret());
  const sep = ctx && !/\s$/.test(ctx) ? ' ' : '';
  const newBefore = ctx + sep + display(s.word, ctx) + ' ';
  textEl.value = newBefore + after;
  textEl.selectionStart = textEl.selectionEnd = newBefore.length;
  textEl.focus();
  manualShift = false;
  onChange();
}

suggBtns.forEach(b => b.addEventListener('pointerdown', e => { e.preventDefault(); accept(+b.dataset.i); }));

textEl.addEventListener('keydown', e => {
  if (e.key === 'Tab' || (e.key === 'ArrowRight' && caretAtEnd() && ghostEl.textContent)) {
    e.preventDefault(); accept(0);
  } else if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
textEl.addEventListener('click', onChange);

/* ---------------- chat ---------------- */

function bubble(who, text, caption) {
  const d = document.createElement('div');
  d.className = 'bubble ' + who;
  if (caption) {
    const c = document.createElement('span');
    c.className = 'caption'; c.textContent = caption;
    d.appendChild(c);
  }
  d.appendChild(document.createTextNode(text));
  chatEl.appendChild(d);
  chatEl.scrollTop = chatEl.scrollHeight;
  return d;
}

function typing() {
  const d = document.createElement('div');
  d.className = 'bubble bot typing';
  d.innerHTML = '<i></i><i></i><i></i>';
  chatEl.appendChild(d);
  chatEl.scrollTop = chatEl.scrollHeight;
  return d;
}

// the bot replies by letting the LSTM finish your sentence
async function reply(text) {
  const dots = typing();
  try {
    const [res] = await Promise.all([
      fetch('/continue', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }) }),
      new Promise(r => setTimeout(r, 650)),
    ]);
    const { words, finished } = await res.json();
    dots.remove();
    if (!words.length) return;
    const ended = atSentenceStart(text + ' ');
    const out = words.map((w, i) => display(w, i === 0 && ended ? '' : 'x ')).join(' ');
    bubble('bot', (ended ? '' : '…') + out + (finished ? '.' : '…'),
           ended ? 'LSTM writes the next sentence' : 'LSTM would finish it');
  } catch (err) { dots.remove(); console.error(err); }
}

function send() {
  const v = textEl.value.trim();
  if (!v) return;
  bubble('me', v);
  textEl.value = '';
  onChange();
  reply(v);
}
document.getElementById('send').addEventListener('click', send);

document.querySelectorAll('.tryme button[data-text]').forEach(b => b.addEventListener('click', () => {
  textEl.value = b.dataset.text;
  textEl.focus();
  textEl.selectionStart = textEl.selectionEnd = textEl.value.length;
  onChange();
}));

/* ---------------- chrome ---------------- */

function tick() {
  const d = new Date();
  document.getElementById('clock').textContent = d.getHours() + ':' + String(d.getMinutes()).padStart(2, '0');
}
tick(); setInterval(tick, 30000);
document.getElementById('today').textContent =
  new Date().toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });

// scale the iPhone so the whole device always fits inside the window
const stage = document.getElementById('stage');
function fit() {
  const iphone = document.getElementById('iphone');
  const w = iphone.offsetWidth, h = iphone.offsetHeight;
  const sideBySide = window.innerWidth > 760;
  const info = document.querySelector('.info');
  const gap = parseFloat(getComputedStyle(document.body).columnGap) || 0;
  const availW = window.innerWidth - 32 - (sideBySide ? info.offsetWidth + gap : 0);
  const s = Math.min(1, (window.innerHeight - 32) / h, availW / w);
  stage.style.setProperty('--s', sideBySide ? s.toFixed(4) : 1);
}
window.addEventListener('resize', fit);
fit();
autoShift();
renderKeys();
