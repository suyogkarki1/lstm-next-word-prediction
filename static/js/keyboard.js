const textEl = document.getElementById('text');
const chatEl = document.getElementById('chat');
const keysEl = document.getElementById('keys');
const suggBtns = [...document.querySelectorAll('.sugg')];
const inputWrap = document.querySelector('.input-wrap');

let shift = false, numbers = false, current = [], reqId = 0, timer = null;

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
    case 'SHIFT': shift = !shift; renderKeys(); return;
    case 'BACK': backspace(); return;
    case '123': numbers = true; renderKeys(); return;
    case 'ABC': numbers = false; renderKeys(); return;
    case '#+=': return;
    case 'SPACE': insert(' '); return;
    case 'RETURN': send(); return;
    default:
      insert(shift ? k.toUpperCase() : k);
      if (shift) { shift = false; renderKeys(); }
  }
}

keysEl.addEventListener('pointerdown', e => {
  const b = e.target.closest('.key');
  if (!b) return;
  e.preventDefault();          // keep focus/caret in the textarea
  pressKey(b.dataset.k, b);
});

function autosize() {
  textEl.style.height = 'auto';
  textEl.style.height = Math.min(textEl.scrollHeight, 108) + 'px';
  inputWrap.classList.toggle('has-text', textEl.value.trim().length > 0);
  // keep the caret visible once the box hits its max height and starts scrolling
  if (textEl.selectionEnd === textEl.value.length) textEl.scrollTop = textEl.scrollHeight;
}

function onChange() {
  autosize();
  clearTimeout(timer);
  timer = setTimeout(fetchSuggestions, 70);
}
textEl.addEventListener('input', onChange);

async function fetchSuggestions() {
  const id = ++reqId;
  const text = textEl.value.slice(0, textEl.selectionStart);
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

function showSuggestions({ partial, suggestions }) {
  current = suggestions;
  const max = suggestions.length ? suggestions[0].score : 1;
  for (const btn of suggBtns) {
    const s = suggestions[+btn.dataset.i];
    const w = btn.querySelector('.w'), bar = btn.querySelector('.bar i');
    btn.classList.toggle('empty', !s);
    btn.classList.remove('fade'); void btn.offsetWidth; btn.classList.add('fade');
    if (!s) { w.innerHTML = ''; bar.style.width = '0'; btn.title = ''; continue; }
    const p = partial && s.word.startsWith(partial) ? partial.length : 0;
    w.innerHTML = `<b>${escapeHtml(s.word.slice(0, p))}</b>${escapeHtml(s.word.slice(p))}`;
    bar.style.width = Math.max(8, (s.score / max) * 100) + '%';
    btn.title = `${(s.score * 100).toFixed(1)}% confidence`;
  }
}

function accept(i) {
  const s = current[i];
  if (!s) return;
  const caret = textEl.selectionStart;
  const before = textEl.value.slice(0, caret), after = textEl.value.slice(caret);
  const trimmed = before.replace(/\S+$/, m => (/\s$/.test(before) ? m : ''));
  const sep = trimmed && !/\s$/.test(trimmed) ? ' ' : '';
  const newBefore = trimmed + sep + s.word + ' ';
  textEl.value = newBefore + after;
  textEl.selectionStart = textEl.selectionEnd = newBefore.length;
  textEl.focus();
  onChange();
}

suggBtns.forEach(b => b.addEventListener('pointerdown', e => { e.preventDefault(); accept(+b.dataset.i); }));

textEl.addEventListener('keydown', e => {
  if (e.key === 'Tab') { e.preventDefault(); accept(0); }
  else if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
textEl.addEventListener('click', onChange);

function bubble(text, who) {
  const d = document.createElement('div');
  d.className = 'bubble ' + who;
  d.textContent = text;
  chatEl.appendChild(d);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function send() {
  const v = textEl.value.trim();
  if (!v) return;
  bubble(v, 'me');
  textEl.value = '';
  onChange();
}
document.getElementById('send').addEventListener('click', send);

document.querySelectorAll('.tryme button[data-text]').forEach(b => b.addEventListener('click', () => {
  textEl.value = b.dataset.text;
  textEl.focus();
  textEl.selectionStart = textEl.selectionEnd = textEl.value.length;
  onChange();
}));

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
renderKeys();
