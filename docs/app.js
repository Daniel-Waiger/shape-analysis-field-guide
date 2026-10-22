/* Shape Analysis Field Guide — reader app
   Hash router (#06, #06/section-slug), markdown + math rendering, demo mounting.
   Chapters live in chapters/*.md and may be missing while the guide is written;
   every failure path renders a friendly placeholder instead of a broken page. */

const CHAPTERS = [
  { id: '00', file: '00-start-here.md', title: 'Start here', blurb: 'What "shape" means, why biologists care, the two datasets, and the map of the guide.' },
  { id: '01', file: '01-from-pixels-to-outlines.md', title: 'From pixels to outlines', blurb: 'Masks, contour tracing, skeleton ordering, arc-length resampling and three ways to represent a shape.' },
  { id: '02', file: '02-preshape.md', title: 'Preshape: removing position and size', blurb: 'Centering, the Frobenius norm, the preshape hypersphere, and what is thrown away.' },
  { id: '03', file: '03-rotation-and-complex-numbers.md', title: 'Rotation, solved in one line', blurb: 'Complex numbers as 2-D points, e^{iθ}, and the closed-form best rotation.' },
  { id: '04', file: '04-kendall-shape-space.md', title: 'Kendall shape space', blurb: 'Geodesic, Procrustes and chord distances; the geodesic path between two shapes.' },
  { id: '05', file: '05-mean-shapes.md', title: 'Mean shapes', blurb: 'The Fréchet mean as a top eigenvector, why plain averages shrink, and GPA as power iteration.' },
  { id: '06', file: '06-tangent-space-and-pca.md', title: 'Tangent space and PCA', blurb: 'Log and exp maps, PCA on the tangent plane, modes of variation, dead vs live worms.' },
  { id: '07', file: '07-splines.md', title: 'Shapes as smooth curves', blurb: 'The cubic B-spline bump, local control, interpolation, and the Gram matrix.' },
  { id: '08', file: '08-learned-embeddings.md', title: 'Learned embeddings', blurb: 'PointNet, supervised contrastive loss, temperature, and linear-probe evaluation.' },
  { id: '09', file: '09-field-notes.md', title: 'Field notes: bugs and how to avoid them', blurb: 'Lessons from the original tutorial: axis conventions, eval mode, unseeded splits, dead code.' },
  { id: 'glossary', file: 'glossary.md', title: 'Glossary', blurb: 'Every term, in one or two sentences.', aux: true },
  { id: 'cheatsheet', file: 'cheatsheet.md', title: 'Cheat sheet', blurb: 'All formulas on one page, with symbol legends and shapekit names.', aux: true },
  { id: 'references', file: 'references.md', title: 'References', blurb: 'Papers, datasets and upstream repositories.', aux: true },
];

const $ = (sel, root = document) => root.querySelector(sel);
const content = $('#content');
const pager = $('#pager');
const toc = $('#toc');
const chapterList = $('#chapter-list');

/* ---------- storage helpers (never throw) ---------- */
const store = {
  get(key, fallback) {
    try { const v = localStorage.getItem(key); return v === null ? fallback : JSON.parse(v); }
    catch (e) { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch (e) { /* ignore */ }
  },
};

/* ---------- theme ---------- */
function currentTheme() {
  const explicit = document.documentElement.getAttribute('data-theme');
  if (explicit === 'light' || explicit === 'dark') return explicit;
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}
function setTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  try { localStorage.setItem('sa-theme', theme); } catch (e) { /* ignore */ }
  document.dispatchEvent(new CustomEvent('themechange', { detail: { theme } }));
}
document.querySelectorAll('[data-theme-toggle]').forEach((btn) => {
  btn.addEventListener('click', () => setTheme(currentTheme() === 'dark' ? 'light' : 'dark'));
});

function themeColors() {
  const cs = getComputedStyle(document.documentElement);
  const read = (name) => cs.getPropertyValue(name).trim();
  return {
    paper: read('--paper'), surface: read('--surface'), ink: read('--ink'), muted: read('--muted'), line: read('--line'),
    teal: read('--teal'), coral: read('--coral'), gold: read('--gold'), lavender: read('--lavender'), slate: read('--slate'),
    tealText: read('--teal-text'), coralText: read('--coral-text'), theme: currentTheme(),
  };
}

/* ---------- sidebar ---------- */
function buildSidebar() {
  chapterList.innerHTML = '';
  CHAPTERS.forEach((ch, i) => {
    if (ch.aux && !CHAPTERS[i - 1].aux) {
      const d = document.createElement('li'); d.className = 'divider'; chapterList.appendChild(d);
    }
    const li = document.createElement('li');
    li.dataset.id = ch.id;
    const a = document.createElement('a');
    a.href = '#' + ch.id;
    a.innerHTML = `<span class="num">${ch.aux ? '—' : ch.id}</span><span>${ch.title}</span><span class="tick" aria-hidden="true">✓</span>`;
    li.appendChild(a);
    chapterList.appendChild(li);
  });
  updateProgress();
}
function visited() { return new Set(store.get('sa-visited', [])); }
function markVisited(id) {
  const v = visited(); if (v.has(id)) return; v.add(id); store.set('sa-visited', [...v]); updateProgress();
}
function updateProgress() {
  const v = visited();
  chapterList.querySelectorAll('li[data-id]').forEach((li) => li.classList.toggle('visited', v.has(li.dataset.id)));
  const n = CHAPTERS.filter((c) => v.has(c.id)).length;
  $('#progress-fill').style.width = `${(100 * n) / CHAPTERS.length}%`;
  $('#progress-text').textContent = `${n} of ${CHAPTERS.length} read`;
}
function setActive(id) {
  chapterList.querySelectorAll('a').forEach((a) => {
    if (a.getAttribute('href') === '#' + id) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
  });
}

/* ---------- mobile drawer ---------- */
const navToggle = $('#nav-toggle');
const scrim = $('#scrim');
function openNav(open) {
  document.body.classList.toggle('nav-open', open);
  navToggle.setAttribute('aria-expanded', String(open));
  scrim.hidden = !open;
}
navToggle.addEventListener('click', () => openNav(!document.body.classList.contains('nav-open')));
scrim.addEventListener('click', () => openNav(false));
chapterList.addEventListener('click', () => openNav(false));
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') openNav(false); });

/* ---------- keyboard prev/next ---------- */
document.addEventListener('keydown', (e) => {
  if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
  const t = e.target;
  if (t && (t.closest('input, textarea, select, [contenteditable="true"], .demo'))) return;
  if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
  const idx = CHAPTERS.findIndex((c) => c.id === state.chapter);
  const next = e.key === 'ArrowRight' ? idx + 1 : idx - 1;
  if (idx === -1) { if (e.key === 'ArrowRight') location.hash = CHAPTERS[0].id; return; }
  if (next < 0) location.hash = ''; else if (next < CHAPTERS.length) location.hash = CHAPTERS[next].id;
});

/* ---------- scroll progress ---------- */
const scrollFill = $('#scroll-progress span');
function onScroll() {
  const h = document.documentElement;
  const max = h.scrollHeight - h.clientHeight;
  scrollFill.style.width = max > 0 ? `${Math.min(100, (100 * h.scrollTop) / max)}%` : '0%';
}
window.addEventListener('scroll', onScroll, { passive: true });

/* ---------- markdown: math-aware marked setup ---------- */
function escapeHtml(s) { return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
let markedReady = false;
function setupMarked() {
  if (markedReady || typeof marked === 'undefined') return markedReady;
  const blockMath = {
    name: 'blockMath', level: 'block',
    start(src) { const m = src.match(/(^|\n)\$\$/); return m ? m.index + m[1].length : -1; },
    tokenizer(src) {
      const m = /^\$\$([\s\S]+?)\$\$(?:\n+|$)/.exec(src);
      if (m) return { type: 'blockMath', raw: m[0], text: m[1].trim() };
    },
    renderer(tok) { return `<div class="math math-display">$$${escapeHtml(tok.text)}$$</div>\n`; },
  };
  const inlineMath = {
    name: 'inlineMath', level: 'inline',
    start(src) { return src.indexOf('$'); },
    tokenizer(src) {
      // GFM-style rules: "$" not followed by space; closing "$" not preceded by space, not followed by a digit.
      const m = /^\$(?!\s)((?:\\\$|[^$\n])+?)(?<!\s)\$(?!\d)/.exec(src);
      if (m) return { type: 'inlineMath', raw: m[0], text: m[1] };
    },
    renderer(tok) { return `<span class="math math-inline">$${escapeHtml(tok.text)}$</span>`; },
  };
  marked.use({ gfm: true, breaks: false, extensions: [blockMath, inlineMath] });
  markedReady = true;
  return true;
}
function renderMarkdown(md) {
  if (!setupMarked()) {
    // marked failed to load (offline?): show the raw text so nothing is lost.
    return `<p class="loading">The markdown renderer could not be loaded; showing the raw chapter.</p><pre>${escapeHtml(md)}</pre>`;
  }
  return marked.parse(md);
}
function renderMath(root) {
  if (typeof renderMathInElement !== 'function') return;
  const opts = {
    delimiters: [{ left: '$$', right: '$$', display: true }, { left: '$', right: '$', display: false }],
    throwOnError: false, strict: 'ignore',
  };
  root.querySelectorAll('.math').forEach((el) => { try { renderMathInElement(el, opts); } catch (e) { /* leave source */ } });
  // raw HTML blocks (e.g. <details><summary>) bypass the markdown tokenizer, so render their maths directly
  root.querySelectorAll('details').forEach((el) => {
    try { renderMathInElement(el, { ...opts, ignoredClasses: ['math', 'katex'] }); } catch (e) { /* leave source */ }
  });
}

/* ---------- post-processing of rendered chapter HTML ---------- */
function slugify(text) {
  return text.toLowerCase().replace(/[^\w\s-]/g, '').trim().replace(/\s+/g, '-').slice(0, 60) || 'section';
}
function postProcess(root, chapterId) {
  // figures: ../figures/ -> figures/, wrap in <figure>, italic line after becomes the caption
  root.querySelectorAll('img').forEach((img) => {
    const src = img.getAttribute('src') || '';
    if (src.startsWith('../figures/')) img.setAttribute('src', src.replace('../figures/', 'figures/'));
    else if (src.startsWith('../')) img.setAttribute('src', src.replace('../', ''));
    img.loading = 'lazy';
    img.addEventListener('error', () => {
      img.classList.add('missing');
      const note = document.createElement('div'); note.className = 'fig-missing';
      note.textContent = `Figure not generated yet: ${img.getAttribute('src').split('/').pop()}`;
      img.insertAdjacentElement('afterend', note);
    });
    const p = img.parentElement;
    if (p && p.tagName === 'P') {
      const others = [...p.childNodes].filter((n) => n !== img && !(n.nodeType === 3 && !n.textContent.trim()) && n.nodeName !== 'BR');
      const onlyEm = others.length === 1 && others[0].nodeName === 'EM';
      if (others.length === 0 || onlyEm) {
        const fig = document.createElement('figure');
        p.replaceWith(fig);
        fig.appendChild(img);
        let cap = onlyEm ? others[0] : null;
        if (!cap) {
          const nxt = fig.nextElementSibling;
          if (nxt && nxt.tagName === 'P' && nxt.children.length === 1 && nxt.firstElementChild.tagName === 'EM' && !nxt.textContent.replace(nxt.firstElementChild.textContent, '').trim()) {
            cap = nxt.firstElementChild; nxt.remove();
          }
        }
        if (cap) { const fc = document.createElement('figcaption'); fc.innerHTML = cap.innerHTML; fig.appendChild(fc); }
      }
    }
  });
  // tables scroll horizontally instead of breaking the layout
  root.querySelectorAll('table').forEach((t) => {
    const wrap = document.createElement('div'); wrap.className = 'table-wrap'; t.replaceWith(wrap); wrap.appendChild(t);
  });
  // <details> become "reveal answer" cards
  root.querySelectorAll('details').forEach((d) => {
    let summary = d.querySelector(':scope > summary');
    if (!summary) { summary = document.createElement('summary'); summary.textContent = 'Reveal answer'; d.prepend(summary); }
    // if the body is bare text (no block children), parse it as markdown
    const bodyNodes = [...d.childNodes].filter((n) => n !== summary);
    const hasBlocks = bodyNodes.some((n) => n.nodeType === 1 && /^(P|UL|OL|PRE|DIV|BLOCKQUOTE|TABLE|H[1-6])$/.test(n.nodeName));
    if (!hasBlocks && markedReady) {
      const raw = bodyNodes.map((n) => n.nodeType === 3 ? n.textContent : n.outerHTML).join('').trim();
      bodyNodes.forEach((n) => n.remove());
      const body = document.createElement('div'); body.className = 'details-body'; body.innerHTML = marked.parse(raw); d.appendChild(body);
    }
  });
  // first blockquote right after the h1 is the "In one breath" summary
  const h1 = root.querySelector('h1');
  if (h1) {
    let n = h1.nextElementSibling;
    while (n && n.tagName === 'DIV' && n.classList.contains('toc-inline')) n = n.nextElementSibling;
    if (n && n.tagName === 'BLOCKQUOTE') {
      n.classList.add('lede');
      if (!/^\s*in one breath/i.test(n.textContent)) n.classList.add('lede-label');
    }
  }
  // headings get stable ids; links to other chapters route through the hash
  const seen = new Set();
  root.querySelectorAll('h1, h2, h3, h4').forEach((h) => {
    let id = slugify(h.textContent); let k = 2;
    while (seen.has(id)) id = `${slugify(h.textContent)}-${k++}`;
    seen.add(id); h.id = id;
  });
  root.querySelectorAll('a[href]').forEach((a) => {
    const href = a.getAttribute('href');
    const m = /^(?:\.\/)?([0-9a-z-]+)\.md(?:#(.*))?$/i.exec(href);
    if (m) {
      const target = CHAPTERS.find((c) => c.file.toLowerCase() === `${m[1]}.md`.toLowerCase());
      if (target) a.setAttribute('href', `#${target.id}${m[2] ? '/' + m[2] : ''}`);
    } else if (/^#/.test(href) && chapterId) {
      a.setAttribute('href', `#${chapterId}/${href.slice(1)}`);
    } else if (/^https?:\/\//.test(href)) {
      a.target = '_blank'; a.rel = 'noopener';
    }
  });
  // mark [Inference] / [Unverified] labels
  root.querySelectorAll('strong').forEach((s) => {
    const t = s.textContent.trim();
    if (t === '[Inference]') s.classList.add('inference'); else if (t === '[Unverified]') s.classList.add('unverified');
  });
  // copy buttons on code blocks
  root.querySelectorAll('pre').forEach((pre) => {
    const btn = document.createElement('button'); btn.className = 'copy-btn'; btn.type = 'button'; btn.textContent = 'Copy';
    btn.addEventListener('click', async () => {
      try { await navigator.clipboard.writeText(pre.querySelector('code')?.textContent ?? pre.textContent); btn.textContent = 'Copied'; }
      catch (e) { btn.textContent = 'Select & copy'; }
      setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
    });
    pre.appendChild(btn);
  });
}

/* ---------- mini-TOC ---------- */
let tocObserver = null;
function buildToc(root, chapterId) {
  if (tocObserver) { tocObserver.disconnect(); tocObserver = null; }
  const h2s = [...root.querySelectorAll('h2')];
  toc.innerHTML = ''; toc.hidden = true; toc.removeAttribute('data-has-items');
  root.querySelectorAll('.toc-inline').forEach((n) => n.remove());
  if (h2s.length < 2) return;
  const items = h2s.map((h) => `<li><a href="#${chapterId}/${h.id}" data-target="${h.id}">${escapeHtml(h.textContent)}</a></li>`).join('');
  toc.innerHTML = `<h2>On this page</h2><ol>${items}</ol>`;
  toc.hidden = false; toc.setAttribute('data-has-items', '');
  // inline version (narrow screens) placed after the lede
  const inline = document.createElement('details'); inline.className = 'toc-inline';
  inline.innerHTML = `<summary>On this page</summary><ol>${items}</ol>`;
  const h1 = root.querySelector('h1');
  const lede = h1 && h1.nextElementSibling && h1.nextElementSibling.tagName === 'BLOCKQUOTE' ? h1.nextElementSibling : h1;
  if (lede) lede.insertAdjacentElement('afterend', inline); else root.prepend(inline);
  if ('IntersectionObserver' in window) {
    const links = new Map([...toc.querySelectorAll('a')].map((a) => [a.dataset.target, a]));
    tocObserver = new IntersectionObserver((entries) => {
      entries.forEach((en) => { if (en.isIntersecting) { links.forEach((a) => a.classList.remove('active')); links.get(en.target.id)?.classList.add('active'); } });
    }, { rootMargin: '-10% 0px -75% 0px' });
    h2s.forEach((h) => tocObserver.observe(h));
  }
}

/* ---------- pager ---------- */
function buildPager(idx) {
  const prev = idx > 0 ? CHAPTERS[idx - 1] : null;
  const next = idx < CHAPTERS.length - 1 ? CHAPTERS[idx + 1] : null;
  const prevHtml = prev
    ? `<a class="prev" href="#${prev.id}"><small>← Previous</small><span>${prev.title}</span></a>`
    : `<a class="prev" href="#"><small>← Back to</small><span>Overview</span></a>`;
  const nextHtml = next ? `<a class="next" href="#${next.id}"><small>Next →</small><span>${next.title}</span></a>` : '';
  pager.innerHTML = prevHtml + nextHtml;
}

/* ---------- demo mounting ---------- */
async function loadJSON(path) {
  const url = new URL(path, document.baseURI);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.json();
}
const demoCleanups = [];
async function mountDemos(root) {
  while (demoCleanups.length) { try { demoCleanups.pop()(); } catch (e) { /* ignore */ } }
  const nodes = [...root.querySelectorAll('div.demo[data-demo]')];
  await Promise.all(nodes.map(async (el) => {
    const name = (el.dataset.demo || '').trim();
    if (!/^[a-z0-9-]+$/.test(name)) { demoError(el, `Unknown demo "${name}".`); return; }
    try {
      const mod = await import(`./demos/${name}.js`);
      const mount = mod.default || mod.mount;
      if (typeof mount !== 'function') throw new Error('module has no default export mount(el, ctx)');
      const ctx = {
        colors: themeColors(), loadJSON,
        reducedMotion: window.matchMedia('(prefers-reduced-motion: reduce)').matches,
        onThemeChange(cb) {
          const h = () => cb(themeColors()); document.addEventListener('themechange', h);
          demoCleanups.push(() => document.removeEventListener('themechange', h));
        },
      };
      const cleanup = await mount(el, ctx);
      if (typeof cleanup === 'function') demoCleanups.push(cleanup);
    } catch (err) {
      console.warn(`demo "${name}" failed:`, err);
      demoError(el, `The interactive demo “${name}” could not be started here. The static figure above tells the same story.`);
    }
  }));
}
function demoError(el, msg) { el.innerHTML = `<div class="demo-error">${escapeHtml(msg)}</div>`; }

/* ---------- views ---------- */
const state = { chapter: null, loadedId: null };

function fallbackWormSvg() {
  // An inline worm drawn by hand, used when figures/hero.gif is not generated yet.
  const pts = [];
  for (let i = 0; i < 12; i++) { const x = 90 + i * 56; const y = 150 + 46 * Math.sin((i / 11) * Math.PI * 1.6 - 0.4); pts.push([x, y]); }
  const path = pts.map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join(' ');
  const dots = pts.map((p) => `<circle cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="5" fill="var(--ink)"/>`).join('');
  const ghost = pts.map((p, i) => `L${(p[0]).toFixed(1)} ${(300 - p[1] * 0.55 - 40 + i * 2).toFixed(1)}`).join(' ').replace(/^L/, 'M');
  return `<svg viewBox="0 0 800 300" role="img" aria-label="A worm outline with twelve landmarks">
    <path d="${ghost}" fill="none" stroke="var(--coral)" stroke-width="14" stroke-linecap="round" stroke-linejoin="round" opacity=".35"/>
    <path d="${path}" fill="none" stroke="var(--teal)" stroke-width="14" stroke-linecap="round" stroke-linejoin="round" opacity=".9"/>
    ${dots}
  </svg>`;
}

function renderLanding() {
  state.chapter = null; state.loadedId = 'landing';
  setActive(null);
  toc.hidden = true; toc.removeAttribute('data-has-items');
  const v = visited();
  const cards = CHAPTERS.map((c) => `
    <a class="card ${v.has(c.id) ? 'visited' : ''}" href="#${c.id}">
      <span class="num">${c.aux ? 'Appendix' : 'Chapter ' + c.id}</span>
      <span class="title">${c.title}</span>
      <span class="blurb">${c.blurb}</span>
    </a>`).join('');
  content.innerHTML = `
    <section class="hero">
      <figure class="hero-figure" id="hero-figure">
        <img src="figures/hero.gif" alt="A live worm morphing along a geodesic into the mean shape and on to a dead worm" id="hero-img">
      </figure>
      <p class="eyebrow">A field guide for biologists</p>
      <h1>Shape analysis, from the outline up</h1>
      <p class="promise">This guide takes you from a stack of microscope images to a working understanding of shape analysis:
      how to strip position, size and rotation from an outline, how to measure the distance between two shapes,
      average them, find their main modes of variation, describe them as smooth curves, and learn embeddings
      from point clouds. Every idea comes with a formula read aloud in plain words, a runnable
      <code>shapekit</code> snippet, and, where it helps, something you can drag.</p>
    </section>
    <h2 class="section-label">Chapters</h2>
    <div class="card-grid">${cards}</div>
    <p class="loading">Reading time is roughly 15 minutes per chapter. Use the arrow keys to move between chapters.</p>`;
  const img = $('#hero-img');
  img.addEventListener('error', () => { $('#hero-figure').innerHTML = fallbackWormSvg(); });
  pager.innerHTML = '';
  document.title = 'Shape Analysis Field Guide';
  window.scrollTo({ top: 0, behavior: 'auto' });
}

function placeholder(ch, reason) {
  return `
    <h1>${ch.aux ? '' : ch.id + ' · '}${ch.title}</h1>
    <div class="placeholder">
      <h2>This chapter is still being written.</h2>
      <p>When it lands it will cover: ${ch.blurb}</p>
      <p>The site looked for <code>chapters/${ch.file}</code>${reason ? ` (${escapeHtml(reason)})` : ''}. Refresh once the file exists.</p>
    </div>`;
}

async function renderChapter(id, anchor) {
  const idx = CHAPTERS.findIndex((c) => c.id === id);
  if (idx === -1) { renderLanding(); return; }
  const ch = CHAPTERS[idx];
  const sameChapter = state.loadedId === id;
  state.chapter = id;
  setActive(id);
  if (!sameChapter) {
    content.innerHTML = '<p class="loading">Loading chapter…</p>';
    let html;
    try {
      const res = await fetch(`chapters/${ch.file}`, { cache: 'no-cache' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const md = await res.text();
      if (!md.trim()) throw new Error('empty file');
      html = renderMarkdown(md);
    } catch (err) {
      html = placeholder(ch, err.message);
    }
    if (state.chapter !== id) return; // navigated away while loading
    content.innerHTML = html;
    try { postProcess(content, id); } catch (e) { console.warn('post-process', e); }
    renderMath(content);
    buildToc(content, id);
    buildPager(idx);
    state.loadedId = id;
    document.title = `${ch.title} · Shape Analysis Field Guide`;
    markVisited(id);
    mountDemos(content);
  }
  if (anchor) {
    const target = document.getElementById(anchor);
    if (target) { target.scrollIntoView({ block: 'start' }); return; }
  }
  if (!sameChapter) {
    window.scrollTo({ top: 0, behavior: 'auto' });
    try { $('#main').focus({ preventScroll: true }); } catch (e) { /* ignore */ }
  }
}

function route() {
  const raw = decodeURIComponent(location.hash.replace(/^#/, ''));
  if (!raw) { renderLanding(); return; }
  const [id, ...rest] = raw.split('/');
  const anchor = rest.join('/');
  openNav(false);
  renderChapter(id, anchor);
}

/* ---------- boot ---------- */
buildSidebar();
function boot() {
  // wait briefly for the deferred libraries; render regardless after that
  const ready = () => typeof marked !== 'undefined' && typeof renderMathInElement !== 'undefined';
  let tries = 0;
  const tick = () => { if (ready() || tries++ > 40) route(); else setTimeout(tick, 50); };
  tick();
}
window.addEventListener('hashchange', route);
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
onScroll();
