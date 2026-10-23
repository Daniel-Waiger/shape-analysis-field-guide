/* Phenotype Lab — chapter 10 of the Shape Analysis Field Guide.
   Hash routes: #/ (home), #/<id> (dataset: the four panels), #/<id>/classify (drag a shape).
   Every result comes from docs/lab/data/*.json written by scripts/run_lab.py; no result is typed in here.
   Vanilla JS modules + SVG; engines in ./engines/. */

import { svg, h, pathOf } from '../demos/_lib.js';
import * as classical from './engines/classical.js';
import { createOnnxEngine, onnxUnavailableReason, ORT_VERSION } from './engines/onnx.js';
import { mountPyodidePanel, pyodideOffered, PYODIDE_VERSION, PYODIDE_SIZE_ESTIMATE } from './engines/pyodide.js';

/* ---------- registry metadata (names only; no numbers) ---------- */
const REGISTRY = [
  { id: 'celegans', title: 'C. elegans, live vs dead', representation: 'landmarks', short: 'Worms', fetch: null, images: 'manifest', objWord: 'worm' },
  // titleNote: shown after the title on the dataset page (LAB_SPEC.md Round 3b: the MEF confound)
  { id: 'mef', title: 'Mouse fibroblasts, wild-type vs LMNA−/−', representation: 'pointcloud', short: 'Fibroblasts', fetch: null, images: 'manifest', objWord: 'cell', titleNote: 'genotype confounded with substrate' },
  { id: 'brain', title: 'Brain landmarks, schizophrenia vs control', representation: 'landmarks', short: 'Brain', fetch: 'python scripts/fetch_datasets.py --brain', images: 'none' },
  { id: 'gorilla', title: 'Gorilla skulls, female vs male', representation: 'landmarks', short: 'Gorillas', fetch: 'python scripts/fetch_datasets.py --gorilla', images: 'none' },
  { id: 'leaves', title: 'Leaf outlines, species', representation: 'outline', short: 'Leaves', fetch: 'python scripts/fetch_datasets.py --leaves', images: 'unverified' },
  { id: 'rbc', title: 'Red blood cells, sickle-cell disease', representation: 'outline', short: 'Blood cells', fetch: 'python scripts/fetch_datasets.py --rbc', images: 'nc_nd' },
];
/** The three shape methods the guide teaches, plus the size-only baseline (a baseline, never a headline). */
const METHODS = [
  { key: 'landmarks_lr', name: 'Landmarks + logistic', long: 'Landmarks → Procrustes → tangent PCA → logistic regression', chapters: '02–06', engine: 'classical' },
  { key: 'spline_lr', name: 'Spline + logistic', long: 'Spline control points → the same Procrustes and tangent-PCA geometry on them (no Gram matrix Φ) → logistic regression', chapters: '07', engine: 'classical' },
  { key: 'pointnet_probe', name: 'PointNet + probe', long: 'Point cloud → PointNet embedding → linear probe', chapters: '08', engine: 'onnx' },
  { key: 'size_lr', name: 'Size only', long: 'log centroid size → logistic regression: a baseline with no shape in it', chapters: '02', engine: null, baseline: true },
];
const SHAPE_METHODS = METHODS.filter((m) => !m.baseline);
const REP_LABEL = { landmarks: 'landmarks', outline: 'closed outlines', pointcloud: 'point cloud' };
const REPO_URL = 'https://github.com/Daniel-Waiger/shape-analysis-field-guide';
const ATTRIBUTION_URL = `${REPO_URL}/blob/main/ATTRIBUTION.md`;
/** License name → canonical text (a lookup of URLs, not of results; the JSON's own `license_url` wins). */
const LICENSE_URLS = [
  [/CC BY-NC-ND 4/i, 'https://creativecommons.org/licenses/by-nc-nd/4.0/'],
  [/CC BY 4/i, 'https://creativecommons.org/licenses/by/4.0/'],
  [/CC0/i, 'https://creativecommons.org/publicdomain/zero/1.0/'],
  [/GPL-2/i, 'https://www.gnu.org/licenses/old-licenses/gpl-2.0.html'],
  [/BSD-3/i, 'https://opensource.org/license/bsd-3-clause'],
  [/MIT/i, 'https://opensource.org/license/mit'],
];

/* ---------- class colors ---------- */
const PALETTE = ['coral', 'teal', 'gold', 'lavender', 'slate'];
/** Class colors: class order → coral, teal, gold, lavender, slate; further classes get evenly spaced hues. */
const extraHue = (i) => (200 + 137.5 * (i - PALETTE.length)) % 360;
const stroke = (i) => (i < PALETTE.length ? `var(--${PALETTE[i]})` : `hsl(${extraHue(i).toFixed(0)} 55% 48%)`);
/** Text color of class i: a token for the five named colors, a per-theme variable (--cxK) beyond that. */
const textColor = (i) => (i < PALETTE.length ? `var(--${PALETTE[i]}-text)` : `var(--cx${i})`);
const textClass = (i) => (i < PALETTE.length ? `c${i}` : '');
/** class="cK" for the five named colors, an inline per-theme variable beyond that (used inside HTML strings). */
const textAttr = (i) => (i < PALETTE.length ? `class="c${i}"` : `style="color:var(--cx${i})"`);
/* The extra hues need a lightness per theme that keeps text at ≥ 4.5:1 on the lab's backgrounds; the
   variables are written once into a <style> element so a theme switch needs no re-render. */
function hslToRgb(hh, ss, ll) {
  const s = ss / 100, l = ll / 100, c = (1 - Math.abs(2 * l - 1)) * s, x = c * (1 - Math.abs(((hh / 60) % 2) - 1)), m = l - c / 2;
  const [r, g, b] = hh < 60 ? [c, x, 0] : hh < 120 ? [x, c, 0] : hh < 180 ? [0, c, x] : hh < 240 ? [0, x, c] : hh < 300 ? [x, 0, c] : [c, 0, x];
  return [(r + m) * 255, (g + m) * 255, (b + m) * 255];
}
function luminance([r, g, b]) {
  const f = (v) => { const c = v / 255; return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
  return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
}
const contrast = (a, b) => (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
/** The lightness (%) at which hsl(h 60% l) reaches `target` against a background of luminance `bg`. */
function lightnessFor(hue, bg, dark, target = 4.6) {
  const steps = dark ? [50, 55, 60, 65, 70, 75, 80, 85, 90] : [45, 40, 36, 33, 30, 27, 24, 21, 18, 15];
  for (const l of steps) if (contrast(luminance(hslToRgb(hue, 60, l)), bg) >= target) return l;
  return dark ? 92 : 14;
}
const LIGHT_BG = luminance([253, 239, 221]);   // the lightest-text worst case: gold-tinted boxes on paper
const DARK_BG = luminance([66, 58, 48]);       // the same tint over the dark surface
let extraDefined = 0;
function ensureExtraColors(nClasses) {
  if (nClasses <= extraDefined || nClasses <= PALETTE.length) return;
  let light = '', dark = '';
  for (let i = PALETTE.length; i < nClasses; i++) {
    const hue = extraHue(i);
    light += `--cx${i}: hsl(${hue.toFixed(0)} 60% ${lightnessFor(hue, LIGHT_BG, false)}%);`;
    dark += `--cx${i}: hsl(${hue.toFixed(0)} 60% ${lightnessFor(hue, DARK_BG, true)}%);`;
  }
  let el = document.getElementById('lab-class-colors');
  if (!el) { el = document.createElement('style'); el.id = 'lab-class-colors'; document.head.appendChild(el); }
  el.textContent = `:root{${light}}@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){${dark}}}:root[data-theme="dark"]{${dark}}`;
  extraDefined = nClasses;
}

/* ---------- small helpers ---------- */
const $ = (sel, root = document) => root.querySelector(sel);
const main = $('#lab-main');
const fmt = (x, d = 3) => { if (!Number.isFinite(x)) return '[not run]'; const s = x.toFixed(d); return /^-0\.?0*$/.test(s) ? s.slice(1) : s; };
const pct = (x, d = 1) => (Number.isFinite(x) ? `${(100 * x).toFixed(d)} %` : '[not run]');
const thousands = (n) => (Number.isFinite(n) ? n.toLocaleString('en-US') : '[not run]');
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
/** Escape, then render `code` spans the way the JSON strings write them. */
const mdCode = (s) => esc(s).replace(/`([^`]+)`/g, '<code>$1</code>');
/** Escape a title and keep tokens with a slash (LMNA−/−) on one line. */
const titleHtml = (t) => esc(t).replace(/(\S*\/\S*)/g, '<span class="nowrap">$1</span>');
const COUNT_WORDS = ['no', 'one', 'two', 'three', 'four', 'five', 'six'];
const countWord = (n) => COUNT_WORDS[n] ?? String(n);
function dateOf(iso) { try { const d = new Date(iso); return Number.isNaN(d.getTime()) ? iso : d.toLocaleDateString('en-US', { day: 'numeric', month: 'short', year: 'numeric' }); } catch (e) { return iso; } }
/** A number with its provenance on hover/tap ("computed by … at …"). */
function num(text, ds, cls = '', extra = '') { return `<span class="num ${cls}" title="computed by ${esc(ds.script || 'scripts/run_lab.py')} at ${esc(ds.computed_at || 'unknown time')}${extra ? ' · ' + esc(extra) : ''}">${text}</span>`; }
/** Species names come title-cased from the source CSV; a binomial takes a lowercase epithet. */
function speciesCase(name) {
  const m = /^([A-Z][a-z]+) ([A-Z][a-z]+\.?)$/.exec(String(name));
  return m ? `${m[1]} ${m[2].toLowerCase()}` : name;
}
/** "U. glabra" for a binomial, else the name itself (column heads of a wide confusion matrix). */
function shortClass(name) { const m = /^([A-Z])[a-z]+ ([a-z]+\.?)$/.exec(name); return m ? `${m[1]}. ${m[2]}` : name; }
const store = {
  get(k, fb) { try { const v = localStorage.getItem(k); return v === null ? fb : JSON.parse(v); } catch (e) { return fb; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* ignore */ } },
};

/* ---------- theme (same key and events as the guide) ---------- */
function currentTheme() {
  const t = document.documentElement.getAttribute('data-theme');
  if (t === 'light' || t === 'dark') return t;
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}
document.querySelectorAll('[data-theme-toggle]').forEach((b) => b.addEventListener('click', () => {
  const next = currentTheme() === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  try { localStorage.setItem('sa-theme', next); } catch (e) { /* ignore */ }
  document.dispatchEvent(new CustomEvent('themechange', { detail: { theme: next } }));
}));
// count the lab as a read chapter in the guide's progress
(() => { const v = new Set(store.get('sa-visited', [])); if (!v.has('lab')) { v.add('lab'); store.set('sa-visited', [...v]); } })();

/* ---------- data ---------- */
const state = { index: null, indexError: null, datasets: new Map(), manifests: new Map(), method: new Map(), route: null, cleanups: [] };
async function loadJSON(path) {
  const res = await fetch(new URL(path, document.baseURI), { cache: 'no-cache' });
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.json();
}
async function loadIndex() {
  if (state.index) return state.index;
  try {
    const idx = await loadJSON('data/index.json');
    idx.forEach((d) => { if (Array.isArray(d.classes)) d.classes = d.classes.map((c) => (typeof c === 'string' ? speciesCase(c) : c)); });
    state.index = idx;
  } catch (e) { state.indexError = e; state.index = []; }
  return state.index;
}
function loadDataset(id) {
  if (!state.datasets.has(id)) {
    state.datasets.set(id, loadJSON(`data/${id}.json`).then((ds) => {
      (ds.classes || []).forEach((c) => { c.name = speciesCase(c.name); });
      ensureExtraColors(ds.classes?.length || 0);
      return ds;
    }).catch((e) => { state.datasets.delete(id); throw e; }));
  }
  return state.datasets.get(id);
}
/** Index entries first (in their order), then registry datasets that have no JSON yet, as "proposed". */
function mergedList() {
  const idx = state.index || [];
  const seen = new Set(idx.map((d) => d.id));
  const extra = REGISTRY.filter((r) => !seen.has(r.id)).map((r) => ({ id: r.id, title: r.title, representation: r.representation, status: 'proposed', classes: [], n: null, best_accuracy: null, chance: null }));
  const order = new Map(REGISTRY.map((r, i) => [r.id, i]));
  return [...idx, ...extra].sort((a, b) => (order.get(a.id) ?? 99) - (order.get(b.id) ?? 99));
}
/** "579 dead / 798 live" for a few classes, "15 classes (27–30 each)" for many. */
function classSummary(classes, max = 4) {
  if (!classes?.length) return '[not run]';
  if (classes.length <= max) return classes.map((c) => `${Number.isFinite(c.n) ? thousands(c.n) : '?'} ${esc(c.name)}`).join(' / ');
  const ns = classes.map((c) => c.n).filter(Number.isFinite);
  return `${classes.length} classes${ns.length ? ` (${Math.min(...ns)}–${Math.max(...ns)} each)` : ''}`;
}
const registryOf = (id) => REGISTRY.find((r) => r.id === id);
/* ---------- caveats (a dataset JSON's `caveats`, index.json's `has_caveat`) ---------- */
const caveatsOf = (ds) => (Array.isArray(ds?.caveats) ? ds.caveats.filter((c) => typeof c === 'string' && c.trim()) : []);
/** The registry's note after a caveated dataset's title, as plain text ("" when there is none). */
const titleNote = (id) => registryOf(id)?.titleNote || '';
/** The title with its note, for document.title and the brand line. */
const titleWithNote = (id, t) => (titleNote(id) ? `${t} (${titleNote(id)})` : t);
const firstSentence = (t) => { const m = /^.*?[.!?](?=\s|$)/.exec(String(t || '').trim()); return m ? m[0] : String(t || '').trim(); };
/** The "!" marker: a visible badge with an accessible label. */
function caveatMark(label) { return `<span class="caveat-mark" role="img" aria-label="${esc(label)}" title="${esc(label)}">!</span>`; }
/** One sentence the home page can show for a caveated dataset: the JSON's first caveat sentence, else the registry note. */
function caveatShort(id, ds) {
  const c = caveatsOf(ds)[0];
  if (c) return firstSentence(c);
  const n = titleNote(id);
  return n ? `${n.charAt(0).toUpperCase()}${n.slice(1)}.` : 'This dataset has a caveat; see its page.';
}
const shortName = (d) => registryOf(d.id)?.short || d.title.split(',')[0];
const methodNA = (m) => m && (m.status === 'skipped' || m.status === 'not_applicable');
const methodAvailable = (m) => m && !methodNA(m) && Number.isFinite(m.accuracy);
const methodDef = (key) => METHODS.find((x) => x.key === key);
/** The protocol a method block records, as the JSON writes it. */
const protocolOf = (m) => (m && typeof m.protocol === 'string' && m.protocol ? m.protocol : null);
const protocolText = (m) => (protocolOf(m) ? esc(protocolOf(m)) : '[protocol not recorded in the JSON]');
/** Where a method's mistakes and matrix were scored (a property of the method's design, not a number). */
const scoredWhere = (key) => (key === 'pointnet_probe' ? 'on the held-out split' : 'out of fold');
const sizeInfo = (m) => (m && typeof m.size_information === 'string' ? m.size_information : null);
const allSizeCount = (map) => [...map.values()].reduce((n, set) => n + set.size, 0);

/* ---------- drawing ---------- */
function groupScale(shapes, w, hgt, pad) {
  let mx = 0, my = 0;
  for (const S of shapes) for (const p of S) { mx = Math.max(mx, Math.abs(p[0])); my = Math.max(my, Math.abs(p[1])); }
  return Math.min((w / 2 - pad) / (mx || 1), (hgt / 2 - pad) / (my || 1));
}
/** Draw one preshape into an <svg>: polyline for landmarks, closed path for outlines, dots for point clouds. */
function drawShape(S, points, view, { rep = 'landmarks', color = 'var(--ink)', width = 2.6, opacity = 1, dots = false, dotR = 1.8, dash = null }) {
  const g = svg('g', { opacity }, S);
  if (rep === 'pointcloud') {
    points.forEach((p) => svg('circle', { cx: (view.cx + view.s * p[0]).toFixed(1), cy: (view.cy + view.s * p[1]).toFixed(1), r: dotR, fill: color }, g));
  } else {
    svg('path', { d: pathOf(points, view, rep === 'outline'), fill: 'none', stroke: color, 'stroke-width': width, 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-dasharray': dash }, g);
    if (dots) points.forEach((p) => svg('circle', { cx: (view.cx + view.s * p[0]).toFixed(1), cy: (view.cy + view.s * p[1]).toFixed(1), r: dotR, fill: color }, g));
  }
  return g;
}
function thumb(points, { rep, color, w = 120, hgt = 80, scale, label, dots = false, width = 2.6 }) {
  const S = svg('svg', { viewBox: `0 0 ${w} ${hgt}`, role: 'img', 'aria-label': label });
  const s = scale || groupScale([points], w, hgt, 8);
  drawShape(S, points, { cx: w / 2, cy: hgt / 2, s }, { rep, color, dots, width });
  return S;
}
/** Decorative glyph for a dataset card when no example shapes are loaded yet. */
function glyph(rep, muted = false) {
  const S = svg('svg', { viewBox: '0 0 120 56', 'aria-hidden': 'true' });
  const c0 = muted ? 'var(--slate)' : 'var(--coral)', c1 = muted ? 'var(--slate)' : 'var(--teal)';
  if (rep === 'pointcloud') {
    svg('ellipse', { cx: 34, cy: 28, rx: 26, ry: 17, fill: 'none', stroke: c0, 'stroke-width': 3, 'stroke-dasharray': '1 5', 'stroke-linecap': 'round' }, S);
    svg('path', { d: 'M70,30 C72,14 96,10 108,20 C118,28 112,46 96,48 C82,50 68,44 70,30 Z', fill: 'none', stroke: c1, 'stroke-width': 3, 'stroke-dasharray': '1 5', 'stroke-linecap': 'round' }, S);
  } else if (rep === 'outline') {
    svg('path', { d: 'M30,50 C8,36 10,14 30,6 C50,14 52,36 30,50 Z', fill: 'none', stroke: c0, 'stroke-width': 3 }, S);
    svg('path', { d: 'M88,50 L72,34 L80,32 L70,18 L84,22 L88,6 L92,22 L106,18 L96,32 L104,34 Z', fill: 'none', stroke: c1, 'stroke-width': 3, 'stroke-linejoin': 'round' }, S);
  } else {
    svg('path', { d: 'M8,30 L18,29.6 L28,29.4 L38,29.2 L48,29.1 L58,29.1 L68,29.2 L78,29.5 L88,29.8 L98,30.1', fill: 'none', stroke: c0, 'stroke-width': 3.5, 'stroke-linecap': 'round' }, S);
    svg('path', { d: 'M14,46 L20,38 L28,31 L38,26 L49,24 L60,25 L70,29 L79,35 L87,41 L94,47', fill: 'none', stroke: c1, 'stroke-width': 3.5, 'stroke-linecap': 'round', opacity: 0.9 }, S);
  }
  return S;
}
function cardThumb(ds) {
  const ex = ds.examples;
  if (!ex) return glyph(ds.representation, ds.status !== 'ready');
  const S = svg('svg', { viewBox: '0 0 120 56', role: 'img', 'aria-label': `example shapes of each class, ${ds.classes.map((c) => c.name).join(' and ')}` });
  const shapes = ds.classes.slice(0, 3).map((c) => ex[String(c.id)]?.[0]).filter(Boolean);
  const s = groupScale(shapes, 120, 56, 5);
  shapes.forEach((p, i) => drawShape(S, p, { cx: 60, cy: 28, s }, { rep: ds.representation, color: stroke(i), width: 3, opacity: 0.9, dotR: 1.3 }));
  return S;
}
function pillStatus(status) {
  if (status === 'ready') return '<span class="pill ready">ready</span>';
  if (status === 'fetch_locally') return '<span class="pill fetch">fetch locally</span>';
  return '<span class="pill proposed">proposed</span>';
}
function banner(kind, html) {
  const icon = kind === 'warn' || kind === 'error'
    ? '<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 9v4"/><path d="M12 17h.01"/><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/></svg>'
    : '<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 8h.01M11 12h1v4h1"/></svg>';
  return h('div', { class: `banner ${kind}`, role: kind === 'error' ? 'alert' : 'status', html: `${icon}<span>${html}</span>` });
}
/** Make a wide box scroll sideways with a visible cue (edge fade + hint) whenever it overflows. */
function scrollable(wrap, label) {
  wrap.classList.add('scroll-x');
  wrap.tabIndex = 0; wrap.setAttribute('role', 'region'); wrap.setAttribute('aria-label', label);
  const hint = h('p', { class: 'scroll-hint', html: '<svg width="14" height="14" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h18M15 6l6 6-6 6"/></svg><span>Wider than the box: scroll sideways to see every column.</span>' });
  wrap.insertAdjacentElement('afterend', hint);
  const update = () => {
    const can = wrap.scrollWidth > wrap.clientWidth + 2;
    wrap.classList.toggle('can-scroll', can); hint.classList.toggle('on', can);
    wrap.classList.toggle('at-end', can && wrap.scrollLeft + wrap.clientWidth >= wrap.scrollWidth - 2);
  };
  wrap.addEventListener('scroll', update, { passive: true });
  if ('ResizeObserver' in window) { const ro = new ResizeObserver(update); ro.observe(wrap); state.cleanups.push(() => ro.disconnect()); }
  requestAnimationFrame(update);
  return update;
}

/* ---------- route plumbing ---------- */
function cleanup() { while (state.cleanups.length) { try { state.cleanups.pop()(); } catch (e) { /* ignore */ } } }
function setTitle(t) { document.title = t ? `${t} · Phenotype Lab · Shape Analysis Field Guide` : 'Phenotype Lab · Shape Analysis Field Guide'; }
function setSub(t) { $('#lab-brand-sub').textContent = t || 'Phenotype Lab'; }
function buildChips(activeId) {
  const nav = $('#lab-chips');
  nav.innerHTML = '';
  const list = mergedList();
  if (!list.length) { nav.hidden = true; return; }
  nav.hidden = false;
  list.forEach((d) => {
    const a = h('a', { class: `chip sm${d.status === 'proposed' ? ' dashed' : ''}`, href: `#/${d.id}`, text: shortName(d) }, nav);
    if (d.id === activeId) { a.classList.add('active'); a.setAttribute('aria-current', 'page'); }
  });
}
function afterRender(focus = true) {
  window.scrollTo({ top: 0, behavior: 'auto' });
  if (focus) { try { main.focus({ preventScroll: true }); } catch (e) { /* ignore */ } }
}
function readChapterLinks(parent, extra = '') {
  const p = h('p', { class: 'links' }, parent);
  p.innerHTML = `<a href="../#10">Read chapter 10: the findings in plain language →</a>${extra}`;
  return p;
}

/* ======================================================================
   1 · Lab home
   ====================================================================== */
async function renderHome() {
  setTitle(''); setSub('Chapter 10'); buildChips(null);
  main.innerHTML = '<p class="loading">Loading the lab…</p>';
  await loadIndex();
  if (state.route !== 'home') return;
  buildChips(null);   // again, now that index.json names the computed datasets
  const list = mergedList();
  const nReady = list.filter((d) => d.status === 'ready').length;
  const nFetch = list.filter((d) => d.status === 'fetch_locally').length;
  const nProp = list.filter((d) => d.status === 'proposed').length;

  main.innerHTML = '';
  if (state.indexError) main.appendChild(banner('warn', `<strong>No results yet.</strong> <code>data/index.json</code> could not be loaded (${esc(state.indexError.message)}). Run <code>python scripts/run_lab.py</code> to compute the lab; until then every dataset is shown as proposed.`));

  const hero = h('section', { class: 'section', style: 'gap:14px; max-width:760px' }, main);
  h('span', { class: 'eyebrow', text: 'Chapter 10 · Putting it together' }, hero);
  h('h1', { text: 'Phenotype Lab' }, hero);
  h('p', { class: 'lede', html: 'Pick a dataset, pick a method, and see whether shape alone can tell the phenotypes apart. Every number on this page is produced by a script in the repository (<code>scripts/run_lab.py</code>) with a fixed seed, under the protocol its JSON records (listed under the leaderboard), and the class sizes are always shown next to the accuracy.' }, hero);
  readChapterLinks(hero, ` <a href="${ATTRIBUTION_URL}" target="_blank" rel="noopener">Sources and licenses (ATTRIBUTION.md)</a>`);

  // 1 · datasets
  const s1 = h('section', { class: 'section' }, main);
  const head1 = h('div', { class: 'section-head' }, s1);
  h('h2', { text: '1 · Choose a dataset' }, head1);
  const parts = [];
  if (nReady) parts.push(`${nReady} computed`);
  if (nFetch) parts.push(`${nFetch} to fetch locally (license)`);
  if (nProp) parts.push(`${nProp} proposed`);
  h('span', { class: 'small', text: parts.join(' · ') + '.' }, head1);
  const cards = h('div', { class: 'cards' }, s1);
  const cardEls = new Map();
  list.forEach((d, i) => {
    const a = h('a', { class: `card-ds ${d.status}${i === 0 && d.status === 'ready' ? ' featured' : ''}`, href: `#/${d.id}` }, cards);
    const th = h('div', { class: 'thumb' }, a); th.appendChild(glyph(d.representation, d.status !== 'ready'));
    const txt = h('div', { style: 'display:flex;flex-direction:column;gap:4px' }, a);
    const strong = h('strong', { html: titleHtml(d.title) }, txt);
    if (d.has_caveat) strong.insertAdjacentHTML('beforeend', ` ${caveatMark(`Caveat: ${caveatShort(d.id, null)}`)}`);
    const meta = h('span', { class: 'meta' }, txt);
    meta.textContent = d.status !== 'ready' ? `${REP_LABEL[d.representation] || d.representation}${d.classes?.length ? ' · ' + (d.classes.length > 4 ? d.classes.length + ' classes' : d.classes.join(' / ')) : ''} · [not run]`
      : `${thousands(d.n)} shapes · ${d.classes.length > 4 ? d.classes.length + ' classes' : d.classes.join(' / ') || '[not run]'}`;
    const pills = h('div', { class: 'pills' }, a);
    pills.innerHTML = pillStatus(d.status) + `<span class="pill rep">${esc(REP_LABEL[d.representation] || d.representation)}</span>`;
    // the card's footnote: the registry note now (no layout shift), the JSON's first caveat sentence once loaded
    const cav = d.has_caveat ? h('span', { class: 'card-caveat', html: `<span aria-hidden="true">!</span> ${esc(caveatShort(d.id, null))} See the caveat on its page.` }, a) : null;
    cardEls.set(d.id, { a, th, meta, cav });
  });
  h('div', { class: 'card-own', html: `<strong>Your own masks</strong><span>Turn a folder of labeled masks into an npz in <code>data/</code> with <code>points</code> of shape (K, N, 2) and <code>labels</code> of shape (K,), add an entry to <code>DATASETS</code> in <code>shapekit/lab_datasets.py</code>, run <code>python scripts/run_lab.py --dataset &lt;id&gt;</code>, and the same panels appear here. <a href="../#10/add-your-own-masks">Chapter 10 has the recipe.</a></span>` }, cards);

  // 2 · methods
  const s2 = h('section', { class: 'section' }, main);
  h('h2', { text: '2 · Choose a method' }, s2);
  const chips = h('div', { class: 'chips', role: 'group', 'aria-label': 'Method (highlights its column in the leaderboard)' }, s2);
  const chipEls = METHODS.map((m) => h('button', { type: 'button', class: `chip${m.baseline ? ' baseline' : ''}`, 'aria-pressed': 'false', text: m.long, 'data-key': m.key }, chips));
  h('p', { class: 'small muted', text: 'The three shape methods are chapters you have read: 02–06, 07 and 08. "Size only" is a baseline, not a headline: it shows what size alone would score, so you can see how much a shape method adds. The lab runs every method a dataset supports and reports them side by side. Press one to highlight its column.' }, s2);

  // 3 · leaderboard
  const s3 = h('section', { class: 'section' }, main);
  const head3 = h('div', { class: 'section-head' }, s3);
  h('h2', { text: '3 · Leaderboard' }, head3);
  const when = h('span', { class: 'small', text: 'Accuracy per method · hover or tap a number for its protocol and provenance' }, head3);
  const wrap = h('div', { class: 'table-wrap' }, s3);
  const table = h('table', { class: 'lb' }, wrap);
  table.innerHTML = `<thead><tr><th scope="col">Dataset</th><th scope="col">Classes</th>${METHODS.map((m) => `<th scope="col" data-key="${m.key}"${m.baseline ? ' class="baseline" title="a baseline, not a shape method"' : ''}>${esc(m.name)}${m.baseline ? ' <span class="sub">(baseline)</span>' : ''}</th>`).join('')}<th scope="col">Chance</th></tr></thead>`;
  const tbody = h('tbody', {}, table);
  const rowEls = new Map();
  list.forEach((d) => {
    const tr = h('tr', { class: d.status === 'ready' ? '' : 'row-muted' }, tbody);
    const td0 = h('td', {}, tr);
    if (d.status === 'proposed') td0.innerHTML = titleHtml(d.title); else h('a', { href: `#/${d.id}`, html: titleHtml(d.title), ...(d.has_caveat ? { 'aria-describedby': `lb-caveat-${d.id}` } : {}) }, td0);
    if (d.has_caveat) td0.insertAdjacentHTML('beforeend', ` ${caveatMark('Caveat: see the note below the table')}`);
    const tdc = h('td', { class: 'muted' }, tr);
    tdc.textContent = d.status === 'proposed' ? '[not run]' : (d.classes.length > 4 ? `${d.classes.length} classes` : d.classes.join(' / ') || '[not run]');
    const cells = METHODS.map((m) => h('td', { class: `mono${m.baseline ? ' baseline' : ''}`, 'data-key': m.key, text: d.status === 'ready' ? '…' : (d.status === 'fetch_locally' ? '…' : '—') }, tr));
    const tdch = h('td', { class: 'mono muted', text: d.status === 'proposed' ? '—' : fmt(d.chance) }, tr);
    rowEls.set(d.id, { tr, tdc, cells, tdch });
  });
  scrollable(wrap, 'Leaderboard table');
  // caveat footnotes (one per flagged row), filled with the JSON's caveat and roundness baseline once loaded
  const cavFoot = new Map();
  const cavList = list.filter((d) => d.has_caveat);
  if (cavList.length) {
    const cf = h('div', { class: 'footnotes caveat-notes' }, s3);
    cavList.forEach((d) => cavFoot.set(d.id, h('p', { id: `lb-caveat-${d.id}`, class: 'caveat-foot', html: caveatFootHtml(d, null) }, cf)));
  }
  const foot = h('div', { class: 'footnotes' }, s3);
  const notes = h('div', { class: 'notes' }, s3);
  notes.innerHTML = `
    <div class="note"><strong>Chance</strong> is the accuracy of always guessing the largest class, so a score is only interesting by how far it sits above that number. The bold number in a row is its best shape method; the size-only baseline never counts.</div>
    <div class="note"><strong>Gray rows</strong> are datasets that have not been computed: proposed ones, or ones whose JSON is missing. A license-restricted dataset stays in color once computed, but its page shows numbers only.</div>
    <div class="note"><strong>Click a dataset</strong> for the four-panel story: raw shapes, the aligned cloud, the modes, and the mistakes.</div>`;
  const pager = h('nav', { class: 'pager', 'aria-label': 'Previous and next chapter', style: 'margin-top:0' }, main);
  pager.innerHTML = '<a class="prev" href="../#09"><small>← Previous</small><span>Field notes: bugs and how to avoid them</span></a><a class="next" href="../#10"><small>Next →</small><span>Chapter 10: the findings in plain language</span></a>';

  // method chips highlight a column
  let highlighted = null;
  function highlight(key) {
    highlighted = highlighted === key ? null : key;
    chipEls.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.key === highlighted)));
    table.querySelectorAll('[data-key]').forEach((c) => c.classList.toggle('hl', c.dataset.key === highlighted));
  }
  chipEls.forEach((b) => b.addEventListener('click', () => highlight(b.dataset.key)));

  // fill the per-method columns (and class sizes, thumbnails) from each dataset JSON
  const dates = [];
  const protocols = new Map();   // method key → set of protocol strings seen
  const sizeInfos = new Map();   // method key → set of size_information strings seen
  await Promise.all(list.filter((d) => d.status !== 'proposed').map(async (d) => {
    const row = rowEls.get(d.id), card = cardEls.get(d.id);
    let ds;
    try { ds = await loadDataset(d.id); } catch (e) { row.cells.forEach((c) => { c.textContent = '[not run]'; c.classList.add('na'); }); return; }
    if (state.route !== 'home') return;
    if (ds.computed_at) dates.push(ds.computed_at);
    if (ds.classes?.length && ds.status === 'ready') {
      const cls = ds.classes.length > 4 ? classSummary(ds.classes) : ds.classes.map((c) => `<span ${textAttr(c.id)}>${esc(c.name)}</span>${Number.isFinite(c.n) ? ' ' + thousands(c.n) : ''}`).join(' / ');
      row.tdc.innerHTML = cls;
      card.meta.innerHTML = `${thousands(ds.n)} ${ds.representation === 'pointcloud' ? 'cells' : 'shapes'} · ${ds.n_points ? ds.n_points + ' ' + (ds.representation === 'pointcloud' ? 'unordered points' : ds.representation === 'outline' ? 'outline points' : 'landmarks') + ' · ' : ''}${classSummary(ds.classes)}`;
    }
    if (ds.examples && ds.shapes_shown !== false) { card.th.innerHTML = ''; card.th.appendChild(cardThumb(ds)); }
    if (cavFoot.has(d.id)) cavFoot.get(d.id).innerHTML = caveatFootHtml(d, ds);
    if (card.cav && caveatsOf(ds).length) card.cav.innerHTML = `<span aria-hidden="true">!</span> ${esc(caveatShort(d.id, ds))} See the caveat on its page.`;
    // the best shape method: index.json names it; otherwise the maximum over the shape methods (never the baseline)
    let bestKey = typeof d.best_method === 'string' ? d.best_method : null;
    if (!bestKey) { let best = -1; SHAPE_METHODS.forEach((m) => { const r = ds.methods?.[m.key]; if (methodAvailable(r) && r.accuracy > best) { best = r.accuracy; bestKey = m.key; } }); }
    METHODS.forEach((m, i) => {
      const r = ds.methods?.[m.key], c = row.cells[i];
      if (!r) { c.textContent = '[not run]'; c.classList.add('na'); return; }
      if (methodNA(r)) { c.innerHTML = `<span class="na" title="${esc(r.reason || r.status)}">n/a</span>`; return; }
      if (!Number.isFinite(r.accuracy)) { c.textContent = '[not run]'; c.classList.add('na'); return; }
      const p = protocolOf(r); if (p) { if (!protocols.has(m.key)) protocols.set(m.key, new Set()); protocols.get(m.key).add(p); }
      const si = sizeInfo(r); if (si) { if (!sizeInfos.has(m.key)) sizeInfos.set(m.key, new Set()); sizeInfos.get(m.key).add(si); }
      c.innerHTML = num(fmt(r.accuracy), ds, '', p ? `protocol: ${p}` : '') + (m.key === 'pointnet_probe' && Number.isFinite(r.untrained) ? ` <span class="sub">(${num(fmt(r.untrained), ds, '', p ? `protocol: ${p}` : '')} untrained)</span>` : '');
      if (m.key === bestKey && !m.baseline) c.classList.add('best');
    });
    row.tdch.innerHTML = num(fmt(ds.methods?.landmarks_lr?.chance ?? ds.methods?.pointnet_probe?.chance ?? ds.methods?.spline_lr?.chance ?? ds.chance ?? d.chance), ds);
  }));
  if (state.route !== 'home') return;
  if (dates.length) when.textContent += ` · computed ${dateOf(dates.sort().at(-1))}`;
  // footnotes from the JSON: the protocol of each method and what every method sees
  const protoLines = METHODS.map((m) => {
    const set = protocols.get(m.key);
    return `<strong>${esc(m.name)}:</strong> ${set ? [...set].map(esc).join(' / ') : '[protocol not recorded in the JSON]'}`;
  });
  foot.innerHTML = `<p><strong>Protocols</strong> (as each method's JSON block records them). ${protoLines.join(' · ')}</p>`;
  const shapeSize = new Set(SHAPE_METHODS.flatMap((m) => [...(sizeInfos.get(m.key) || [])]));
  const baseSize = [...(sizeInfos.get('size_lr') || [])];
  if (shapeSize.size === 1 && shapeSize.has('removed')) {
    foot.insertAdjacentHTML('beforeend', `<p><strong>Size.</strong> Every shape method above sees size-normalized shapes (<code>size_information: "removed"</code> in each of their JSON blocks), including PointNet; the Size-only column sees ${baseSize.length ? `size ${baseSize.map(esc).join(' / ')}` : 'size alone'}, measured separately, so a shape method is only interesting by how far it sits above both chance and that column.</p>`);
  } else if (allSizeCount(sizeInfos)) {
    const per = METHODS.filter((m) => sizeInfos.has(m.key)).map((m) => `${esc(m.name)}: ${[...sizeInfos.get(m.key)].map(esc).join(' / ')}`).join(' · ');
    foot.insertAdjacentHTML('beforeend', `<p><strong>Size.</strong> What each method sees, from its <code>size_information</code> field: ${per}.</p>`);
  } else {
    foot.insertAdjacentHTML('beforeend', '<p><strong>Size.</strong> The JSON does not record <code>size_information</code> for these methods, so whether size was removed before each method is not stated here.</p>');
  }
}

/* ======================================================================
   2 · Dataset page
   ====================================================================== */
function classCounts(ds) {
  return ds.classes.map((c) => `<strong ${textAttr(c.id)}>${Number.isFinite(c.n) ? thousands(c.n) : '?'} ${esc(c.name)}</strong>`).join(', ');
}
function licenseUrl(source) {
  if (!source) return null;
  if (typeof source.license_url === 'string' && source.license_url) return source.license_url;
  const name = source.license || '';
  for (const [re, url] of LICENSE_URLS) if (re.test(name)) return url;
  return null;
}
/** "Data: title (link) · authors · license (link) · modified: … · ATTRIBUTION.md" from the JSON's source block. */
function dataLine(ds) {
  const s = ds.source || {};
  const parts = [];
  const name = mdCode(s.name || ds.title || '');
  parts.push(`Data: ${s.url ? `<a href="${esc(s.url)}" target="_blank" rel="noopener">${name}</a>` : name}`);
  if (s.citation) parts.push(esc(s.citation));
  if (s.license) { const u = licenseUrl(s); parts.push(`License: ${u ? `<a href="${esc(u)}" target="_blank" rel="noopener">${esc(s.license)}</a>` : esc(s.license)}`); }
  const mods = Array.isArray(ds.preprocessing) ? ds.preprocessing.filter((x) => typeof x === 'string' && x) : (typeof ds.preprocessing === 'string' ? [ds.preprocessing] : []);
  if (mods.length) parts.push(`modified: ${mods.map(esc).join('; ')}`);
  parts.push(`<a href="${ATTRIBUTION_URL}" target="_blank" rel="noopener">ATTRIBUTION.md</a>`);
  parts.push(`Numbers: <code>${esc(ds.script || 'scripts/run_lab.py')}</code>${ds.computed_at ? ', ' + esc(dateOf(ds.computed_at)) : ''}`);
  return h('p', { class: 'data-line', html: parts.join(' · ') + '.' });
}
function fetchBox(ds, reg) {
  const cmd = ds.fetch_command || reg?.fetch || `python scripts/fetch_datasets.py --${ds.id}`;
  const box = h('section', { class: 'fetch-box' });
  h('h2', { text: ds.status === 'proposed' ? 'Not computed yet' : 'Fetch this dataset locally' }, box);
  const lic = ds.source?.license ? `<p class="soft">License: <strong>${esc(ds.source.license)}</strong>${ds.source.redistributed === false ? ' — as a precaution this repository ships no coordinates, means, modes or weights derived from it.' : ''}</p>` : '';
  box.insertAdjacentHTML('beforeend', lic);
  box.insertAdjacentHTML('beforeend', `<p class="soft">To see the results on your own machine:</p>
    <ol><li>Download and extract the data (checksums are printed): <pre><code>${esc(cmd)}</code></pre></li>
    <li>Compute the lab for it: <pre><code>python scripts/run_lab.py --dataset ${esc(ds.id)}</code></pre></li>
    <li>Reload this page: <code>docs/lab/data/${esc(ds.id)}.json</code> now holds the numbers${ds.shapes_shown === false ? ' (accuracies, confusion matrices and counts only; no coordinates are written for this license)' : ''}.</li></ol>`);
  if (ds.source?.url) box.insertAdjacentHTML('beforeend', `<p class="small muted">Source: ${mdCode(ds.source.name || '')} · <a href="${esc(ds.source.url)}" target="_blank" rel="noopener">${esc(ds.source.url)}</a></p>`);
  return box;
}
/** The confusion matrix of one method. Row totals are the matrix's own row sums (the held-out split for PointNet). */
function confusionTable(ds, m, key) {
  const C = ds.classes.length, cm = m.cm, big = C > 5;
  const name = methodDef(key)?.name || '';
  const rowN = cm.map((r) => r.reduce((a, b) => a + b, 0));
  const total = rowN.reduce((a, b) => a + b, 0);
  const heldOut = rowN.some((n, i) => Number.isFinite(ds.classes[i]?.n) && n !== ds.classes[i].n);
  const t = h('table', { class: `cm${big ? ' big' : ''}`, 'aria-label': `Confusion matrix, ${name}: rows are the true class, columns what the classifier called them` });
  t.innerHTML = `<thead><tr><th scope="col"></th>${ds.classes.map((c) => `<th scope="col" title="called ${esc(c.name)}"><span>called ${esc(big ? shortClass(c.name) : c.name)}</span></th>`).join('')}</tr></thead>`;
  const tb = h('tbody', {}, t);
  // highlight the most frequent confusion (largest off-diagonal cell)
  let hi = null, hiv = -1;
  for (let i = 0; i < C; i++) for (let j = 0; j < C; j++) if (i !== j && cm[i][j] > hiv) { hiv = cm[i][j]; hi = [i, j]; }
  ds.classes.forEach((c, i) => {
    const tr = h('tr', {}, tb);
    const th = h('th', { scope: 'row', class: textClass(i), text: `${big ? shortClass(c.name) : c.name} (${thousands(rowN[i])})`, title: `${c.name}: ${thousands(rowN[i])} ${heldOut ? 'held-out ' : ''}shapes scored` }, tr);
    th.style.color = textColor(i);
    for (let j = 0; j < C; j++) {
      const isHi = hi && hi[0] === i && hi[1] === j && hiv > 0;
      const td = h('td', { class: (i === j ? 'diag ' : '') + (isHi ? 'hi' : ''), html: num(thousands(cm[i][j]), ds) }, tr);
      if (i === j) td.style.background = `color-mix(in srgb, ${stroke(i)} 14%, transparent)`;
      if (isHi) td.title = 'the most frequent confusion';
    }
  });
  const cap = h('caption', {}, t);
  cap.textContent = `Rows: true class, with the number of ${heldOut ? 'held-out' : ''} shapes scored per class; columns: what the classifier called them.${heldOut ? ` ${thousands(total)} held-out shapes, not the full ${thousands(ds.n)}.` : ''}`;
  return { table: t, hi, hiv, rowN, heldOut, total };
}
function confusionSentence(ds, m, info) {
  const { hi, hiv, rowN, heldOut } = info;
  const thing = ds.representation === 'pointcloud' ? 'cells' : 'shapes';
  if (!hi || hiv <= 0) return `No ${heldOut ? 'held-out ' : ''}${thing} were mis-called: every count sits on the diagonal.`;
  const cm = m.cm, [a, b] = hi, na = ds.classes[a], nb = ds.classes[b];
  const rate = (k, n) => (n ? `${(100 * k / n).toFixed(0)} %` : '?');
  const ra = rowN[a] ? cm[a][b] / rowN[a] : 0, rb = rowN[b] ? cm[b][a] / rowN[b] : 0;
  const s1 = `<strong>${thousands(cm[a][b])} of ${thousands(rowN[a])} ${esc(na.name)}</strong> ${heldOut ? 'held-out ' : ''}${thing} are called ${esc(nb.name)} (${rate(cm[a][b], rowN[a])})`;
  const s2 = `${thousands(cm[b][a])} of ${thousands(rowN[b])} ${esc(nb.name)} are called ${esc(na.name)} (${rate(cm[b][a], rowN[b])})`;
  let asym = '';
  if (rb > 0 ? ra > 2 * rb : ra > 0) {
    // lopsided as a share of each class; say whether prevalence alone could explain it
    const towardsLarger = Number.isFinite(na.n) && Number.isFinite(nb.n) && nb.n > na.n;
    asym = towardsLarger
      ? ` The errors lean toward ${esc(nb.name)}, the larger class: an unsure classifier tends to pick the common label, so this lopsidedness is at least partly prevalence, not necessarily shape.`
      : ` The errors run against prevalence (${esc(nb.name)} is not the larger class), so they are more likely a property of the shapes: some ${esc(na.name)} ${thing} look like ${esc(nb.name)}, and few the other way round.`;
  }
  return `${s1}, while ${s2}.${asym} The gold cell marks the most frequent confusion.`;
}
const closedOf = (ds, w) => (w.closed ?? ds.closed ?? ds.representation === 'outline');
function mistakeFigure(ds, w, scale) {
  const fig = h('figure', { class: 'mistake' });
  const t = ds.classes[w.true]?.name ?? w.true, p = ds.classes[w.pred]?.name ?? w.pred;
  if (Array.isArray(w.curve)) {
    // spline route: draw the sampled curve, with the control polygon faint behind it
    const S = svg('svg', { viewBox: '0 0 120 80', role: 'img', 'aria-label': `shape ${w.index}, ${t}, called ${p}` });
    const view = { cx: 60, cy: 40, s: scale };
    drawShape(S, w.shape, view, { rep: closedOf(ds, w) ? 'outline' : 'landmarks', color: 'var(--slate)', width: 1.2, opacity: 0.6, dots: true, dotR: 1.6, dash: '2 2' });
    drawShape(S, w.curve, view, { rep: closedOf(ds, w) ? 'outline' : 'landmarks', color: stroke(w.true), width: 2.6 });
    fig.appendChild(S);
  } else if (Array.isArray(w.shape)) fig.appendChild(thumb(w.shape, { rep: ds.representation, color: stroke(w.true), scale, label: `shape ${w.index}, ${t}, called ${p}` }));
  fig.insertAdjacentHTML('beforeend', `<figcaption>#${esc(w.index)} · <span ${textAttr(w.true)}>${esc(t)}</span> · called <span ${textAttr(w.pred)}>${esc(p)}</span>, p = ${num(fmt(w.p, 2), ds)}</figcaption>`);
  return fig;
}
function crumbsHtml(items) {
  return items.map(([text, href], i) => (i ? '<span aria-hidden="true">›</span>' : '') + (href ? `<a class="crumb" href="${href}">${text}</a>` : `<span class="crumb">${text}</span>`)).join('');
}

/* ---------- 0 · From pixels (sample images + overlays from data/images/<id>/manifest.json) ---------- */
const IMAGES_DIR = (id) => `data/images/${id}/`;
/** Load a dataset's image manifest; only registry entries with `images: 'manifest'` are ever requested. */
function loadManifest(id) {
  if (registryOf(id)?.images !== 'manifest') return Promise.resolve(null);
  if (!state.manifests.has(id)) {
    state.manifests.set(id, loadJSON(`${IMAGES_DIR(id)}manifest.json`).catch((e) => { state.manifests.delete(id); throw e; }));
  }
  return state.manifests.get(id);
}
/** "LMNA−/−", "lmna-/-" and "LMNA" all read as the same class name; so do "wild-type" and "wildtype". */
const classKey = (s) => String(s).toLowerCase().replace(/−/g, '-').replace(/[^a-z0-9]/g, '');
/** The class index an image (or overlay) names: an integer id, or a class name matched loosely. */
function classIndexOf(ds, cls) {
  const classes = ds?.classes || [];
  if (cls === null || cls === undefined || cls === '') return null;
  if (Number.isInteger(cls)) return classes.some((c) => c.id === cls) ? cls : null;
  const k = classKey(cls);
  const hit = classes.find((c) => classKey(c.name) === k || String(c.id) === String(cls)) || (k ? classes.find((c) => classKey(c.name).startsWith(k) || k.startsWith(classKey(c.name))) : null);
  return hit ? hit.id : null;
}
/** An image's class: its `class_id` when the manifest gives one, else its `class` name. */
const imageClass = (ds, img) => (Number.isInteger(img.class_id) ? classIndexOf(ds, img.class_id) : null) ?? classIndexOf(ds, img.class);
const finitePt = (p) => Array.isArray(p) && Number.isFinite(p[0]) && Number.isFinite(p[1]);
/** A short accessible name for an overlay layer: "15 worm outlines and landmarks", "11 cell outlines" (the per-shape labels stay in each <title>). */
function overlaySummary(ovs, word = 'shape') {
  const nO = ovs.filter((o) => o.kind !== 'landmarks').length, nL = ovs.length - nO;
  const plural = (n, w) => `${n} ${w}${n === 1 ? '' : 's'}`;
  if (nO && nO === nL) return `${nO} ${word} outline${nO === 1 ? '' : 's'} and landmarks`;
  const parts = [nO ? plural(nO, `${word} outline`) : '', nL ? plural(nL, `${word} landmark set`) : ''].filter(Boolean);
  return parts.length ? parts.join(' and ') : 'no overlays';
}
/** The SVG layer drawn over one image: viewBox = the PNG's pixel frame, so it stays registered at any size. */
function overlaySvg(img, ds) {
  const W = img.width, H = img.height;
  const imgCls = imageClass(ds, img);
  const ovs = (Array.isArray(img.overlays) ? img.overlays : []).filter((o) => Array.isArray(o.points) && o.points.some(finitePt));
  const dotW = Math.max(W, H) / 120;   // landmark dot diameter in image pixels (5.8 px on a 696-px BBBC010 field)
  const S = svg('svg', { class: 'px-ov', viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: 'none', role: 'img', 'aria-label': overlaySummary(ovs, registryOf(ds?.id)?.objWord) });
  ovs.forEach((o) => {
    const ci = o.class !== undefined ? classIndexOf(ds, o.class) : imgCls;
    const col = ci === null ? 'var(--gold)' : stroke(ci);
    const pts = o.points.filter(finitePt);
    const g = svg('g', { 'data-kind': o.kind || 'outline' }, S);
    if (o.label) svg('title', { text: o.label }, g);
    const d = pts.map((p, i) => `${i ? 'L' : 'M'}${(+p[0]).toFixed(1)},${(+p[1]).toFixed(1)}`).join('') + (o.closed ? 'Z' : '');
    if (o.kind === 'landmarks') {
      // a thin polyline through the landmarks (fixed screen width), then each landmark as a round dot sized in
      // image pixels, so a dense field stays readable on a phone and the dots grow with the larger view
      svg('path', { d, class: 'halo', 'stroke-width': 3 }, g);
      svg('path', { d, class: 'line', stroke: col, 'stroke-width': 1.2 }, g);
      const dots = pts.map((p) => `M${(+p[0]).toFixed(1)},${(+p[1]).toFixed(1)}h0`).join('');
      svg('path', { d: dots, class: 'halo dot', 'stroke-width': (dotW * 1.45).toFixed(2) }, g);
      svg('path', { d: dots, class: 'line dot', stroke: col, 'stroke-width': dotW.toFixed(2) }, g);
    } else {
      svg('path', { d, class: 'halo', 'stroke-width': 3.6 }, g);
      svg('path', { d, class: 'line', stroke: col, 'stroke-width': 1.8 }, g);
    }
  });
  return S;
}
const KIND_WORDS = { outline: ['outline', 'outlines'], landmarks: ['landmarks', 'landmarks'] };
/** "Show outlines and landmarks": the toggle names what the manifest actually draws (constant label; aria-pressed is the state). */
function overlayLabel(img) {
  const ovs = Array.isArray(img.overlays) ? img.overlays : [];
  const kinds = [...new Set(ovs.map((o) => (o.kind === 'landmarks' ? 'landmarks' : 'outline')))];
  const words = kinds.map((k) => KIND_WORDS[k][ovs.filter((o) => (o.kind === 'landmarks' ? 'landmarks' : 'outline') === k).length > 1 ? 1 : 0]);
  return words.length ? `Show ${words.join(' and ')}` : null;
}
/** One image with its overlay: an <img> (lazy, width/height set) under an absolutely positioned SVG. */
function pixelFrame(id, img, ds, { lazy = true } = {}) {
  const frame = h('div', { class: 'px-frame' });
  h('img', { src: IMAGES_DIR(id) + encodeURIComponent(img.file), alt: img.alt || '', width: img.width, height: img.height, loading: lazy ? 'lazy' : null, decoding: 'async' }, frame);
  const ov = overlaySvg(img, ds);
  ov.setAttribute('hidden', '');
  frame.appendChild(ov);
  return { frame, ov };
}
function overlayToggle(label, ovs) {
  const b = h('button', { type: 'button', class: 'chip sm px-toggle', 'aria-pressed': 'false', text: label });
  b.addEventListener('click', () => {
    const on = b.getAttribute('aria-pressed') !== 'true';
    b.setAttribute('aria-pressed', String(on));
    ovs().forEach((o) => o.toggleAttribute('hidden', !on));
  });
  return b;
}
/** The larger view: one shared <dialog>; Escape (native) or Close shuts it and focus returns to the opener. */
function pixelDialog(parent) {
  const dlg = h('dialog', { class: 'px-dialog', 'aria-labelledby': 'px-dialog-title' }, parent);
  let opener = null;
  dlg.addEventListener('close', () => { dlg.querySelector('.px-dialog-body')?.replaceChildren(); if (opener?.isConnected) opener.focus(); opener = null; });
  // a click on the backdrop (outside the dialog box) closes it too
  dlg.addEventListener('click', (e) => { if (e.target === dlg) dlg.close(); });
  return {
    open(id, img, ds, from, overlayOn) {
      opener = from;
      dlg.innerHTML = '';
      const head = h('div', { class: 'px-dialog-head' }, dlg);
      h('h2', { id: 'px-dialog-title', text: firstSentence(img.caption || img.alt || img.file) }, head);
      const close = h('button', { type: 'button', class: 'btn', text: 'Close' }, head);
      close.addEventListener('click', () => dlg.close());
      const body = h('div', { class: 'px-dialog-body' }, dlg);
      const { frame, ov } = pixelFrame(id, img, ds, { lazy: false });
      // as large as fits: never wider than the box, never taller than the viewport, at most 1.5× the PNG
      frame.style.width = `min(100%, ${Math.round(img.width * 1.5)}px, calc((100dvh - 190px) * ${(img.width / img.height).toFixed(4)}))`;
      body.appendChild(frame);
      const tools = h('div', { class: 'px-tools' }, dlg);
      if (img.caption && firstSentence(img.caption) !== String(img.caption).trim()) h('p', { class: 'small', html: mdCode(img.caption) }, tools);
      const label = overlayLabel(img);
      if (label) {
        const t = overlayToggle(label, () => [ov]);
        tools.appendChild(t);
        if (overlayOn) t.click();
      }
      if (img.processing) h('p', { class: 'small muted', html: mdCode(img.processing) }, tools);
      dlg.showModal();
      close.focus();
    },
  };
}
/** Why a dataset has no images here, worded per registry entry (never a legal claim beyond the license's own terms). */
function noImagesHtml(id, ds, reg) {
  const lic = ds?.source?.license ? ds.source.license.split(/\s*[(;:]/)[0].trim() : '';
  const licU = licenseUrl(ds?.source);
  const licHtml = lic ? (licU ? `<a href="${esc(licU)}" target="_blank" rel="noopener">${esc(lic)}</a>` : esc(lic)) : '';
  const fetchCmd = `python scripts/fetch_datasets.py --images ${id}`;
  const cmd = (what) => `<p class="reading">${what}</p><pre><code>${esc(fetchCmd)}</code></pre>`;
  switch (reg?.images) {
    case 'none':
      return '<p class="reading">This dataset is landmark coordinates; there are no images. The R package it comes from ships the coordinates only, so the page starts from the landmarks in panel 1.</p>';
    case 'unverified':
      return `<p class="reading">The outlines on this page are redistributed under their own license${licHtml ? ` (${licHtml})` : ''}. The photographs they were traced from are a different matter: the photographs' license could not be verified, so they are not redistributed.</p>`
        + cmd('To look at the photographs on your own machine (downloaded into <code>data/_local/images/leaves/</code>, which git ignores; the script prints the checksums, the license statement it found and the attribution it requires):');
    case 'nc_nd':
      return `<p class="reading">The images are licensed ${licHtml || 'CC BY-NC-ND 4.0'}: no derivatives and no commercial use. Web-ready samples would be modified copies (cropped, converted to PNG), which the no-derivatives term does not allow, so, as a precaution, no image from this dataset is in the repository.</p>`
        + cmd(`To look at them on your own machine, fetch the dataset first with <code>${esc(reg.fetch || 'python scripts/fetch_datasets.py --rbc')}</code> (into <code>data/_local/</code>, which git ignores); this command then points to the extracted image tree:`);
    default:
      return '<p class="reading">No sample images are registered for this dataset.</p>';
  }
}
/** Panel 0 · From pixels. `man` is the manifest (or null), `manErr` the reason it could not be loaded. */
function pixelsPanel(id, ds, reg, man, manErr) {
  const sec = h('section', { class: 'panel wide px-panel', 'aria-labelledby': 'p0h' });
  const imgs = Array.isArray(man?.images) ? man.images.filter((im) => im && im.file && Number.isFinite(im.width) && Number.isFinite(im.height)) : [];
  sec.innerHTML = `<div class="panel-head"><h2 id="p0h">0 · From pixels</h2><span class="small">${imgs.length ? `${countWord(imgs.length)} sample image${imgs.length === 1 ? '' : 's'}, the pixels the shapes come from` : (reg?.images === 'none' ? 'no images exist' : 'no images in the repository')}</span></div>`;
  if (reg?.images !== 'manifest') { sec.insertAdjacentHTML('beforeend', noImagesHtml(id, ds, reg)); return sec; }
  if (!imgs.length) {
    sec.insertAdjacentHTML('beforeend', `<p class="reading">The sample images have not been generated yet: <code>${esc(IMAGES_DIR(id))}manifest.json</code> ${manErr ? `could not be loaded (${esc(manErr.message)})` : 'lists no images'}. <code>python scripts/make_sample_images.py</code> builds them from the originals that <code>python scripts/fetch_datasets.py --images ${esc(id)}</code> downloads.</p>`);
    return sec;
  }
  // legend: the class colors and the overlay kinds actually present
  const usedClasses = [...new Set(imgs.map((im) => imageClass(ds, im)))].sort((x, y) => (x === null) - (y === null) || x - y);
  const kinds = new Set(imgs.flatMap((im) => (im.overlays || []).map((o) => (o.kind === 'landmarks' ? 'landmarks' : 'outline'))));
  const legend = h('div', { class: 'px-legend', 'aria-label': 'Overlay legend' }, sec);
  legend.innerHTML = `<span class="lbl">Overlay colors:</span>${usedClasses.map((ci) => (ci === null
    ? '<span><i class="sw" style="background:var(--gold)"></i>class not matched</span>'
    : `<span><i class="sw" style="background:${stroke(ci)}"></i><span ${textAttr(ci)}>${esc(ds.classes.find((c) => c.id === ci)?.name ?? ci)}</span></span>`)).join('')}`
    + (kinds.has('outline') ? '<span><i class="kind line" aria-hidden="true"></i>outline</span>' : '')
    + (kinds.has('landmarks') ? '<span><i class="kind dots" aria-hidden="true"><b></b><b></b><b></b></i>landmarks</span>' : '');
  const grid = h('div', { class: 'px-grid' }, sec);
  const dialog = pixelDialog(sec);
  imgs.forEach((img) => {
    const fig = h('figure', { class: 'px-fig' }, grid);
    const ci = imageClass(ds, img);
    const { frame, ov } = pixelFrame(id, img, ds);
    const zoom = h('button', { type: 'button', class: 'px-zoom', 'aria-label': `View larger: ${img.alt || img.file}`, title: 'View larger' }, fig);
    zoom.appendChild(frame);
    h('span', { class: 'px-zoom-badge', 'aria-hidden': 'true', html: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/></svg>' }, zoom);
    const tools = h('div', { class: 'px-tools' }, fig);
    const label = overlayLabel(img);
    let toggle = null;
    if (label) { toggle = overlayToggle(label, () => [ov]); tools.appendChild(toggle); }
    zoom.addEventListener('click', () => dialog.open(id, img, ds, zoom, toggle?.getAttribute('aria-pressed') === 'true'));
    const cap = h('figcaption', {}, fig);
    const clsName = ci === null ? (img.class ? esc(img.class) : '') : `<strong ${textAttr(ci)}>${esc(ds.classes.find((c) => c.id === ci)?.name ?? img.class)}</strong>`;
    cap.innerHTML = `<span>${clsName ? clsName + ' · ' : ''}${mdCode(img.caption || '')}</span>`
      + (img.processing ? `<span class="px-proc">${mdCode(img.processing)}</span>` : '')
      + ((img.source_file || img.channel) ? `<span class="px-proc">${img.source_file ? `Source file <code>${esc(img.source_file)}</code>` : ''}${img.source_file && img.channel ? ', ' : ''}${img.channel ? `channel ${esc(img.channel)}` : ''}.</span>` : '');
  });
  // credit line: title → url, authors, license → license_url, the notice file on GitHub
  const c = man.credit || {};
  const parts = [];
  if (c.title) parts.push(`Images: ${c.url ? `<a href="${esc(c.url)}" target="_blank" rel="noopener">${mdCode(c.title)}</a>` : mdCode(c.title)}`);
  if (c.authors) parts.push(esc(Array.isArray(c.authors) ? c.authors.join(', ') : c.authors));
  if (c.copyright) parts.push(esc(c.copyright));
  if (c.license) { const u = c.license_url || licenseUrl({ license: c.license }); parts.push(`License: ${u ? `<a href="${esc(u)}" target="_blank" rel="noopener">${esc(c.license)}</a>` : esc(c.license)}`); }
  if (c.notice_file) parts.push(`<a href="${REPO_URL}/blob/main/docs/lab/${esc(IMAGES_DIR(id))}${encodeURIComponent(c.notice_file)}" target="_blank" rel="noopener">${esc(c.notice_file)}</a>`);
  if (parts.length) h('p', { class: 'data-line', html: parts.join(' · ') + '.' }, sec);
  const extra = [c.license_note ? mdCode(c.license_note) : '', c.related_paper ? `Related paper: ${mdCode(c.related_paper)}` : '', c.overlay_credit ? `Overlays: ${mdCode(c.overlay_credit)}.` : '', c.citation ? `Requested citation: ${mdCode(c.citation)}` : ''].filter(Boolean);
  if (extra.length) h('p', { class: 'data-line', html: extra.join(' ') }, sec);
  h('p', { class: 'small muted', html: `Overlays are drawn in the PNG's own pixel frame, so they stay on the cells at any size. The full image set: <code>python scripts/fetch_datasets.py --images ${esc(id)}</code> (into <code>data/_local/images/${esc(id)}/</code>, never committed).` }, sec);
  return sec;
}

/** Leaderboard footnote for a caveated row: the first caveat sentence and, when computed, the roundness baseline. */
function caveatFootHtml(d, ds) {
  const rb = ds?.roundness_baseline;
  const best = ds && typeof d.best_method === 'string' ? ds.methods?.[d.best_method] : null;
  const name = METHODS.find((m) => m.key === d.best_method)?.name;
  let extra = '';
  if (rb && Number.isFinite(rb.accuracy)) {
    extra = ` ${esc((rb.label || 'roundness only').charAt(0).toUpperCase() + (rb.label || 'roundness only').slice(1))} scores ${num(fmt(rb.accuracy), ds, '', protocolOf(rb) ? `protocol: ${protocolOf(rb)}` : '')}`
      + (best && Number.isFinite(best.accuracy) ? `, against ${num(fmt(best.accuracy), ds, '', protocolOf(best) ? `protocol: ${protocolOf(best)}` : '')} for the best method (${[name ? esc(name) : '', protocolOf(best) !== protocolOf(rb) ? 'different protocols' : ''].filter(Boolean).join('; ')})`.replace(' ()', '') : '') + '.';
  }
  return `${caveatMark('Caveat')} <strong>${esc(shortName(d))}:</strong> ${esc(caveatShort(d.id, ds))}${extra} <a href="#/${esc(d.id)}">Read the caveat</a>.`;
}
/** The caveat box at the top of a dataset page (above panel 0): every caveat, then the roundness baseline next to the methods. */
function caveatBox(id, ds) {
  const cav = caveatsOf(ds);
  if (!cav.length) return null;
  const box = h('section', { class: 'caveat-box', 'aria-labelledby': 'cavh' });
  const note = titleNote(id);
  h('div', { class: 'caveat-head', html: `<span class="caveat-mark big" aria-hidden="true">!</span><h2 id="cavh">Caveat${note ? `: ${esc(note)}` : ''}</h2>` }, box);
  cav.forEach((c, i) => h('p', { class: i === 0 ? 'caveat-lead' : 'caveat-more', html: mdCode(c) }, box));
  const rb = ds.roundness_baseline;
  if (rb && Number.isFinite(rb.accuracy)) {
    const rows = [[rb.label || 'roundness only', rb, true], ...METHODS.filter((m) => methodAvailable(ds.methods?.[m.key])).map((m) => [m.baseline ? `${m.name} (baseline)` : m.name, ds.methods[m.key], false])];
    const tiles = h('div', { class: 'caveat-tiles', role: 'list', 'aria-label': 'Roundness baseline next to the methods' }, box);
    rows.forEach(([label, r, diag]) => {
      const p = protocolOf(r);
      h('div', { class: `tile${diag ? ' diag' : ''}`, role: 'listitem', html: `<span>${esc(label)}</span><strong>${num(fmt(r.accuracy), ds, '', p ? `protocol: ${p}` : '')}</strong><span class="proto">${esc(p || '')}</span>` }, tiles);
    });
    const cls = ds.classes || [];
    const med = (arr, d) => (Array.isArray(arr) ? cls.map((c) => `<span ${textAttr(c.id)}>${esc(c.name)}</span> ${num(fmt(arr[c.id], d), ds)}`).join(', ') : '');
    const circ = med(rb.circularity_median_by_class, 2), ar = med(rb.axis_ratio_median_by_class, 2);
    h('p', { class: 'caveat-more', html: `<strong>${esc((rb.label || 'roundness only').charAt(0).toUpperCase() + (rb.label || 'roundness only').slice(1))}</strong> is a diagnostic, not a method: each cell's circularity (4πA/P² of its outline${rb.point_order ? `, the points joined in ${esc(rb.point_order)} order` : ''}) and the log of its axis ratio go into a logistic regression, scored like the other methods (the tiles name each protocol; cross-validated and hold-out numbers are not a ranking).${circ ? ` Median circularity: ${circ}.` : ''}${ar ? ` Median axis ratio: ${ar}.` : ''} If two hand-made roundness numbers score about as well as the best method, the classifiers need not be seeing anything beyond roundness, which a circle micropattern would be expected to impose [Inference].` }, box);
  }
  return box;
}

async function renderDataset(id) {
  const reg = registryOf(id);
  setTitle(titleWithNote(id, reg?.title || id)); setSub(titleWithNote(id, reg?.title || id)); buildChips(id);
  main.innerHTML = '<p class="loading">Loading dataset…</p>';
  // the manifest is fetched alongside the dataset and awaited before rendering, so panel 0 never shifts the page
  const manP = loadManifest(id).then((m) => [m, null], (e) => [null, e]);
  await loadIndex();
  let ds = null, err = null;
  try { ds = await loadDataset(id); } catch (e) { err = e; }
  const [man, manErr] = await manP;
  if (state.route !== `ds:${id}`) return;
  buildChips(id);
  main.innerHTML = '';
  const crumbs = h('div', { class: 'crumbs' }, main);
  crumbs.innerHTML = crumbsHtml([['Phenotype Lab', '#/'], [titleHtml(ds?.title || reg?.title || id), null]]);
  if (!ds) {
    if (!reg) { main.insertAdjacentHTML('beforeend', `<h1>Unknown dataset</h1><p class="lede">There is no dataset called <code>${esc(id)}</code>. <a href="#/">Back to the lab.</a></p>`); afterRender(); return; }
    main.insertAdjacentHTML('beforeend', `<div class="ds-head"><div class="text"><h1>${titleHtml(reg.title)}</h1><p class="lede">This dataset is in the registry (<code>shapekit/lab_datasets.py</code>) but has not been computed: <code>data/${esc(id)}.json</code> is missing (${esc(err?.message || '')}).</p></div></div>`);
    main.appendChild(pixelsPanel(id, null, reg, man, manErr));
    main.appendChild(fetchBox({ id, status: 'proposed', source: null, shapes_shown: true }, reg));
    afterRender(); return;
  }
  setTitle(titleWithNote(id, ds.title)); setSub(titleWithNote(id, ds.title));
  const head = h('div', { class: 'ds-head' }, main);
  const text = h('div', { class: 'text' }, head);
  h('h1', { html: titleHtml(ds.title) + (titleNote(id) && caveatsOf(ds).length ? ` <span class="title-note">(${esc(titleNote(id))})</span>` : '') }, text);
  const cavBox = caveatBox(id, ds);
  const thing = ds.representation === 'pointcloud' ? 'cells' : ds.representation === 'outline' ? 'outlines' : 'shapes';
  const ptsWord = ds.representation === 'pointcloud' ? 'unordered points' : ds.representation === 'outline' ? 'outline points' : 'landmarks';
  const classesHtml = ds.classes?.length > 6 ? `<strong>${ds.classes.length} classes</strong> (${esc(classSummary(ds.classes).replace(/^.*\(|\)$/g, ''))})` : classCounts(ds);
  h('p', { class: 'lede', html: ds.status === 'ready'
    ? `${num(thousands(ds.n), ds)} ${thing}${ds.source?.name ? ' from ' + mdCode(ds.source.name) : ''}${ds.n_points ? `, ${num(ds.n_points, ds)} ${ptsWord} each` : ''}. ${ds.classes?.length ? classesHtml + '.' : ''} ${ds.question ? 'Question: ' + esc(ds.question) : ''}`
    : `${ds.source?.name ? mdCode(ds.source.name) + '. ' : ''}${ds.classes?.length ? 'Classes: ' + ds.classes.map((c) => `<span ${textAttr(c.id)}>${esc(c.name)}</span>`).join(', ') + '. ' : ''}${ds.question ? 'Question: ' + esc(ds.question) : ''}` }, text);

  if (ds.status === 'fetch_locally') { if (cavBox) main.appendChild(cavBox); main.appendChild(pixelsPanel(id, ds, reg, man, manErr)); main.appendChild(fetchBox(ds, reg)); main.appendChild(dataLine(ds)); afterRender(); return; }

  // method tabs
  const tabs = h('div', { class: 'chips', role: 'tablist', 'aria-label': 'Method' }, head);
  const avail = METHODS.filter((m) => methodAvailable(ds.methods?.[m.key]));
  let current = state.method.get(id) && methodAvailable(ds.methods?.[state.method.get(id)]) ? state.method.get(id) : (avail.find((m) => !m.baseline)?.key || avail[0]?.key || null);
  const tabEls = METHODS.map((m) => {
    const r = ds.methods?.[m.key];
    const ok = methodAvailable(r);
    const label = ok ? (m.baseline ? `${m.name} (baseline)` : m.name) : `${m.name} (${methodNA(r) ? 'n/a' : 'not run'})`;
    const b = h('button', { type: 'button', class: `chip sm${ok ? '' : ' off'}${m.baseline ? ' baseline' : ''}`, role: 'tab', 'aria-selected': String(m.key === current), text: label, 'data-key': m.key, title: ok ? m.long : (r?.reason || 'not computed for this dataset') }, tabs);
    if (!ok) b.disabled = true;
    return b;
  });

  const engineCheck = ds.engine ? classical.selfCheck(ds) : null;
  if (engineCheck && engineCheck.checked > 0 && !engineCheck.ok) main.appendChild(banner('warn', `The in-browser engine does not reproduce the script's probabilities for this dataset (max |Δp| = ${engineCheck.maxErr.toExponential(1)} over ${engineCheck.checked} shapes; this page's tolerance is 1e-3). The classifier screen may disagree with these numbers.`));

  const shown = ds.shapes_shown !== false;
  if (cavBox) main.appendChild(cavBox);
  const panels = h('div', { class: 'panels' }, main);
  panels.appendChild(pixelsPanel(id, ds, reg, man, manErr));

  if (shown) {
    // panel 1: raw shapes by class
    const p1 = h('section', { class: 'panel', 'aria-labelledby': 'p1h' }, panels);
    const perClass = Math.max(0, ...ds.classes.map((c) => (ds.examples?.[String(c.id)] || []).length));
    const seeded = Array.isArray(ds.example_indices) || !!ds.example_indices;
    p1.innerHTML = `<div class="panel-head"><h2 id="p1h">1 · Raw shapes, by class</h2><span class="small">${perClass ? `${countWord(Math.min(perClass, 5))} ${seeded ? 'seeded examples' : 'examples'} of each, ` : ''}${ds.representation === 'pointcloud' ? 'centered and scaled' : 'Procrustes-aligned'}</span></div>`;
    const grid = h('div', { class: 'thumbs' }, p1);
    const all = []; ds.classes.forEach((c) => (ds.examples?.[String(c.id)] || []).forEach((s) => all.push(s)));
    const sc = groupScale(all, 120, 80, 8);
    ds.classes.forEach((c) => (ds.examples?.[String(c.id)] || []).slice(0, 5).forEach((s, k) => grid.appendChild(thumb(s, { rep: ds.representation, color: stroke(c.id), scale: sc, label: `${c.name} ${k + 1}` }))));
    if (!all.length) grid.insertAdjacentHTML('beforeend', '<p class="muted small">[not run]</p>');
    h('p', { class: 'reading', html: `Colors: ${ds.classes.map((c) => `<span ${textAttr(c.id)}>${esc(c.name)}</span>`).join(', ')}. Position, size and rotation are already removed, so what remains is shape.${ds.representation === 'pointcloud' ? ' The dots are unordered points; the method that classifies them (PointNet) never sees an order.' : ''}` }, p1);

    // panel 2: aligned cloud
    const p2 = h('section', { class: 'panel', 'aria-labelledby': 'p2h' }, panels);
    const sc2 = ds.pca?.scatter || [];
    p2.innerHTML = `<div class="panel-head"><h2 id="p2h">2 · The aligned cloud</h2><span class="small">${sc2.length ? `tangent PCA, ${thousands(sc2.length)} of ${thousands(ds.n)} shown` : 'tangent PCA'}</span></div>`;
    if (sc2.length) {
      const W = 320, H = 200, S = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `Scatter of ${sc2.length} shapes on the first two principal components, colored by class` }, p2);
      svg('line', { x1: 20, y1: 180, x2: 300, y2: 180, stroke: 'var(--line)' }, S);
      svg('line', { x1: 20, y1: 20, x2: 20, y2: 180, stroke: 'var(--line)' }, S);
      const ex = ds.pca.explained || [];
      svg('text', { x: 300, y: 194, 'text-anchor': 'end', style: 'font-size:9px', text: `PC1${ex[0] !== undefined ? ` (${pct(ex[0])})` : ''}` }, S);
      svg('text', { x: 10, y: 14, style: 'font-size:9px', text: `PC2${ex[1] !== undefined ? ` (${pct(ex[1])})` : ''}` }, S);
      let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
      sc2.forEach(([x, y]) => { xmin = Math.min(xmin, x); xmax = Math.max(xmax, x); ymin = Math.min(ymin, y); ymax = Math.max(ymax, y); });
      const sx = (x) => 30 + (260 * (x - xmin)) / ((xmax - xmin) || 1), sy = (y) => 170 - (140 * (y - ymin)) / ((ymax - ymin) || 1);
      // one <g> per class, the larger classes created (hence drawn) first so the smaller stay visible on top
      const byClass = [...ds.classes].sort((a, b) => (b.n || 0) - (a.n || 0));
      const groups = [];
      byClass.forEach((c) => { groups[c.id] = svg('g', { fill: stroke(c.id), 'fill-opacity': 0.6 }, S); });
      sc2.forEach(([x, y, c]) => svg('circle', { cx: sx(x).toFixed(1), cy: sy(y).toFixed(1), r: 2.6 }, groups[c] || groups[ds.classes[0].id]));
      const legend = h('div', { class: 'demo-legend' }, p2);
      ds.classes.forEach((c) => { const s = h('span', {}, legend); h('i', { style: `background:${stroke(c.id)}` }, s); s.appendChild(document.createTextNode(c.name)); });
    } else p2.insertAdjacentHTML('beforeend', '<p class="muted">[not run]</p>');
    h('p', { class: 'reading', text: `Each dot is one shape after position, size and rotation are removed, flattened onto the tangent plane at the mean. Where the colors overlap on these two components, these two directions alone do not separate the classes; any separation then lives in later components, which the classifier also sees.${ds.pca?.note ? ' ' + ds.pca.note.charAt(0).toUpperCase() + ds.pca.note.slice(1) + '.' : ''}` }, p2);

    // panel 3: mode 1
    const p3 = h('section', { class: 'panel', 'aria-labelledby': 'p3h' }, panels);
    p3.innerHTML = '<div class="panel-head"><h2 id="p3h">3 · What varies: mode 1</h2><span class="small">mean ± 1 SD along the first mode</span></div>';
    const mode = ds.pca?.modes?.[0];
    if (mode && ds.pca.mean) {
      const W = 200, H = 120, S = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'The mean shape with the plus and minus one standard deviation shapes of the first mode' }, p3);
      const s3 = groupScale([ds.pca.mean, mode.plus, mode.minus], W, H, 12);
      const view = { cx: W / 2, cy: H / 2, s: s3 };
      drawShape(S, mode.plus, view, { rep: ds.representation, color: 'var(--gold)', width: 2.6, dotR: 1.4 });
      drawShape(S, mode.minus, view, { rep: ds.representation, color: 'var(--lavender)', width: 2.6, dotR: 1.4 });
      drawShape(S, ds.pca.mean, view, { rep: ds.representation, color: 'var(--ink)', width: 3.2, dots: ds.representation !== 'pointcloud', dotR: 1.8 });
      const legend = h('div', { class: 'demo-legend' }, p3);
      [['var(--gold)', '+1 SD'], ['var(--ink)', 'mean'], ['var(--lavender)', '−1 SD']].forEach(([col, lab]) => { const s = h('span', {}, legend); h('i', { style: `background:${col}` }, s); s.appendChild(document.createTextNode(lab)); });
      const tiles = h('div', { class: 'mode-tiles' }, p3);
      (ds.pca.explained || []).slice(0, 4).forEach((e, i) => { tiles.innerHTML += `<div class="tile"><span>mode ${i + 1}</span><strong>${num(pct(e), ds)}</strong></div>`; });
      const e0 = ds.pca.explained?.[0];
      h('p', { class: 'reading', html: Number.isFinite(e0) ? `${num(pct(e0, 0), ds)} of all shape variation lies along one direction. Gold is one standard deviation along it, lavender the opposite way; the ink shape is the Fréchet mean.` : 'Gold and lavender are ± 1 SD along mode 1; ink is the mean.' }, p3);
    } else p3.insertAdjacentHTML('beforeend', '<p class="muted">[not run]</p>');
  } else {
    const lic = h('section', { class: 'panel wide', 'aria-labelledby': 'lich' }, panels);
    const licU = licenseUrl(ds.source);
    lic.innerHTML = `<div class="panel-head"><h2 id="lich">1–3 · No shapes can be shown</h2><span class="small">license</span></div>
      <p class="reading">This dataset is licensed <strong>${licU ? `<a href="${esc(licU)}" target="_blank" rel="noopener">${esc(ds.source?.license || '')}</a>` : esc(ds.source?.license || '')}</strong>. The license forbids sharing adapted material and commercial use, so, as a precaution, this page publishes only counts, accuracies, confusion matrices and the indices of the worst mistakes with their probabilities: no coordinates, means, modes, scatter, neighbors or fitted weights. To see the shapes, fetch the data and compute the lab on your own machine:</p>
      <pre><code>${esc(ds.fetch_command || reg?.fetch || `python scripts/fetch_datasets.py --${ds.id}`)}\npython scripts/run_lab.py --dataset ${esc(ds.id)}</code></pre>
      ${ds.source?.url ? `<p class="small muted">Source: <a href="${esc(ds.source.url)}" target="_blank" rel="noopener">${esc(ds.source.url)}</a></p>` : ''}`;
  }

  // panel 4: results for the selected method
  const p4 = h('section', { class: `panel${shown ? '' : ' wide'}`, 'aria-labelledby': 'p4h' }, panels);
  function renderResults() {
    const m = current ? ds.methods[current] : null;
    p4.innerHTML = `<div class="panel-head"><h2 id="p4h">4 · Results and mistakes</h2><span class="small">${m ? protocolText(m) : ''}</span></div>`;
    if (!m) { p4.insertAdjacentHTML('beforeend', '<p class="muted">[not run] No method has been computed for this dataset yet.</p>'); return; }
    const mdef = methodDef(current);
    const others = SHAPE_METHODS.filter((x) => x.key !== current && methodAvailable(ds.methods?.[x.key])).map((x) => `${x.name.split(' + ')[0].toLowerCase()} route ${num(fmt(ds.methods[x.key].accuracy), ds, '', protocolOf(ds.methods[x.key]) ? `protocol: ${protocolOf(ds.methods[x.key])}` : '')}`).join(' · ');
    const res = h('div', { class: 'results' }, p4);
    const bits = [];
    if (current === 'pointnet_probe') {
      if (Number.isFinite(m.accuracy_select)) bits.push(`selection split ${num(fmt(m.accuracy_select), ds, '', 'the number the epoch choice saw; not the hold-out')}`);
      if (Number.isFinite(m.untrained)) bits.push(`untrained network ${num(fmt(m.untrained), ds)}`);
      if (Number.isFinite(m.accuracy_cv_val) && Number.isFinite(m.accuracy_sd)) bits.push(`inner cross-validated probe on the held-out part ${num(fmt(m.accuracy_cv_val), ds)} ± ${num(fmt(m.accuracy_sd), ds)}`);
    } else if (Number.isFinite(m.accuracy_sd)) bits.push(`± ${num(fmt(m.accuracy_sd), ds)} across ${Number.isFinite(m.n_folds) ? m.n_folds : (Array.isArray(m.fold_accuracies) ? m.fold_accuracies.length : 'the')} folds`);
    bits.push(`chance ${num(fmt(m.chance), ds)}`);
    if (others) bits.push(others);
    if (current === 'spline_lr' && m.M) bits.push(`M = ${num(m.M, ds)}`);
    if (mdef.baseline) bits.push('a baseline, not a headline');
    res.innerHTML = `<div class="acc"><span class="label">accuracy · ${esc(mdef.name)}</span><strong>${num(fmt(m.accuracy), ds, '', protocolOf(m) ? `protocol: ${protocolOf(m)}` : '')}</strong><span class="sub">${bits.join(' · ')}</span><span class="proto">${protocolText(m)}${sizeInfo(m) ? ` · size ${esc(sizeInfo(m))}` : ''}</span></div>`;
    const rb = ds.roundness_baseline;
    if (rb && Number.isFinite(rb.accuracy)) {
      const acc = res.querySelector('.acc');
      acc.insertAdjacentHTML('beforeend', `<span class="diag-line">${caveatMark('Caveat')} ${esc(rb.label || 'roundness only')}: ${num(fmt(rb.accuracy), ds, '', protocolOf(rb) ? `protocol: ${protocolOf(rb)}` : '')}${protocolOf(rb) ? ` (${esc(protocolOf(rb))})` : ''}; a diagnostic, see the caveat above</span>`);
    }
    let info = null;
    if (Array.isArray(m.cm)) { const ct = confusionTable(ds, m, current); const cw = h('div', { class: 'cm-wrap' }, res); cw.appendChild(ct.table); scrollable(cw, `Confusion matrix, ${mdef.name}`); info = ct; }
    else res.insertAdjacentHTML('beforeend', '<p class="muted small">confusion matrix [not run]</p>');
    if (mdef.baseline) h('p', { class: 'reading', html: `<strong>Size only</strong> is a baseline: the logarithm of each shape's centroid size goes straight into a logistic regression, with no shape at all. It is here so the shape methods, which all see size-normalized shapes, can be read against what size alone would give.` }, p4);
    if (info) h('p', { class: 'reading', html: confusionSentence(ds, m, info) }, p4);
    const worst = Array.isArray(m.worst) ? m.worst : [];
    const wbox = h('div', { style: 'display:flex;flex-direction:column;gap:6px' }, p4);
    const perfect = Array.isArray(m.worst) && m.worst.length === 0 && Array.isArray(m.cm) && m.cm.every((row, i) => row.every((v, j) => i === j || v === 0));
    const label = worst.length
      ? `The ${worst.length === 1 ? 'most confident mistake' : countWord(worst.length) + ' most confident mistakes'} ${scoredWhere(current)}`
      : (perfect ? `No mistakes ${scoredWhere(current)}: there is nothing to show here.` : (mdef.baseline ? 'Worst mistakes are not stored for the size baseline (there is no shape to draw).' : 'Worst mistakes: [not run]'));
    h('span', { class: 'small muted', text: label }, wbox);
    if (worst.length) {
      const g = h('div', { class: 'mistakes' }, wbox);
      const sc = groupScale(worst.map((w) => w.curve || w.shape).filter(Array.isArray), 120, 80, 8);
      worst.forEach((w) => g.appendChild(mistakeFigure(ds, w, sc)));
      if (!shown) h('p', { class: 'small muted', text: 'Shapes are not drawn for this license; the indices refer to the locally fetched data.' }, wbox);
    }
  }
  tabEls.forEach((b) => b.addEventListener('click', () => {
    if (b.disabled) return;
    current = b.dataset.key; state.method.set(id, current);
    tabEls.forEach((x) => x.setAttribute('aria-selected', String(x === b)));
    renderResults();
  }));
  renderResults();

  // CTA → classify
  const canClassify = shown && (classical.engineUsable(ds.engine, ds.n_points).ok || !onnxUnavailableReason(ds));
  if (canClassify) {
    const cta = h('section', { class: 'cta' }, main);
    cta.innerHTML = `<div><strong>Try it on a shape of your own</strong><span>Drag the ${ptsWord}, or load one of the mistakes above, and watch the classifier change its mind.</span></div><a href="#/${esc(id)}/classify">Open the classifier →</a>`;
  }
  main.appendChild(dataLine(ds));
  const pager = h('nav', { class: 'pager', 'aria-label': 'Lab navigation', style: 'margin-top:0' }, main);
  pager.innerHTML = '<a class="prev" href="#/"><small>← Back to</small><span>All datasets</span></a><a class="next" href="../#10"><small>Read →</small><span>Chapter 10: the findings in plain language</span></a>';
  afterRender();
}

/* ======================================================================
   3 · Classify
   ====================================================================== */
/** How many numbers the classical engine block holds (mean, PCA mean, components, scaler, regression). */
function engineSize(e) {
  if (!e || !Array.isArray(e.mean)) return 0;
  const N = e.mean.length, k = Array.isArray(e.components) ? e.components.length : 0, C = Array.isArray(e.lr_coef) ? e.lr_coef.length : 0;
  return 2 * N + (Array.isArray(e.pca_mean) ? e.pca_mean.length : 0) + k * 2 * N + 2 * k + C * k + C;
}
async function renderClassify(id) {
  const reg = registryOf(id);
  setTitle('Classify a shape'); setSub(reg?.title || id); buildChips(id);
  main.innerHTML = '<p class="loading">Loading dataset…</p>';
  await loadIndex();
  let ds = null, err = null;
  try { ds = await loadDataset(id); } catch (e) { err = e; }
  if (state.route !== `cls:${id}`) return;
  buildChips(id);
  main.innerHTML = '';
  const crumbs = h('div', { class: 'crumbs' }, main);
  crumbs.innerHTML = crumbsHtml([['Phenotype Lab', '#/'], [titleHtml(ds?.title || reg?.title || id), `#/${esc(id)}`], ['Classify', null]]);
  if (!ds || ds.status === 'fetch_locally' || ds.shapes_shown === false) {
    main.insertAdjacentHTML('beforeend', '<h1>Classify a shape</h1>');
    main.appendChild(banner('info', !ds ? `This dataset has not been computed (${esc(err?.message || 'no JSON')}).` : 'No shapes or fitted weights are published for this dataset, so there is nothing to classify in the browser. Fetch the data and run the lab locally.'));
    main.appendChild(fetchBox(ds || { id, status: 'proposed', shapes_shown: true }, reg));
    afterRender(); return;
  }
  setSub(ds.title);
  const rep = ds.representation, isCloud = rep === 'pointcloud';
  const ptsWord = isCloud ? 'points' : rep === 'outline' ? 'outline points' : 'landmarks';
  const N = ds.n_points || ds.pca?.mean?.length || ds.neighbours?.shapes?.[0]?.length;
  const pp = ds.methods?.pointnet_probe || {};
  const pointnetSizeRemoved = sizeInfo(pp) === 'removed';

  /* engines available */
  const classicalOk = classical.engineUsable(ds.engine, N);
  const onnxReason = onnxUnavailableReason(ds);
  const engines = {
    classical: { label: 'fitted weights, JS', ok: classicalOk.ok, reason: classicalOk.reason },
    onnx: { label: 'PointNet, ONNX', ok: !onnxReason, reason: onnxReason },
    pyodide: { label: 'shapekit, Pyodide', ok: pyodideOffered(), reason: `desktop only: needs a wide screen and a mouse (${PYODIDE_SIZE_ESTIMATE})` },
  };
  const preferred = state.method.get(id) === 'pointnet_probe' ? 'onnx' : 'classical';
  let engine = engines[preferred].ok ? preferred : (engines.classical.ok ? 'classical' : (engines.onnx.ok ? 'onnx' : null));
  const onnx = engines.onnx.ok ? createOnnxEngine(ds) : null;

  /* header */
  const head = h('div', { class: 'cls-head' }, main);
  const text = h('div', { style: 'display:flex;flex-direction:column;gap:8px;max-width:700px' }, head);
  h('h1', { text: 'Classify a shape' }, text);
  const lede = h('p', { class: 'lede', style: 'font-size:1rem' }, text);
  if (caveatsOf(ds).length) h('p', { class: 'small cls-caveat', html: `${caveatMark('Caveat')} ${esc(caveatShort(id, ds))} <a href="#/${esc(id)}">See the caveat</a>.` }, text);
  const sel = h('div', { class: 'engine-sel', role: 'group', 'aria-label': 'Engine on this device' }, head);
  h('span', { text: 'Engine on this device:' }, sel);
  const pillEls = {};
  for (const [key, e] of Object.entries(engines)) {
    const b = h('button', { type: 'button', class: 'pill-btn', 'aria-pressed': String(engine === key), text: e.label, title: e.ok ? e.label : e.reason }, sel);
    if (!e.ok) b.disabled = true;
    pillEls[key] = b;
  }
  const check = ds.engine ? classical.selfCheck(ds) : null;
  if (check && check.checked > 0 && !check.ok) main.appendChild(banner('warn', `Self-check failed: the JS engine reproduces the script's probabilities only to |Δp| = ${check.maxErr.toExponential(1)} (this page's tolerance is 1e-3, ${check.checked} shapes). Treat the bars below as approximate until <code>scripts/run_lab.py</code> and this page agree.`));

  if (!engine) {
    main.appendChild(banner('info', `No on-page engine is available for this dataset (${esc(engines.classical.reason)}; ${esc(engines.onnx.reason)}).`));
    afterRender(); return;
  }

  /* layout */
  const grid = h('div', { class: 'cls-grid' }, main);
  const stageBox = h('section', { class: 'box cls-stage', 'aria-label': 'Drag the shape' }, grid);
  const side = h('div', { class: 'cls-side' }, grid);
  const loadbar = h('div', { class: 'loadbar' }, stageBox);
  h('span', { class: 'lbl', text: 'Load:' }, loadbar);
  const stage = h('div', { class: 'stage' }, stageBox);
  const W = 600, H = 360;
  const S = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `${N} draggable ${ptsWord} over the faint mean shape` }, stage);
  const defs = svg('defs', {}, S);
  const pat = svg('pattern', { id: 'lab-grid', width: 30, height: 30, patternUnits: 'userSpaceOnUse' }, defs);
  svg('path', { d: 'M30,0 L0,0 L0,30', fill: 'none', stroke: 'var(--code-bg)', 'stroke-width': 1 }, pat);
  svg('rect', { width: W, height: H, fill: 'url(#lab-grid)' }, S);
  const meanShape = ds.pca?.mean || (ds.engine?.mean ? ds.engine.mean.map((p) => [p[0], p[1]]) : null);
  const refShapes = [meanShape, ...(ds.neighbours?.shapes || []).slice(0, 40)].filter(Boolean);
  const SC = groupScale(refShapes, W, H, 48);
  const view = { cx: W / 2, cy: H / 2, s: SC };
  const meanG = svg('g', {}, S);
  if (meanShape) drawShape(meanG, meanShape, view, { rep, color: 'var(--slate)', width: 4, opacity: 0.45, dotR: 2.2 });
  const curPath = svg('path', { fill: 'none', stroke: 'var(--teal)', 'stroke-width': isCloud ? 0 : 5, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, S);
  const handlesG = svg('g', {}, S);
  const hint = svg('text', { x: 24, y: H - 16, style: 'font-size:11px', text: '' }, S);
  const legend = h('div', { class: 'stage-legend' }, stageBox);
  legend.innerHTML = `<span><i style="background:var(--slate);opacity:.6"></i>${meanShape ? 'the mean shape, faint' : 'no mean shape stored'}</span><span><i style="background:var(--teal)"></i>your shape</span><span><i style="background:var(--surface);border:2px solid var(--ink);height:10px;width:10px;border-radius:50%"></i>a ${isCloud ? 'point' : 'landmark'} you can drag</span>`;
  const stageNote = h('p', { class: 'stage-note' }, stageBox);

  /* side boxes */
  const out = h('section', { class: 'box', 'aria-live': 'polite', 'aria-label': 'Classifier output' }, side);
  out.innerHTML = '<span class="kicker">Classifier output</span>';
  const verdict = h('div', { class: 'verdict' }, out);
  const bars = h('div', { class: 'bars' }, out);
  const barEls = ds.classes.map((c) => {
    const row = h('div', { class: 'bar-row' }, bars);
    const nm = h('span', { class: `name ${textClass(c.id)}`, text: c.name }, row);
    nm.style.color = textColor(c.id);
    const track = h('div', { class: 'track' }, row);
    const fill = h('div', { class: `fill ${c.id < PALETTE.length ? 'bg' + c.id : ''}` }, track);
    if (c.id >= PALETTE.length) fill.style.background = stroke(c.id);
    const val = h('span', { class: 'val', text: '—' }, row);
    return { fill, val };
  });
  const truth = h('div', { class: 'truth', hidden: '' }, out);
  const onnxTools = h('div', { class: 'onnx-tools', hidden: '' }, out);
  const shuffleBtn = h('button', { type: 'button', class: 'btn', text: `Shuffle the ${N} points` }, onnxTools);
  const shuffleOut = h('span', {}, onnxTools);
  const engineMsg = h('div', { class: 'banner info', hidden: '' }, out);

  const stepBox = h('section', { class: 'box' }, side);
  stepBox.innerHTML = '<span class="kicker">What happened to your shape</span>';
  const stepper = h('ol', { class: 'stepper' }, stepBox);

  const nnBox = h('section', { class: 'box' }, side);
  nnBox.innerHTML = `<span class="kicker">Nearest training ${isCloud ? 'cells' : 'shapes'}</span>`;
  const nnGrid = h('div', { class: 'nn' }, nnBox);
  const neigh = ds.neighbours?.shapes?.length ? ds.neighbours : null;
  h('span', { class: 'small muted', text: neigh ? `By geodesic distance arccos |⟨z, w⟩| in shape space, over ${thousands(neigh.shapes.length)} stored training shapes, with their true labels.${neigh.note ? ' ' + neigh.note.charAt(0).toUpperCase() + neigh.note.slice(1) + '.' : ''}` : 'No neighbor shapes are stored for this dataset.' }, nnBox);
  const nnScale = neigh ? groupScale(neigh.shapes.slice(0, 60), 120, 80, 8) : 1;

  /* engine notes */
  const notes = h('section', { class: 'engine-notes' }, main);
  const nComp = ds.engine?.n_components;
  const selfCheckText = !check ? '' : (check.checked
    ? ` Self-check: ${check.checked} stored shapes reproduced to |Δp| ≤ ${check.maxErr.toExponential(1)} (this page's tolerance is 1e-3).`
    : (Array.isArray(ds.methods?.[ds.engine?.method]?.worst) && ds.methods[ds.engine.method].worst.length === 0
      ? ' Self-check: no mistakes were stored for this dataset, so there is no stored shape to check against.'
      : ' Self-check: not available (the JSON carries no <code>p_engine</code>).'));
  notes.innerHTML = `
    <div class="note"><strong>Fitted weights, JS</strong> (every device): ${engines.classical.ok
      ? `the classical pipeline is ${thousands(engineSize(ds.engine))} numbers (the mean shape, ${esc(nComp ?? '[not run]')} PCA directions, ${Number.isFinite(nComp) ? nComp + 1 : '[not run]'} weights per class); it runs instantly in plain JavaScript (<code>engines/classical.js</code>).${selfCheckText}`
      : `not available for this dataset: ${esc(engines.classical.reason)}.`}</div>
    <div class="note"><strong>PointNet, ONNX</strong> (every device): the trained network exported to ONNX runs in the browser with ONNX Runtime Web ${esc(ORT_VERSION)}; real inference on a phone, no server. ${pointnetSizeRemoved ? 'Each cloud is scaled to RMS radius 1 before the network sees it, exactly as in training, so size never reaches the network.' : 'A dragged shape has no size, so it is given the dataset’s typical size before the network sees it.'}${engines.onnx.ok ? '' : ` <em>Not offered here: ${esc(engines.onnx.reason)}.</em>`}<span class="onnx-check"></span></div>
    <div class="note"><strong>shapekit, Pyodide</strong> (desktop only): the "Run it yourself" panel loads Pyodide ${esc(PYODIDE_VERSION)} (CPython compiled to WebAssembly) in the browser and re-runs the classical cross-validation with parameters you change. ${esc(PYODIDE_SIZE_ESTIMATE.charAt(0).toUpperCase() + PYODIDE_SIZE_ESTIMATE.slice(1))} on first load, so phones get the two engines above.</div>`;
  let pyPanel = null;
  if (engines.pyodide.ok) { const wrap = h('div', {}, main); state.cleanups.push(mountPyodidePanel(wrap, ds)); pyPanel = wrap.querySelector('details'); }

  /* working shape, in stage pixels */
  let cur = [], loaded = { label: null, note: '', seen: false };
  const toStage = (Z) => Z.map((p) => [view.cx + SC * p[0], view.cy + SC * p[1]]);
  const handles = [];
  const roving = (N || 0) > 20;   // many points: one tab stop, [ and ] move between points
  let focusIdx = 0;
  /** Screen pixels per SVG unit (the stage is scaled to its box); hit circles are 22 screen px. */
  const screenScale = () => { const m = S.getScreenCTM(); return m ? m.a || 1 : 1; };
  const HIT_PX = 22;
  function sizeHits() { const r = HIT_PX / screenScale(); handles.forEach((g) => g.querySelector('.hit').setAttribute('r', r.toFixed(1))); }
  function buildHandles() {
    handlesG.innerHTML = ''; handles.length = 0;
    cur.forEach((p, k) => {
      const g = svg('g', { class: `lm${isCloud ? ' pc' : ''}`, tabindex: roving ? (k === focusIdx ? 0 : -1) : 0, role: 'button', 'aria-label': `${isCloud ? 'point' : 'landmark'} ${k + 1} of ${N}; arrow keys move it${roving ? ', [ and ] pick the previous or next point' : ''}` }, handlesG);
      svg('circle', { class: 'hit', r: 16 }, g);
      svg('circle', { class: 'dot', r: isCloud ? 3.4 : 7 }, g);
      g.addEventListener('keydown', (e) => onKey(e, k));
      g.addEventListener('focus', () => { focusIdx = k; });
      handles.push(g);
    });
    sizeHits();
  }
  function onKey(e, k) {
    if (roving && (e.key === '[' || e.key === ']')) {
      e.preventDefault();
      const next = (k + (e.key === ']' ? 1 : -1) + cur.length) % cur.length;
      handles[k].setAttribute('tabindex', '-1'); handles[next].setAttribute('tabindex', '0'); handles[next].focus();
      return;
    }
    const step = e.shiftKey ? 10 : 3; let dx = 0, dy = 0;
    if (e.key === 'ArrowLeft') dx = -step; else if (e.key === 'ArrowRight') dx = step; else if (e.key === 'ArrowUp') dy = -step; else if (e.key === 'ArrowDown') dy = step; else return;
    e.preventDefault();
    cur[k] = [Math.max(6, Math.min(W - 6, cur[k][0] + dx)), Math.max(6, Math.min(H - 6, cur[k][1] + dy))];
    update();
  }
  /* Dragging is handled on the stage itself: a press picks the nearest point within 22 screen px, so a
     finger does not have to land exactly on a small dot, and a press on the dot of a crowded cloud works too. */
  let dragK = -1;
  const svgPoint = (e) => { const pt = S.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY; const ctm = S.getScreenCTM(); return ctm ? pt.matrixTransform(ctm.inverse()) : null; };
  S.addEventListener('pointerdown', (e) => {
    const p = svgPoint(e); if (!p) return;
    const maxD = HIT_PX / screenScale();
    let best = -1, bestD = Infinity;
    cur.forEach(([x, y], k) => { const d = Math.hypot(x - p.x, y - p.y); if (d < bestD) { bestD = d; best = k; } });
    if (best < 0 || bestD > maxD) return;
    dragK = best; handles[best].classList.add('active');
    try { S.setPointerCapture(e.pointerId); } catch (_) { /* ignore */ }
    e.preventDefault();
  });
  S.addEventListener('pointermove', (e) => {
    if (dragK < 0) return;
    const p = svgPoint(e); if (!p) return;
    cur[dragK] = [Math.max(6, Math.min(W - 6, p.x)), Math.max(6, Math.min(H - 6, p.y))];
    update();
  });
  const endDrag = (e) => { if (dragK < 0) return; handles[dragK]?.classList.remove('active'); dragK = -1; try { S.releasePointerCapture(e.pointerId); } catch (_) { /* ignore */ } };
  S.addEventListener('pointerup', endDrag); S.addEventListener('pointercancel', endDrag);
  if ('ResizeObserver' in window) { const ro = new ResizeObserver(sizeHits); ro.observe(stage); state.cleanups.push(() => ro.disconnect()); }
  function drawCurrent() {
    curPath.setAttribute('d', isCloud ? '' : pathOf(cur, { cx: 0, cy: 0, s: 1 }, rep === 'outline'));
    handles.forEach((g, k) => g.setAttribute('transform', `translate(${cur[k][0].toFixed(1)} ${cur[k][1].toFixed(1)})`));
  }
  function setShape(Z, meta) {
    cur = toStage(Z); loaded = meta;
    if (handles.length !== cur.length) buildHandles();
    update();
  }

  /* outputs */
  function renderProbs(probs, extra) {
    const best = probs.indexOf(Math.max(...probs));
    const c = ds.classes[best];
    verdict.innerHTML = `<strong class="cls ${textClass(best)}" style="color:${textColor(best)}">${esc(c.name)}</strong><span class="with">with probability</span><strong class="p">${fmt(probs[best], 2)}</strong>${extra ? `<span class="with">${extra}</span>` : ''}`;
    barEls.forEach((b, i) => { b.fill.style.width = `${(100 * probs[i]).toFixed(1)}%`; b.val.textContent = fmt(probs[i], 2); });
    if (loaded.label !== null && loaded.label !== undefined) {
      const t = ds.classes[loaded.label];
      truth.hidden = false;
      const seen = loaded.seen && engine === 'classical' && best === loaded.label ? ' The engine on this page was fitted on all shapes, so it has seen this one; that is why it gets it right now.' : '';
      truth.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 9v4"/><path d="M12 17h.01"/><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/></svg><span>Ground truth: <strong ${textAttr(loaded.label)}>${esc(t?.name ?? loaded.label)}</strong>${loaded.note ? '. ' + loaded.note : ''}${seen}${best !== loaded.label ? ' The classifier is wrong here; drag a few points and watch the bars move.' : ''}</span>`;
    } else truth.hidden = true;
  }
  function stepsClassical(r) {
    const top = r.scores.map((s, i) => [Math.abs(s), i, s]).sort((a, b) => b[0] - a[0]).slice(0, 2);
    const logitTxt = r.logits.length === 1 ? `logit ${fmt(r.logits[0], 2)} → p(${esc(ds.classes[1]?.name)}) = ${fmt(r.probs[1], 3)}` : `logits ${r.logits.map((l) => fmt(l, 2)).join(', ')} → softmax`;
    return [
      ['Centroid moved to the origin', `centroid (${fmt(r.centroid[0], 1)}, ${fmt(r.centroid[1], 1)}) px subtracted`, 'ch. 02'],
      ['Size scaled to 1', `‖X − x̄‖_F = ${fmt(r.size, 1)} px → 1`, 'ch. 02'],
      ['Rotated onto the mean shape, one complex formula', `θ = angle Σ conj(z)·m = ${fmt(r.thetaDeg, 1)}°`, 'ch. 03'],
      [`Flattened with the log map, ${r.nComponents} tangent scores`, `ρ = ${fmt(r.rho, 3)}; largest scores: s${top[0][1] + 1} = ${fmt(top[0][2], 3)}, s${top[1] ? top[1][1] + 1 : '?'} = ${top[1] ? fmt(top[1][2], 3) : '—'}`, 'ch. 06'],
      ['Logistic regression: weighted sum, then an S-curve', logitTxt, 'ch. 08'],
    ];
  }
  function stepsOnnx(r) {
    const logitTxt = r.logits.length === 1 ? `logit ${fmt(r.logits[0], 2)} → p(${esc(ds.classes[1]?.name)}) = ${fmt(r.probs[1], 3)}` : `logits ${r.logits.map((l) => fmt(l, 2)).join(', ')} → softmax`;
    const pooled = Number.isFinite(pp.pooled_dim) ? `${pp.pooled_dim} pooled features` : 'one max-pooled feature vector';
    return [
      ['Centroid moved to the origin', `centroid (${fmt(r.centroid[0], 1)}, ${fmt(r.centroid[1], 1)}) px subtracted`, 'ch. 02'],
      pointnetSizeRemoved
        ? ['Size removed: scaled to RMS radius 1, as in training', `‖X − x̄‖_F = ${fmt(r.size, 1)} px → RMS radius ${fmt(r.rms, 2)}`, 'ch. 02']
        : ['Given the typical size, then divided by input_scale', `input_scale = ${fmt(r.inputScale, 2)} px → RMS radius ${fmt(r.rms, 2)}`, 'ch. 08'],
      ['Shared per-point MLP, then max-pool (order ignored)', `${N} points → ${pooled}`, 'ch. 08'],
      ['Embedding', `${r.embedDim} numbers, ‖e‖ = ${fmt(r.embedNorm, 2)} · ${fmt(r.ms, 1)} ms on this device`, 'ch. 08'],
      ['Linear probe: standardize, weighted sum, S-curve', logitTxt, 'ch. 08'],
    ];
  }
  function renderSteps(rows) {
    stepper.innerHTML = rows.map(([what, read, ch], i) => `<li class="${i === rows.length - 1 ? 'last' : ''}"><span class="n">${i + 1}</span><span class="what"><span>${what}</span><span class="read">${read}</span></span><span class="ch">${ch}</span></li>`).join('');
  }
  function renderNeighbours(Z) {
    if (!neigh) return;
    const d = neigh.shapes.map((w, i) => [classical.geodesicDistance(Z, w), i]).sort((a, b) => a[0] - b[0]).slice(0, 3);
    nnGrid.innerHTML = '';
    d.forEach(([dist, i]) => {
      const lab = neigh.labels[i], idx = neigh.indices?.[i] ?? i;
      const fig = h('figure', {}, nnGrid);
      fig.appendChild(thumb(neigh.shapes[i], { rep, color: stroke(lab), scale: nnScale, label: `training shape ${idx}, ${ds.classes[lab]?.name}` }));
      fig.insertAdjacentHTML('beforeend', `<figcaption>#${esc(idx)} · <span ${textAttr(lab)}>${esc(ds.classes[lab]?.name ?? lab)}</span><br>d = ${fmt(dist, 3)}</figcaption>`);
    });
  }

  let raf = null, onnxPending = false;
  function update() {
    drawCurrent();
    if (raf) return;
    raf = requestAnimationFrame(() => { raf = null; compute(); });
  }
  function compute() {
    const Z = classical.preshape(cur).Z;
    renderNeighbours(Z);
    if (engine === 'classical') {
      const r = classical.classify(cur, ds.engine);
      renderProbs(r.probs);
      renderSteps(stepsClassical(r));
    } else if (engine === 'onnx' && onnx) {
      if (onnxPending) return;
      onnxPending = true;
      onnx.run(cur).then((r) => {
        onnxPending = false;
        if (engine !== 'onnx') return;
        engineMsg.hidden = true;
        renderProbs(r.probs, `in ${fmt(r.ms, 1)} ms on this device`);
        renderSteps(stepsOnnx(r));
      }).catch((e) => {
        onnxPending = false;
        engineMsg.hidden = false; engineMsg.className = 'banner error';
        engineMsg.innerHTML = `<span>PointNet could not run here: ${esc(e.message || e)}</span>`;
        verdict.innerHTML = '<span class="with">[not run]</span>';
        barEls.forEach((b) => { b.fill.style.width = '0%'; b.val.textContent = '—'; });
        renderSteps([['Centroid moved to the origin', '', 'ch. 02'], ['Scaled to RMS radius 1', '', 'ch. 02'], ['Shared MLP + max-pool', '[not run]', 'ch. 08'], ['Embedding', '[not run]', 'ch. 08'], ['Linear probe', '[not run]', 'ch. 08']]);
      });
    }
  }
  function setEngine(key) {
    if (!engines[key]?.ok) return;
    if (key === 'pyodide') { if (pyPanel) { pyPanel.open = true; pyPanel.scrollIntoView({ behavior: 'smooth', block: 'start' }); } return; }
    engine = key;
    Object.entries(pillEls).forEach(([k, b]) => b.setAttribute('aria-pressed', String(k === key)));
    onnxTools.hidden = key !== 'onnx';
    engineMsg.hidden = true;
    lede.innerHTML = key === 'classical'
      ? `Drag any ${isCloud ? 'point' : 'landmark'}. The shape is centered, scaled and rotated onto the mean, flattened to ${esc(nComp ?? '[not run]')} tangent scores, and passed through the fitted logistic regression, in the browser, as you drag.`
      : `Drag any ${isCloud ? 'point' : 'landmark'}. The ${N} points are centered, scaled, and pushed through the trained PointNet (ONNX Runtime Web, on this device); the embedding goes to the fitted linear probe. Point order does not matter: try the shuffle button.`;
    if (key === 'onnx' && onnx && !onnx.ready) {
      engineMsg.hidden = false; engineMsg.className = 'banner info'; engineMsg.innerHTML = '<span>Loading ONNX Runtime Web and the model…</span>';
      onnx.load().then(() => { engineMsg.hidden = true; compute(); runOnnxCheck(); }).catch((e) => { engineMsg.className = 'banner error'; engineMsg.innerHTML = `<span>PointNet could not start: ${esc(e.message || e)}</span>`; });
    } else { compute(); if (key === 'onnx') runOnnxCheck(); }
  }
  /* Self-check of the ONNX route against the script's `p_engine` for the stored PointNet mistakes, once per visit. */
  let onnxChecked = false;
  async function runOnnxCheck() {
    if (onnxChecked || !onnx) return;
    onnxChecked = true;
    const el = notes.querySelector('.onnx-check');
    try {
      const r = await onnx.selfCheck();
      if (!r.checked) { el.textContent = ' Self-check: the JSON stores no PointNet mistake with p_engine to check against.'; return; }
      el.textContent = ` Self-check: ${r.checked} stored shapes reproduced to |Δp| ≤ ${r.maxErr.toExponential(1)} (this page's tolerance is 1e-3).`;
      if (!r.ok) main.insertBefore(banner('warn', `Self-check failed for the ONNX route: the browser reproduces the script's PointNet probabilities only to |Δp| = ${r.maxErr.toExponential(1)} (this page's tolerance is 1e-3, ${r.checked} shapes). Treat the PointNet bars as approximate.`), grid);
    } catch (e) { el.textContent = ` Self-check could not run: ${esc(e.message || e)}.`; }
  }
  Object.entries(pillEls).forEach(([k, b]) => b.addEventListener('click', () => setEngine(k)));
  const SHUFFLE_TOL = 1e-4;   // this page's definition of "no change" in float32 arithmetic
  shuffleBtn.addEventListener('click', async () => {
    if (!onnx) return;
    shuffleBtn.disabled = true; shuffleOut.textContent = 'running…';
    try {
      const r = await onnx.shuffleTest(cur);
      shuffleOut.innerHTML = `Embedding after shuffling the point order: max |Δ| = <span class="mono">${r.maxDiff.toExponential(2)}</span>, ${r.maxDiff < SHUFFLE_TOL ? `below the ${SHUFFLE_TOL.toExponential(0)} this page treats as zero (float32 arithmetic): order did not matter.` : `above the ${SHUFFLE_TOL.toExponential(0)} this page treats as zero, so order mattered slightly here.`}`;
    } catch (e) { shuffleOut.textContent = `could not run: ${e.message || e}`; }
    shuffleBtn.disabled = false;
  });

  /* load buttons */
  const mkLoad = (label, Z, meta, primary = false) => {
    const b = h('button', { type: 'button', class: 'chip sm', text: label, 'aria-pressed': String(primary) }, loadbar);
    b.addEventListener('click', () => { loadbar.querySelectorAll('.chip').forEach((x) => x.setAttribute('aria-pressed', 'false')); b.setAttribute('aria-pressed', 'true'); setShape(Z, meta); });
    return b;
  };
  const items = [];
  if (meanShape) items.push(['mean shape', meanShape, { label: null, note: '', seen: false }]);
  const methodKey = engine === 'onnx' ? 'pointnet_probe' : (ds.engine?.method || 'landmarks_lr');
  const worstKey = Array.isArray(ds.methods?.[methodKey]?.worst) ? methodKey : 'landmarks_lr';
  const worst = (ds.methods?.[worstKey]?.worst || []).filter((w) => Array.isArray(w.shape));
  const mistakeNote = (w, first) => `${first ? "This is the dataset's most confident mistake" : 'A confident mistake'} (called ${ds.classes[w.pred]?.name} with p = ${fmt(w.p, 2)} ${scoredWhere(worstKey)}).`;
  worst.forEach((w, i) => items.push([`mistake #${w.index}`, w.shape, { label: w.true, note: mistakeNote(w, i === 0), seen: worstKey !== 'pointnet_probe' }]));
  if (neigh) ds.classes.forEach((c) => {
    const pool = neigh.labels.map((l, i) => (l === c.id ? i : -1)).filter((i) => i >= 0);
    if (!pool.length) return;
    const b = h('button', { type: 'button', class: 'chip sm', text: `random ${c.name}`, 'aria-pressed': 'false' }, loadbar);
    b.addEventListener('click', () => {
      const i = pool[Math.floor(Math.random() * pool.length)];
      loadbar.querySelectorAll('.chip').forEach((x) => x.setAttribute('aria-pressed', 'false')); b.setAttribute('aria-pressed', 'true');
      setShape(neigh.shapes[i], { label: c.id, note: `Training shape #${neigh.indices?.[i] ?? i}.`, seen: true });
    });
  });
  items.forEach(([label, Z, meta], i) => mkLoad(label, Z, meta, i === (worst.length ? 1 : 0)));
  h('span', { class: 'spacer' }, loadbar);
  const initial = worst[0] ? [worst[0].shape, { label: worst[0].true, note: mistakeNote(worst[0], true), seen: worstKey !== 'pointnet_probe' }] : [meanShape, { label: null, note: '', seen: false }];
  const reset = h('button', { type: 'button', class: 'chip sm', text: 'Reset' }, loadbar);
  reset.addEventListener('click', () => { loadbar.querySelectorAll('.chip').forEach((x) => x.setAttribute('aria-pressed', 'false')); setShape(initial[0], initial[1]); });
  stageNote.textContent = `Drag with a mouse or finger (a press picks the nearest ${isCloud ? 'point' : 'landmark'}); or focus a ${isCloud ? 'point' : 'landmark'} and use the arrow keys (shift = bigger steps${roving ? '; [ and ] pick the previous or next point' : ''}). Everything recomputes as you drag.`;
  hint.textContent = isCloud ? 'Moving the whole cloud, or making it bigger, changes nothing: position and size are removed first.' : 'Moving the whole shape, or making it bigger, changes nothing: position and size are removed first.';

  if (!initial[0]) { main.appendChild(banner('info', 'The JSON carries no shapes to start from.')); afterRender(); return; }
  setShape(initial[0], initial[1]);
  setEngine(engine);
  afterRender();
}

/* ---------- router ---------- */
function route() {
  cleanup();
  const raw = decodeURIComponent(location.hash.replace(/^#\/?/, '')).replace(/\/+$/, '');
  const [id, sub] = raw.split('/');
  if (!id) { state.route = 'home'; $('#lab-nav-lab').setAttribute('aria-current', 'page'); renderHome(); return; }
  if (!/^[a-z0-9_-]+$/i.test(id)) { location.hash = '#/'; return; }
  if (sub === 'classify') { state.route = `cls:${id}`; renderClassify(id); return; }
  state.route = `ds:${id}`; renderDataset(id);
}
window.addEventListener('hashchange', route);
route();
