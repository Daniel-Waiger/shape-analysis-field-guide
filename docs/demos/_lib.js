/* Shared helpers for the interactive demos (vanilla JS + SVG, no frameworks).
   Each demo imports what it needs from here; the maths is deliberately explicit. */

export const NS = 'http://www.w3.org/2000/svg';

/* ---------- element builders ---------- */
export function svg(tag, attrs = {}, parent) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'text') el.textContent = v; else if (v !== null && v !== undefined) el.setAttribute(k, v);
  }
  if (parent) parent.appendChild(el);
  return el;
}
export function h(tag, attrs = {}, parent) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'text') el.textContent = v; else if (k === 'html') el.innerHTML = v; else if (k === 'class') el.className = v;
    else if (v !== null && v !== undefined) el.setAttribute(k, v);
  }
  if (parent) parent.appendChild(el);
  return el;
}
export const fmt = (x, d = 3) => (Number.isFinite(x) ? x.toFixed(d) : '—');
export const deg = (rad) => (rad * 180) / Math.PI;

/* ---------- demo card scaffold ---------- */
export function frame(el, { title, hint, viewBox = '0 0 600 320', ariaLabel = title, sample = false }) {
  el.innerHTML = '';
  const card = h('div', { class: 'demo-card' }, el);
  const head = h('div', { class: 'demo-head' }, card);
  h('h4', { class: 'demo-title', text: title }, head);
  if (sample) h('span', { class: 'demo-badge', text: 'built-in sample shapes', title: 'data/*.json not found; showing illustrative shapes, not results' }, head);
  h('p', { class: 'demo-hint', text: hint }, head);
  const stage = h('div', { class: 'demo-stage' }, card);
  const s = svg('svg', { viewBox, role: 'img', 'aria-label': ariaLabel }, stage);
  const controls = h('div', { class: 'demo-controls' }, card);
  const readout = h('div', { class: 'demo-readout', 'aria-live': 'polite' }, card);
  return { card, svg: s, controls, readout, legend: (items) => legend(card, items) };
}
function legend(card, items) {
  const l = h('div', { class: 'demo-legend' }, card);
  items.forEach(([color, label]) => { const s = h('span', {}, l); h('i', { style: `background:${color}` }, s); s.appendChild(document.createTextNode(label)); });
  return l;
}
export function slider(parent, { label, min, max, step, value, format = (v) => fmt(v, 2), onInput }) {
  const wrap = h('label', { class: 'demo-slider' }, parent);
  h('span', { text: label }, wrap);
  const input = h('input', { type: 'range', min, max, step, value }, wrap);
  const val = h('span', { class: 'val', text: format(value) }, wrap);
  const fire = () => { const v = parseFloat(input.value); val.textContent = format(v); onInput(v); };
  input.addEventListener('input', fire);
  return { input, set(v) { input.value = v; fire(); }, get: () => parseFloat(input.value) };
}
export function button(parent, label, onClick, primary = false) {
  const b = h('button', { type: 'button', class: 'demo-btn' + (primary ? ' primary' : ''), text: label }, parent);
  b.addEventListener('click', onClick);
  return b;
}
export function checkbox(parent, label, checked, onChange) {
  const wrap = h('label', { class: 'demo-check' }, parent);
  const input = h('input', { type: 'checkbox' }, wrap); input.checked = checked;
  wrap.appendChild(document.createTextNode(label));
  input.addEventListener('change', () => onChange(input.checked));
  return input;
}
export function readoutRow(readout, entries) {
  readout.innerHTML = entries.map(([k, v, cls]) => `<span${cls ? ` class="${cls}"` : ''}>${k} <b>${v}</b></span>`).join('');
}

/* ---------- pointer drag (mouse + touch via Pointer Events) ---------- */
export function svgPoint(svgEl, clientX, clientY) {
  const pt = svgEl.createSVGPoint(); pt.x = clientX; pt.y = clientY;
  const ctm = svgEl.getScreenCTM();
  if (!ctm) return { x: clientX, y: clientY };
  const p = pt.matrixTransform(ctm.inverse());
  return { x: p.x, y: p.y };
}
export function draggable(svgEl, handle, onMove, onEnd) {
  handle.classList.add('handle');
  handle.setAttribute('tabindex', '0');
  let active = false;
  handle.addEventListener('pointerdown', (e) => {
    active = true; handle.setPointerCapture(e.pointerId); e.preventDefault();
  });
  handle.addEventListener('pointermove', (e) => {
    if (!active) return;
    const p = svgPoint(svgEl, e.clientX, e.clientY); onMove(p.x, p.y);
  });
  const end = (e) => { if (!active) return; active = false; try { handle.releasePointerCapture(e.pointerId); } catch (_) { /* ignore */ } if (onEnd) onEnd(); };
  handle.addEventListener('pointerup', end);
  handle.addEventListener('pointercancel', end);
  // keyboard nudging for accessibility
  handle.addEventListener('keydown', (e) => {
    const step = e.shiftKey ? 10 : 3; let dx = 0, dy = 0;
    if (e.key === 'ArrowLeft') dx = -step; else if (e.key === 'ArrowRight') dx = step;
    else if (e.key === 'ArrowUp') dy = -step; else if (e.key === 'ArrowDown') dy = step; else return;
    e.preventDefault(); e.stopPropagation();
    const cx = parseFloat(handle.getAttribute('cx')), cy = parseFloat(handle.getAttribute('cy'));
    onMove(cx + dx, cy + dy); if (onEnd) onEnd();
  });
}

/* ---------- shape maths (points are rows: [[x, y], ...]) ---------- */
export function centroid(P) { const n = P.length; let x = 0, y = 0; for (const p of P) { x += p[0]; y += p[1]; } return [x / n, y / n]; }
export function center(P) { const c = centroid(P); return P.map((p) => [p[0] - c[0], p[1] - c[1]]); }
export function frobenius(P) { let s = 0; for (const p of P) s += p[0] * p[0] + p[1] * p[1]; return Math.sqrt(s); }
export function normalize(P) { const n = frobenius(P) || 1; return P.map((p) => [p[0] / n, p[1] / n]); }
export function preshape(P) { return normalize(center(P)); }
/** Σ_k conj(z_k)·w_k for z, w given as [[x,y],...]; returns {re, im}. */
export function hermitian(z, w) {
  let re = 0, im = 0;
  for (let k = 0; k < z.length; k++) { re += z[k][0] * w[k][0] + z[k][1] * w[k][1]; im += z[k][0] * w[k][1] - z[k][1] * w[k][0]; }
  return { re, im };
}
/** Closed-form angle that rotates z onto w: angle(Σ conj(z_k) w_k). */
export function optimalRotationAngle(z, w) { const s = hermitian(z, w); return Math.atan2(s.im, s.re); }
export function rotate(P, theta) {
  const c = Math.cos(theta), s = Math.sin(theta);
  return P.map(([x, y]) => [x * c - y * s, x * s + y * c]);
}
export function sqDist(P, Q) { let s = 0; for (let k = 0; k < P.length; k++) { const dx = P[k][0] - Q[k][0], dy = P[k][1] - Q[k][1]; s += dx * dx + dy * dy; } return s; }
export function lerpShape(P, Q, a, b) { return P.map((p, k) => [a * p[0] + b * Q[k][0], a * p[1] + b * Q[k][1]]); }
export const scaleShape = (P, s) => P.map(([x, y]) => [x * s, y * s]);
export const addShape = (P, Q) => P.map((p, k) => [p[0] + Q[k][0], p[1] + Q[k][1]]);

/** Shape → SVG path "M x y L x y …" through a view {cx, cy, s} (y is image-down, as in SVG). */
export function pathOf(P, view, close = false) {
  const d = P.map((p, i) => `${i ? 'L' : 'M'}${(view.cx + view.s * p[0]).toFixed(2)} ${(view.cy + view.s * p[1]).toFixed(2)}`).join(' ');
  return close ? d + ' Z' : d;
}
export function smoothPathOf(P, view, close = false) {
  // Catmull-Rom → cubic Bézier, purely for a softer-looking polyline
  const pts = P.map((p) => [view.cx + view.s * p[0], view.cy + view.s * p[1]]);
  if (pts.length < 3) return pathOf(P, view, close);
  let d = `M${pts[0][0].toFixed(2)} ${pts[0][1].toFixed(2)}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[i - 1] || pts[i], p1 = pts[i], p2 = pts[i + 1], p3 = pts[i + 2] || p2;
    const c1 = [p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6];
    const c2 = [p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6];
    d += ` C${c1[0].toFixed(2)} ${c1[1].toFixed(2)} ${c2[0].toFixed(2)} ${c2[1].toFixed(2)} ${p2[0].toFixed(2)} ${p2[1].toFixed(2)}`;
  }
  return close ? d + ' Z' : d;
}
export function drawDots(parent, P, view, attrs) {
  return P.map((p) => svg('circle', { cx: (view.cx + view.s * p[0]).toFixed(2), cy: (view.cy + view.s * p[1]).toFixed(2), r: 3.5, ...attrs }, parent));
}

/* ---------- built-in sample data (used when docs/data/*.json is absent) ---------- */
/** A straight 12-point worm and a bent one, in pixel-like image coordinates (y down). */
export function sampleWorms() {
  const straight = [], bent = [];
  for (let i = 0; i < 12; i++) {
    const t = i / 11;
    straight.push([40 + 120 * t, 100 + 6 * Math.sin(t * Math.PI * 2)]);
    const a = -0.9 + 1.8 * t; // arc from -0.9 to 0.9 rad
    bent.push([110 + 95 * Math.sin(a), 150 - 95 * Math.cos(a) + 20]);
  }
  return { straight, bent };
}
/** Six control points and 20 "landmarks" sampled along a gently curved worm. */
export function sampleSpline() {
  const control = [[60, 180], [150, 70], [260, 220], [370, 90], [470, 230], [550, 120]];
  const landmarks = [];
  for (let i = 0; i < 20; i++) { const t = i / 19; landmarks.push([60 + 490 * t, 160 + 60 * Math.sin(t * Math.PI * 2.2)]); }
  return { control, landmarks, closed: false };
}
/** Try worms.json (fetched once per page); fall back to sample preshapes. Returns {mean, dead, live, pca?, sample}. */
let wormsPromise = null;
export async function loadWorms(ctx) {
  try {
    if (!wormsPromise) wormsPromise = ctx.loadJSON('data/worms.json').catch((e) => { wormsPromise = null; throw e; });
    const d = await wormsPromise;
    const dead = d.examples?.dead ?? [], live = d.examples?.live ?? [];
    if (!d.mean || !dead.length || !live.length) throw new Error('worms.json incomplete');
    return { mean: preshape(d.mean), dead: dead.map(preshape), live: live.map(preshape), pca: d.pca || null, classMeans: d.class_means || null, sample: false };
  } catch (e) {
    const { straight, bent } = sampleWorms();
    const dead = [preshape(bent)], live = [preshape(straight)];
    const mean = preshape(lerpShape(live[0], rotate(dead[0], optimalRotationAngle(dead[0], live[0])), 0.5, 0.5));
    return { mean, dead, live, pca: null, classMeans: null, sample: true };
  }
}
