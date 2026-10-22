/* Demo "contrastive": a toy 2-D embedding with two classes. "Step" applies one
   supervised-contrastive update: each point is pulled toward its positives (same class)
   and pushed away from a softmax-weighted average of everyone else; the temperature τ
   sets how sharply that softmax concentrates on the nearest neighbours.
   Similarity here is s_ij = −2·‖z_i − z_j‖² (a Gaussian-kernel stand-in for the cosine of unit vectors). */
import { frame, svg, slider, button, checkbox, readoutRow, fmt } from './_lib.js';

const N_PER_CLASS = 8;
const VIEW = { cx: 300, cy: 160, s: 130 };     // unit square [−1, 1]² → stage
const LR = 0.22;                              // fixed step length so every τ is watchable

function rng(seed) { let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
function initialPoints() {
  const r = rng(7); const pts = [];
  for (let i = 0; i < 2 * N_PER_CLASS; i++) {
    const cls = i < N_PER_CLASS ? 0 : 1;
    // both classes start in one mixed blob, with a tiny class offset so there is something to learn
    pts.push({ x: (r() * 2 - 1) * 0.7 + (cls ? 0.08 : -0.08), y: (r() * 2 - 1) * 0.7, cls });
  }
  return pts;
}
const sim = (a, b) => -2 * ((a.x - b.x) ** 2 + (a.y - b.y) ** 2);

/** SupCon loss (Khosla et al. 2020) averaged over anchors, plus the per-point update direction. */
function supcon(pts, tau) {
  let loss = 0; const forces = pts.map(() => ({ x: 0, y: 0 }));
  pts.forEach((zi, i) => {
    const others = pts.map((_, a) => a).filter((a) => a !== i);
    const logits = others.map((a) => sim(zi, pts[a]) / tau);
    const mx = Math.max(...logits);
    const den = logits.reduce((s, l) => s + Math.exp(l - mx), 0);
    const q = logits.map((l) => Math.exp(l - mx) / den);                    // softmax over all others
    const pos = others.filter((a) => pts[a].cls === zi.cls);
    if (!pos.length) return;
    let li = 0;
    pos.forEach((p) => { const idx = others.indexOf(p); li += -(logits[idx] - mx - Math.log(den)); });
    loss += li / pos.length;
    // −∂L/∂z_i ∝ (1/|P|) Σ_p (z_p − z_i)  −  Σ_a q_a (z_a − z_i)
    let px = 0, py = 0; pos.forEach((p) => { px += pts[p].x - zi.x; py += pts[p].y - zi.y; }); px /= pos.length; py /= pos.length;
    let nx = 0, ny = 0; others.forEach((a, j) => { nx += q[j] * (pts[a].x - zi.x); ny += q[j] * (pts[a].y - zi.y); });
    forces[i] = { x: px - nx, y: py - ny };
  });
  return { loss: loss / pts.length, forces };
}

export default function mount(el, ctx) {
  const ui = frame(el, {
    title: 'Pull together, push apart',
    hint: 'Sixteen cells embedded in 2-D, two classes. Press Step to apply one supervised-contrastive update; lower the temperature to make each point care only about its nearest neighbours.',
  });
  const S = ui.svg;
  let pts = initialPoints(); let tau = 0.3; let steps = 0; let showForces = true;
  const X = (p) => VIEW.cx + VIEW.s * p.x, Y = (p) => VIEW.cy + VIEW.s * p.y;

  svg('rect', { x: VIEW.cx - 150, y: VIEW.cy - 150, width: 300, height: 300, rx: 10, fill: 'var(--paper)', stroke: 'var(--line)' }, S);
  svg('text', { x: 20, y: 28, text: 'embedding space (toy, 2-D)' }, S);
  const arrows = svg('g', { stroke: 'var(--slate)', 'stroke-width': 1.5, fill: 'none', opacity: 0.7 }, S);
  const dots = pts.map((p) => svg('circle', { cx: X(p), cy: Y(p), r: 7, fill: p.cls ? 'var(--teal)' : 'var(--coral)', stroke: 'var(--surface)', 'stroke-width': 1.5 }, S));
  const arrowEls = pts.map(() => svg('line', {}, arrows));
  // a small softmax bar chart for the first anchor, to make τ visible
  const BAR = { x: 470, y: 70, w: 110, h: 170 };
  svg('text', { x: BAR.x + BAR.w / 2, y: BAR.y - 24, text: 'softmax weights', 'text-anchor': 'middle' }, S);
  svg('text', { x: BAR.x + BAR.w / 2, y: BAR.y - 10, text: 'seen by one anchor', 'text-anchor': 'middle', 'font-size': 11 }, S);
  const bars = pts.slice(1).map((p, j) => svg('rect', { x: BAR.x + (j * BAR.w) / 15, width: BAR.w / 15 - 1.5, y: BAR.y + BAR.h, height: 0, fill: p.cls ? 'var(--teal)' : 'var(--coral)', opacity: 0.85 }, S));
  svg('line', { x1: BAR.x, x2: BAR.x + BAR.w, y1: BAR.y + BAR.h, y2: BAR.y + BAR.h, stroke: 'var(--line)' }, S);
  const anchorRing = svg('circle', { r: 11, fill: 'none', stroke: 'var(--gold)', 'stroke-width': 2 }, S);

  slider(ui.controls, { label: 'temperature τ', min: 0.05, max: 1, step: 0.01, value: tau, onInput: (v) => { tau = v; update(); } });
  button(ui.controls, 'Step', () => step(1), true);
  button(ui.controls, 'Step ×10', () => step(10));
  button(ui.controls, 'Reset', () => { cancel(); pts = initialPoints(); steps = 0; update(); });
  checkbox(ui.controls, 'show update arrows', true, (v) => { showForces = v; update(); });
  ui.legend([['var(--coral)', 'class 0'], ['var(--teal)', 'class 1'], ['var(--slate)', 'direction of the next update'], ['var(--gold)', 'the anchor whose softmax is plotted']]);

  let raf = null, queue = 0;
  function cancel() { cancelAnimationFrame(raf); queue = 0; }
  function step(n) {
    queue += n; if (raf) return; runOne();
  }
  function runOne() {
    if (queue <= 0) { raf = null; return; }
    queue--;
    const { forces } = supcon(pts, tau);
    const from = pts.map((p) => ({ ...p }));
    const to = pts.map((p, i) => ({ ...p, x: clamp(p.x + LR * forces[i].x), y: clamp(p.y + LR * forces[i].y) }));
    steps++;
    if (ctx.reducedMotion) { pts = to; update(); raf = requestAnimationFrame(() => runOne()); return; }
    const t0 = performance.now(), dur = 420;
    const tick = (now) => {
      const u = Math.min(1, (now - t0) / dur); const e = 1 - Math.pow(1 - u, 3);
      pts = from.map((p, i) => ({ ...p, x: p.x + (to[i].x - p.x) * e, y: p.y + (to[i].y - p.y) * e }));
      update(u >= 1);
      raf = u < 1 ? requestAnimationFrame(tick) : requestAnimationFrame(() => runOne());
    };
    raf = requestAnimationFrame(tick);
  }
  const clamp = (v) => Math.max(-1.1, Math.min(1.1, v));

  function update(settled = true) {
    const { loss, forces } = supcon(pts, tau);
    pts.forEach((p, i) => { dots[i].setAttribute('cx', X(p).toFixed(1)); dots[i].setAttribute('cy', Y(p).toFixed(1)); });
    arrows.setAttribute('opacity', showForces && settled ? 0.7 : 0);
    pts.forEach((p, i) => {
      const f = forces[i]; const L = Math.hypot(f.x, f.y); const k = L > 1e-9 ? Math.min(40, VIEW.s * LR * L) / L : 0;
      arrowEls[i].setAttribute('x1', X(p).toFixed(1)); arrowEls[i].setAttribute('y1', Y(p).toFixed(1));
      arrowEls[i].setAttribute('x2', (X(p) + k * f.x).toFixed(1)); arrowEls[i].setAttribute('y2', (Y(p) + k * f.y).toFixed(1));
    });
    // softmax of anchor 0
    const a0 = pts[0]; anchorRing.setAttribute('cx', X(a0).toFixed(1)); anchorRing.setAttribute('cy', Y(a0).toFixed(1));
    const logits = pts.slice(1).map((p) => sim(a0, p) / tau); const mx = Math.max(...logits);
    const den = logits.reduce((s, l) => s + Math.exp(l - mx), 0);
    logits.forEach((l, j) => { const q = Math.exp(l - mx) / den; bars[j].setAttribute('y', (BAR.y + BAR.h - q * BAR.h).toFixed(1)); bars[j].setAttribute('height', (q * BAR.h).toFixed(1)); });
    readoutRow(ui.readout, [['τ =', fmt(tau, 2)], ['SupCon loss =', fmt(loss, 3)], ['steps =', String(steps)], ['identical points would give', fmt(Math.log(2 * N_PER_CLASS - 1), 3)]]);
  }
  update();
  return () => cancel();
}
