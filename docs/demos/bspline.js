/* Demo "bspline": drag the control points of a cubic B-spline curve
   curve(t) = Σ_k c_k · β(t − k), with the cubic bump
     β(x) = 2/3 − x² + |x|³/2  for |x| < 1,   (2 − |x|)³ / 6  for 1 ≤ |x| ≤ 2,   0 otherwise.
   Toggle the basis functions to see which bumps are active; dragging highlights the local segment. */
import { frame, svg, button, checkbox, readoutRow, draggable, fmt, sampleSpline } from './_lib.js';

export function bump(x) {
  const a = Math.abs(x);
  if (a < 1) return 2 / 3 - a * a + (a * a * a) / 2;
  if (a <= 2) return ((2 - a) ** 3) / 6;
  return 0;
}
/** Control point k with end handling: wrap-around (closed) or mirrored ends (open),
   matching shapekit's export: c[-1] = c[1], c[M] = c[M-2]. */
function coef(c, k, closed) {
  const M = c.length;
  if (closed) return c[((k % M) + M) % M];
  if (k < 0) return c[-k];
  if (k >= M) return c[2 * M - 2 - k];
  return c[k];
}
export function evalCurve(c, t, closed) {
  const k0 = Math.floor(t); let x = 0, y = 0;
  for (let k = k0 - 1; k <= k0 + 2; k++) { const b = bump(t - k); if (b) { const p = coef(c, k, closed); x += b * p[0]; y += b * p[1]; } }
  return [x, y];
}

export default async function mount(el, ctx) {
  let data, sample = false;
  try {
    const d = await ctx.loadJSON('data/spline.json');
    const control = d.control_points || d.control || d.coefficients || d.c;
    const landmarks = d.landmarks || d.points || [];
    if (!Array.isArray(control) || control.length < 4) throw new Error('spline.json has no control points');
    data = { control, landmarks, closed: !!d.closed, reference: Array.isArray(d.curve) ? d.curve : null };
  } catch (e) { data = sampleSpline(); sample = true; }

  // fit everything into the stage with a margin
  const all = data.control.concat(data.landmarks);
  const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const sc = Math.min(520 / ((maxX - minX) || 1), 190 / ((maxY - minY) || 1));
  const map = (p) => [40 + (p[0] - minX) * sc + (520 - (maxX - minX) * sc) / 2, 40 + (p[1] - minY) * sc + (190 - (maxY - minY) * sc) / 2];
  const initial = data.control.map(map);
  const landmarks = data.landmarks.map(map);
  let c = initial.map((p) => [...p]);
  const M = c.length, closed = data.closed;
  const tMax = closed ? M : M - 1;

  const ui = frame(el, {
    title: 'A curve made of bumps',
    hint: 'Drag a control point: only the nearby part of the curve moves, because each bump is just 4 units wide. Tick the box to see the bumps themselves.',
    viewBox: '0 0 600 340', sample,
  });
  const S = ui.svg;
  const polygon = svg('path', { fill: 'none', stroke: 'var(--slate)', 'stroke-width': 1.5, 'stroke-dasharray': '5 4', opacity: 0.6 }, S);
  if (landmarks.length) landmarks.forEach((p) => svg('circle', { cx: p[0], cy: p[1], r: 3, fill: 'var(--slate)', opacity: 0.5 }, S));
  const curve = svg('path', { fill: 'none', stroke: 'var(--ink)', 'stroke-width': 3, 'stroke-linecap': 'round' }, S);
  const local = svg('path', { fill: 'none', stroke: 'var(--gold)', 'stroke-width': 5, 'stroke-linecap': 'round', opacity: 0 }, S);
  // basis strip (hidden until toggled)
  const strip = svg('g', { opacity: 0 }, S);
  const BX = { x: 40, y: 250, w: 520, h: 70 };
  svg('rect', { x: BX.x, y: BX.y, width: BX.w, height: BX.h, rx: 6, fill: 'var(--paper)', stroke: 'var(--line)' }, strip);
  svg('text', { x: BX.x + 8, y: BX.y + 14, text: 'basis functions β(t − k); they always add up to 1', 'font-size': 11 }, strip);
  const tx = (t) => BX.x + (t / tMax) * BX.w, ty = (b) => BX.y + BX.h - 6 - b * (BX.h - 24);
  const bumps = [];
  const bumpColors = ['var(--teal)', 'var(--lavender)', 'var(--coral)', 'var(--gold)', 'var(--slate)', 'var(--muted)'];
  for (let k = 0; k < M; k++) {
    let d = '';
    for (let i = 0; i <= 80; i++) { const t = (tMax * i) / 80; let b = bump(t - k); if (closed) b += bump(t - k - M) + bump(t - k + M); d += `${i ? 'L' : 'M'}${tx(t).toFixed(1)} ${ty(b).toFixed(1)}`; }
    bumps.push(svg('path', { d, fill: 'none', stroke: bumpColors[k % bumpColors.length], 'stroke-width': 1.5, opacity: 0.8 }, strip));
  }
  svg('line', { x1: tx(0), x2: tx(tMax), y1: ty(1), y2: ty(1), stroke: 'var(--muted)', 'stroke-dasharray': '3 3' }, strip);
  const marker = svg('line', { y1: BX.y, y2: BX.y + BX.h, stroke: 'var(--gold)', 'stroke-width': 1.5, opacity: 0 }, strip);

  let active = -1;
  const handles = c.map((p, k) => {
    const g = svg('g', {}, S);
    const circle = svg('circle', { cx: p[0], cy: p[1], r: 8, fill: 'var(--teal)', stroke: 'var(--surface)', 'stroke-width': 2, 'aria-label': `control point ${k}` }, g);
    svg('text', { x: p[0] + 12, y: p[1] - 10, text: `c${k}`, class: 'mono', 'font-size': 11 }, g);
    draggable(S, circle, (x, y) => {
      c[k] = [Math.min(590, Math.max(10, x)), Math.min(235, Math.max(10, y))];
      circle.setAttribute('cx', c[k][0]); circle.setAttribute('cy', c[k][1]);
      g.lastChild.setAttribute('x', c[k][0] + 12); g.lastChild.setAttribute('y', c[k][1] - 10);
      active = k; update();
    }, () => { active = -1; update(); });
    return circle;
  });

  let showBasis = false;
  checkbox(ui.controls, 'show basis functions', false, (v) => { showBasis = v; strip.setAttribute('opacity', v ? 1 : 0); update(); });
  button(ui.controls, 'Reset', () => { c = initial.map((p) => [...p]); handles.forEach((hd, k) => { hd.setAttribute('cx', c[k][0]); hd.setAttribute('cy', c[k][1]); hd.parentNode.lastChild.setAttribute('x', c[k][0] + 12); hd.parentNode.lastChild.setAttribute('y', c[k][1] - 10); }); update(); });
  ui.legend([['var(--teal)', 'control points (drag)'], ['var(--slate)', 'control polygon' + (landmarks.length ? ' · landmarks' : '')], ['var(--ink)', 'B-spline curve'], ['var(--gold)', 'segment the dragged point can reach']]);

  function curvePath(t0, t1) {
    const n = Math.max(2, Math.round((t1 - t0) * 40)); let d = '';
    for (let i = 0; i <= n; i++) { const [x, y] = evalCurve(c, t0 + ((t1 - t0) * i) / n, closed); d += `${i ? 'L' : 'M'}${x.toFixed(1)} ${y.toFixed(1)}`; }
    return d;
  }
  function update() {
    polygon.setAttribute('d', c.map((p, i) => `${i ? 'L' : 'M'}${p[0]} ${p[1]}`).join(' ') + (closed ? ' Z' : ''));
    curve.setAttribute('d', curvePath(0, tMax) + (closed ? ' Z' : ''));
    if (active >= 0) {
      const t0 = Math.max(0, active - 2), t1 = Math.min(tMax, active + 2);
      local.setAttribute('d', curvePath(t0, t1)); local.setAttribute('opacity', 0.75);
      marker.setAttribute('x1', tx(active)); marker.setAttribute('x2', tx(active)); marker.setAttribute('opacity', 1);
    } else { local.setAttribute('opacity', 0); marker.setAttribute('opacity', 0); }
    bumps.forEach((b, k) => b.setAttribute('stroke-width', k === active ? 3 : 1.5));
    const tProbe = active >= 0 ? active : tMax / 2;
    const wts = [];
    for (let k = Math.floor(tProbe) - 1; k <= Math.floor(tProbe) + 2; k++) { const b = bump(tProbe - k); if (b > 1e-9) wts.push(`β(${fmt(tProbe - k, 1)})=${fmt(b, 3)}`); }
    readoutRow(ui.readout, [
      ['M =', `${M} control points, ${closed ? 'closed' : 'open'} curve, t ∈ [0, ${tMax}]`],
      [`weights at t = ${fmt(tProbe, 1)}:`, wts.join('  ')],
      ['sum of weights =', fmt(wts.reduce((s, w) => s + parseFloat(w.split('=')[1]), 0), 3)],
    ]);
  }
  update();
}
