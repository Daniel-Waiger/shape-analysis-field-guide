/* Demo "geodesic": slide t ∈ [0, 1] to morph one preshape into another along the
   great-circle (slerp) path in Kendall shape space; shows geodesic, chord and Procrustes distances. */
import { frame, svg, slider, button, readoutRow, rotate, hermitian, optimalRotationAngle, sqDist, lerpShape, pathOf, drawDots, fmt, loadWorms } from './_lib.js';

export default async function mount(el, ctx) {
  const worms = await loadWorms(ctx);
  const z = worms.dead[0];                                           // class 0, coral
  const w = rotate(worms.live[0], optimalRotationAngle(worms.live[0], z)); // class 1, teal, aligned to z
  const inner = hermitian(z, w);
  const absInner = Math.min(1, Math.hypot(inner.re, inner.im));      // |zᴴw|
  const rho = Math.acos(absInner);                                   // geodesic distance
  const chord = Math.sqrt(Math.max(0, 2 - 2 * absInner));            // chord distance
  const procrustes = Math.sqrt(sqDist(z, w));                        // ‖z − e^{iθ*}w‖ (equals the chord)

  const ui = frame(el, {
    title: 'Walking the geodesic',
    hint: 'Slide t from 0 to 1. The shape follows the shortest arc on the preshape sphere from the coral worm to the teal one; the dot on the right follows the same arc.',
    sample: worms.sample,
  });
  const S = ui.svg;
  const SV = { cx: 175, cy: 165, s: 250 };

  svg('text', { x: SV.cx, y: 28, text: 'path(t) between two preshapes', 'text-anchor': 'middle' }, S);
  svg('path', { d: pathOf(z, SV), fill: 'none', stroke: 'var(--coral)', 'stroke-width': 2, opacity: 0.45 }, S);
  svg('path', { d: pathOf(w, SV), fill: 'none', stroke: 'var(--teal)', 'stroke-width': 2, opacity: 0.45 }, S);
  drawDots(S, z, SV, { fill: 'var(--coral)', opacity: 0.45, r: 3 });
  drawDots(S, w, SV, { fill: 'var(--teal)', opacity: 0.45, r: 3 });
  const curPath = svg('path', { fill: 'none', stroke: 'var(--ink)', 'stroke-width': 3, 'stroke-linejoin': 'round' }, S);
  const curDots = svg('g', { fill: 'var(--ink)' }, S);
  for (let k = 0; k < z.length; k++) svg('circle', { r: 3.5 }, curDots);

  // right: sphere cross-section, arc drawn with an exaggerated angle so it is visible
  const C = { x: 470, y: 170, r: 95 };
  const span = Math.PI * 0.62;                                       // display angle (not to scale)
  const ang = (t) => -Math.PI / 2 - span / 2 + t * span;
  const onCircle = (t) => [C.x + C.r * Math.cos(ang(t)), C.y + C.r * Math.sin(ang(t))];
  svg('circle', { cx: C.x, cy: C.y, r: C.r, fill: 'none', stroke: 'var(--line)', 'stroke-width': 1.5 }, S);
  svg('text', { x: C.x, y: 28, text: 'on the sphere (angle not to scale)', 'text-anchor': 'middle' }, S);
  const [zx, zy] = onCircle(0), [wx, wy] = onCircle(1);
  svg('line', { x1: zx, y1: zy, x2: wx, y2: wy, stroke: 'var(--muted)', 'stroke-width': 1.5, 'stroke-dasharray': '5 4' }, S);
  svg('path', { d: `M${zx} ${zy} A${C.r} ${C.r} 0 0 1 ${wx} ${wy}`, fill: 'none', stroke: 'var(--ink)', 'stroke-width': 2.5 }, S);
  svg('line', { x1: C.x, y1: C.y, x2: zx, y2: zy, stroke: 'var(--line)', 'stroke-width': 1 }, S);
  svg('line', { x1: C.x, y1: C.y, x2: wx, y2: wy, stroke: 'var(--line)', 'stroke-width': 1 }, S);
  svg('circle', { cx: zx, cy: zy, r: 6, fill: 'var(--coral)' }, S);
  svg('circle', { cx: wx, cy: wy, r: 6, fill: 'var(--teal)' }, S);
  svg('text', { x: zx - 12, y: zy + 4, text: 'z', 'text-anchor': 'end', class: 'mono' }, S);
  svg('text', { x: wx + 12, y: wy + 4, text: 'w', class: 'mono' }, S);
  svg('text', { x: C.x, y: C.y - C.r - 10, text: 'arc = geodesic ρ', 'text-anchor': 'middle', 'font-size': 11 }, S);
  svg('text', { x: C.x, y: (zy + wy) / 2 + 16, text: 'chord = Procrustes', 'text-anchor': 'middle', 'font-size': 11 }, S);
  svg('text', { x: C.x, y: C.y + 6, text: 'ρ', 'text-anchor': 'middle', class: 'mono' }, S);
  const walker = svg('circle', { r: 6, fill: 'var(--ink)', stroke: 'var(--surface)', 'stroke-width': 2 }, S);

  let t = 0;
  const sl = slider(ui.controls, { label: 't', min: 0, max: 1, step: 0.01, value: 0, onInput: (v) => { t = v; update(); } });
  button(ui.controls, 'Play', () => play());
  ui.legend([['var(--coral)', 'z (class 0)'], ['var(--teal)', 'w (class 1), aligned to z'], ['var(--ink)', 'path(t)']]);

  let raf = null;
  function play() {
    if (ctx.reducedMotion) { sl.set(t < 0.5 ? 1 : 0); return; }
    cancelAnimationFrame(raf);
    const from = t, to = from < 0.5 ? 1 : 0, t0 = performance.now(), dur = 1400;
    const step = (now) => {
      const u = Math.min(1, (now - t0) / dur); const ease = u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
      sl.set((from + (to - from) * ease).toFixed(2));
      if (u < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
  }

  function geodesicPoint(tt) {
    // slerp: [sin((1−t)ρ)·z + sin(tρ)·w] / sin ρ   (z and w already rotation-aligned)
    if (rho < 1e-9) return z;
    return lerpShape(z, w, Math.sin((1 - tt) * rho) / Math.sin(rho), Math.sin(tt * rho) / Math.sin(rho));
  }
  function update() {
    const P = geodesicPoint(t);
    curPath.setAttribute('d', pathOf(P, SV));
    [...curDots.children].forEach((d, k) => { d.setAttribute('cx', (SV.cx + SV.s * P[k][0]).toFixed(1)); d.setAttribute('cy', (SV.cy + SV.s * P[k][1]).toFixed(1)); });
    const [px, py] = onCircle(t); walker.setAttribute('cx', px); walker.setAttribute('cy', py);
    const distSoFar = t * rho;
    readoutRow(ui.readout, [
      ['t =', fmt(t, 2)],
      ['geodesic ρ = arccos|zᴴw| =', fmt(rho, 4)],
      ['chord √(2 − 2|zᴴw|) =', fmt(chord, 4)],
      ['Procrustes ‖z − e^{iθ*}w‖ =', fmt(procrustes, 4)],
      ['walked so far t·ρ =', fmt(distSoFar, 4)],
    ]);
  }
  update();
  return () => cancelAnimationFrame(raf);
}
