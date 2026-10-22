/* Demo "rotate-align": a slider rotates shape B over shape A with a live error readout;
   "Solve" jumps to the closed-form angle θ* = angle(Σ conj(b_k) a_k). */
import { frame, svg, slider, button, readoutRow, rotate, sqDist, optimalRotationAngle, pathOf, drawDots, fmt, deg, loadWorms } from './_lib.js';

export default async function mount(el, ctx) {
  const worms = await loadWorms(ctx);
  const A = worms.mean;                                   // target (ink)
  const B = rotate(worms.live[0], (70 * Math.PI) / 180); // the shape to rotate, deliberately turned away
  const thetaStar = optimalRotationAngle(B, A);           // rotates B onto A

  const ui = frame(el, {
    title: 'Rotation, by hand and in one line',
    hint: 'Move the slider to rotate B (teal) over A (ink) and watch the error. Then press Solve: the closed-form angle lands on the minimum of the curve.',
    sample: worms.sample,
  });
  const S = ui.svg;
  const SV = { cx: 170, cy: 165, s: 250 };               // shape view
  const PL = { x: 370, y: 50, w: 210, h: 200 };           // error plot

  // shapes
  svg('text', { x: SV.cx, y: 28, text: 'rotate B onto A', 'text-anchor': 'middle' }, S);
  svg('path', { d: pathOf(B, SV), fill: 'none', stroke: 'var(--slate)', 'stroke-width': 2, opacity: 0.5, 'stroke-dasharray': '4 3' }, S);
  svg('path', { d: pathOf(A, SV), fill: 'none', stroke: 'var(--ink)', 'stroke-width': 3, 'stroke-linejoin': 'round' }, S);
  drawDots(S, A, SV, { fill: 'var(--ink)' });
  const bPath = svg('path', { fill: 'none', stroke: 'var(--teal)', 'stroke-width': 2.5, 'stroke-linejoin': 'round' }, S);
  const bDots = svg('g', { fill: 'var(--teal)' }, S);
  for (let k = 0; k < B.length; k++) svg('circle', { r: 3.5 }, bDots);
  const ties = svg('g', { stroke: 'var(--coral)', 'stroke-width': 1, opacity: 0.6 }, S);
  for (let k = 0; k < B.length; k++) svg('line', {}, ties);

  // error curve E(θ) = ‖A − e^{iθ}B‖² for θ in [−π, π]
  const samples = [];
  let emax = 0;
  for (let i = 0; i <= 120; i++) { const th = -Math.PI + (2 * Math.PI * i) / 120; const e = sqDist(A, rotate(B, th)); samples.push([th, e]); emax = Math.max(emax, e); }
  const px = (th) => PL.x + ((th + Math.PI) / (2 * Math.PI)) * PL.w;
  const py = (e) => PL.y + PL.h - (e / (emax || 1)) * PL.h;
  svg('rect', { x: PL.x, y: PL.y, width: PL.w, height: PL.h, fill: 'var(--paper)', stroke: 'var(--line)', rx: 6 }, S);
  svg('path', { d: samples.map(([th, e], i) => `${i ? 'L' : 'M'}${px(th).toFixed(1)} ${py(e).toFixed(1)}`).join(' '), fill: 'none', stroke: 'var(--muted)', 'stroke-width': 1.5 }, S);
  svg('text', { x: PL.x + PL.w / 2, y: PL.y - 12, text: 'error ‖A − e^{iθ}B‖² against θ', 'text-anchor': 'middle' }, S);
  svg('text', { x: PL.x, y: PL.y + PL.h + 16, text: '−180°', 'font-size': 11 }, S);
  svg('text', { x: PL.x + PL.w, y: PL.y + PL.h + 16, text: '180°', 'text-anchor': 'end', 'font-size': 11 }, S);
  svg('text', { x: PL.x + PL.w / 2, y: PL.y + PL.h + 16, text: '0°', 'text-anchor': 'middle', 'font-size': 11 }, S);
  const minDot = svg('circle', { cx: px(thetaStar), cy: py(sqDist(A, rotate(B, thetaStar))), r: 6, fill: 'var(--gold)', opacity: 0 }, S);
  const curDot = svg('circle', { r: 5, fill: 'var(--teal)', stroke: 'var(--surface)', 'stroke-width': 1.5 }, S);
  const curLine = svg('line', { y1: PL.y, y2: PL.y + PL.h, stroke: 'var(--teal)', 'stroke-width': 1, 'stroke-dasharray': '3 3' }, S);

  let solved = false;
  let theta = 0;
  const sl = slider(ui.controls, { label: 'θ', min: -180, max: 180, step: 0.1, value: 0, format: (v) => `${v.toFixed(1)}°`, onInput: (v) => { theta = (v * Math.PI) / 180; update(); } });
  button(ui.controls, 'Solve', () => animateTo(deg(thetaStar)), true);
  button(ui.controls, 'Reset', () => { solved = false; minDot.setAttribute('opacity', 0); sl.set(0); });
  ui.legend([['var(--ink)', 'A (target)'], ['var(--teal)', 'B, rotated by θ'], ['var(--slate)', 'B before rotating'], ['var(--coral)', 'point-to-point error']]);

  let raf = null;
  function animateTo(target) {
    solved = true; minDot.setAttribute('opacity', 1);
    if (ctx.reducedMotion) { sl.set(target.toFixed(1)); return; }
    const from = sl.get(); const t0 = performance.now(); const dur = 650;
    cancelAnimationFrame(raf);
    const step = (now) => {
      const u = Math.min(1, (now - t0) / dur); const ease = 1 - Math.pow(1 - u, 3);
      sl.set((from + (target - from) * ease).toFixed(1));
      if (u < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
  }

  function update() {
    const Bt = rotate(B, theta);
    const err = sqDist(A, Bt);
    bPath.setAttribute('d', pathOf(Bt, SV));
    [...bDots.children].forEach((d, k) => { d.setAttribute('cx', (SV.cx + SV.s * Bt[k][0]).toFixed(1)); d.setAttribute('cy', (SV.cy + SV.s * Bt[k][1]).toFixed(1)); });
    [...ties.children].forEach((l, k) => {
      l.setAttribute('x1', (SV.cx + SV.s * A[k][0]).toFixed(1)); l.setAttribute('y1', (SV.cy + SV.s * A[k][1]).toFixed(1));
      l.setAttribute('x2', (SV.cx + SV.s * Bt[k][0]).toFixed(1)); l.setAttribute('y2', (SV.cy + SV.s * Bt[k][1]).toFixed(1));
    });
    curDot.setAttribute('cx', px(theta)); curDot.setAttribute('cy', py(err));
    curLine.setAttribute('x1', px(theta)); curLine.setAttribute('x2', px(theta));
    readoutRow(ui.readout, [
      ['θ =', `${fmt(deg(theta), 1)}° (${fmt(theta, 3)} rad)`],
      ['error ‖A − e^{iθ}B‖² =', fmt(err, 4)],
      ['closed form θ* = angle(Σ conj(b_k)·a_k) =', solved ? `${fmt(deg(thetaStar), 1)}°` : 'press Solve', solved ? 'gold' : ''],
    ]);
  }
  update();
  return () => cancelAnimationFrame(raf);
}
