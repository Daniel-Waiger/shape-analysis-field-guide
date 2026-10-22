/* Demo "preshape": drag any of the 12 landmarks in the image frame; the centered and
   size-normalized version (the preshape) updates live. Readouts show the centroid
   that is subtracted and the Frobenius norm that is divided out. */
import { frame, svg, h, button, draggable, readoutRow, centroid, center, frobenius, normalize, pathOf, fmt, sampleWorms, loadWorms, scaleShape } from './_lib.js';

export default async function mount(el, ctx) {
  const worms = await loadWorms(ctx);
  // Raw "image" landmarks in a 260 × 280 pixel frame. With real data we place the live example
  // preshape into the frame at a realistic pixel scale; otherwise the built-in bent worm is used.
  const initial = worms.sample
    ? sampleWorms().bent
    : scaleShape(worms.live[0], 230).map(([x, y]) => [x + 130, y + 150]);
  let raw = initial.map((p) => [...p]);

  const ui = frame(el, {
    title: 'Preshape: remove position and size',
    hint: 'Drag any landmark in the image frame (left). The preshape (right) is the same worm after subtracting the centroid and dividing by the Frobenius norm.',
    sample: worms.sample,
  });
  const S = ui.svg;
  const FR = { x: 20, y: 20, w: 260, h: 280 };          // image frame, pixel coordinates map 1:1
  const PV = { cx: 450, cy: 160, s: 240 };               // preshape view

  // left: image frame
  svg('rect', { x: FR.x, y: FR.y, width: FR.w, height: FR.h, rx: 8, fill: 'var(--paper)', stroke: 'var(--line)' }, S);
  svg('text', { x: FR.x + 8, y: FR.y + 16, text: 'image frame (pixels)' }, S);
  const rawPath = svg('path', { fill: 'none', stroke: 'var(--ink)', 'stroke-width': 2, 'stroke-linejoin': 'round', opacity: 0.55 }, S);
  const centroidMark = svg('g', { stroke: 'var(--gold)', 'stroke-width': 2, fill: 'none' }, S);
  svg('circle', { r: 6 }, centroidMark);
  svg('path', { d: 'M-10 0H10M0 -10V10' }, centroidMark);
  const handles = raw.map((p, k) => {
    const c = svg('circle', { cx: p[0] + FR.x, cy: p[1] + FR.y, r: 7, fill: 'var(--teal)', stroke: 'var(--surface)', 'stroke-width': 2, 'aria-label': `landmark ${k + 1}` }, S);
    draggable(S, c, (x, y) => {
      raw[k] = [Math.min(FR.w, Math.max(0, x - FR.x)), Math.min(FR.h, Math.max(0, y - FR.y))];
      update();
    });
    return c;
  });

  // right: preshape panel
  svg('text', { x: PV.cx, y: FR.y + 16, text: 'preshape (centroid 0, norm 1)', 'text-anchor': 'middle' }, S);
  const axes = svg('g', { stroke: 'var(--line)', 'stroke-width': 1, 'stroke-dasharray': '3 4' }, S);
  svg('line', { x1: PV.cx - 130, x2: PV.cx + 130, y1: PV.cy, y2: PV.cy }, axes);
  svg('line', { y1: PV.cy - 130, y2: PV.cy + 130, x1: PV.cx, x2: PV.cx }, axes);
  svg('circle', { cx: PV.cx, cy: PV.cy, r: 4, fill: 'var(--gold)' }, S);
  const centeredPath = svg('path', { fill: 'none', stroke: 'var(--slate)', 'stroke-width': 2, opacity: 0.5, 'stroke-dasharray': '4 3' }, S);
  const prePath = svg('path', { fill: 'none', stroke: 'var(--ink)', 'stroke-width': 2.5, 'stroke-linejoin': 'round' }, S);
  const preDots = svg('g', { fill: 'var(--ink)' }, S);
  for (let k = 0; k < 12; k++) svg('circle', { r: 3.5 }, preDots);
  svg('text', { x: PV.cx, y: PV.cy + 136, text: 'dashed: centered, before scaling (pixel size)', 'text-anchor': 'middle', 'font-size': 11 }, S);

  button(ui.controls, 'Reset landmarks', () => { raw = initial.map((p) => [...p]); handles.forEach((c, k) => { c.setAttribute('cx', raw[k][0] + FR.x); c.setAttribute('cy', raw[k][1] + FR.y); }); update(); });
  ui.legend([['var(--teal)', 'draggable landmarks'], ['var(--gold)', 'centroid'], ['var(--ink)', 'preshape']]);

  function update() {
    const c = centroid(raw);
    const X0 = center(raw);
    const norm = frobenius(X0);
    const Z = normalize(X0);
    rawPath.setAttribute('d', pathOf(raw, { cx: FR.x, cy: FR.y, s: 1 }));
    centroidMark.setAttribute('transform', `translate(${(c[0] + FR.x).toFixed(1)} ${(c[1] + FR.y).toFixed(1)})`);
    // centered-only shape, drawn at the pixel scale of the frame so the reader sees "same size, moved"
    centeredPath.setAttribute('d', pathOf(X0, { cx: PV.cx, cy: PV.cy, s: 1 }));
    prePath.setAttribute('d', pathOf(Z, PV));
    [...preDots.children].forEach((dot, k) => { dot.setAttribute('cx', (PV.cx + PV.s * Z[k][0]).toFixed(1)); dot.setAttribute('cy', (PV.cy + PV.s * Z[k][1]).toFixed(1)); });
    readoutRow(ui.readout, [
      ['centroid x̄, ȳ =', `(${fmt(c[0], 1)}, ${fmt(c[1], 1)}) px`],
      ['‖X − x̄‖_F =', `${fmt(norm, 1)} px`],
      ['preshape centroid =', '(0, 0)'],
      ['preshape norm =', fmt(frobenius(Z), 3)],
    ]);
  }
  update();
}
