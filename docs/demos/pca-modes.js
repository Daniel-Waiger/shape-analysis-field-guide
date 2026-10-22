/* Demo "pca-modes": four sliders (in SD units) move along the first tangent-PCA modes;
   the shape shown is exp_map(mean, Σ wᵢ sᵢ cᵢ) with exp_map(m, v) = cos(t)·m + (sin t / t)·v, t = ‖v‖.
   Uses data/worms.json (pca block); falls back to hand-made illustrative modes. */
import { frame, svg, slider, button, readoutRow, pathOf, drawDots, fmt, frobenius, hermitian, loadWorms, sampleWorms, preshape } from './_lib.js';

/** Turn a 24-vector into [[x,y]×12] using the detected layout ('stacked' = 12 re then 12 im). */
function toPoints(v, layout) {
  const n = v.length / 2, P = [];
  for (let k = 0; k < n; k++) P.push(layout === 'interleaved' ? [v[2 * k], v[2 * k + 1]] : [v[k], v[n + k]]);
  return P;
}
function detectLayout(meanVec, mean) {
  if (!meanVec || meanVec.length !== 2 * mean.length) return 'stacked';
  const errS = mean.reduce((s, p, k) => s + Math.abs(meanVec[k] - p[0]) + Math.abs(meanVec[mean.length + k] - p[1]), 0);
  const errI = mean.reduce((s, p, k) => s + Math.abs(meanVec[2 * k] - p[0]) + Math.abs(meanVec[2 * k + 1] - p[1]), 0);
  return errI < errS ? 'interleaved' : 'stacked';
}
/** Project out the complex mean direction and normalise: a clean unit tangent vector. */
function tangentize(v, mean) {
  const ip = hermitian(mean, v); // ⟨m, v⟩ = Σ conj(m_k) v_k
  const proj = v.map((p, k) => {
    const mx = mean[k][0], my = mean[k][1];
    // v − m · ⟨m, v⟩ with complex multiplication (mx + i my)(re + i im)
    return [p[0] - (mx * ip.re - my * ip.im), p[1] - (mx * ip.im + my * ip.re)];
  });
  const n = frobenius(proj) || 1; return proj.map(([x, y]) => [x / n, y / n]);
}
function sampleModes(mean) {
  // Four illustrative directions: C-bend, S-bend, one-end hook, short wave (NOT fitted results).
  const xs = mean.map((p) => p[0]); const span = Math.max(...xs) - Math.min(...xs) || 1;
  const u = xs.map((x) => (2 * (x - Math.min(...xs))) / span - 1); // −1 … 1 along the body
  const make = (f) => tangentize(mean.map((p, k) => [0, f(u[k])]), mean);
  return {
    components: [make((x) => x * x - 1 / 3), make((x) => x * x * x - 0.6 * x), make((x) => (x > 0.3 ? (x - 0.3) ** 2 : 0)), make((x) => Math.sin(3 * Math.PI * x) * 0.5)],
    sdev: [0.14, 0.07, 0.05, 0.035], explained: null,
  };
}

export default async function mount(el, ctx) {
  const worms = await loadWorms(ctx);
  const mean = worms.mean;
  let modes;
  if (worms.pca && Array.isArray(worms.pca.components) && worms.pca.components.length >= 1) {
    const declared = String(worms.pca.tangent_layout || '').toLowerCase();
    const layout = declared.includes('re') ? 'stacked' : declared.includes('xy') || declared.includes('inter') ? 'interleaved' : detectLayout(worms.pca.mean_tangent, mean);
    const comps = worms.pca.components.slice(0, 4).map((c) => toPoints(c, layout));
    const offset = Array.isArray(worms.pca.mean_tangent) && worms.pca.mean_tangent.length === 2 * mean.length ? toPoints(worms.pca.mean_tangent, layout) : null;
    modes = { components: comps, sdev: (worms.pca.sdev || []).slice(0, 4), explained: worms.pca.explained || null, offset };
    while (modes.sdev.length < comps.length) modes.sdev.push(0.05);
  } else modes = sampleModes(mean);
  const nModes = modes.components.length;

  const ui = frame(el, {
    title: 'Modes of variation',
    hint: 'Each slider moves along one principal direction of the tangent space, in standard deviations. The shape is mapped back onto the sphere with the exponential map.',
    sample: worms.sample || !worms.pca,
  });
  const S = ui.svg;
  const SV = { cx: 300, cy: 160, s: 300 };
  svg('text', { x: 20, y: 28, text: 'shape = exp_map(mean, Σ wᵢ · sᵢ · cᵢ)', class: 'mono' }, S);
  svg('path', { d: pathOf(mean, SV), fill: 'none', stroke: 'var(--slate)', 'stroke-width': 2, opacity: 0.5, 'stroke-dasharray': '4 3' }, S);
  drawDots(S, mean, SV, { fill: 'var(--slate)', opacity: 0.5, r: 3 });
  const curPath = svg('path', { fill: 'none', stroke: 'var(--ink)', 'stroke-width': 3, 'stroke-linejoin': 'round' }, S);
  const curDots = svg('g', { fill: 'var(--ink)' }, S);
  for (let k = 0; k < mean.length; k++) svg('circle', { r: 3.5 }, curDots);

  const weights = new Array(nModes).fill(0);
  const sliders = [];
  for (let i = 0; i < nModes; i++) {
    const label = `mode ${i + 1}` + (modes.explained && Number.isFinite(modes.explained[i]) ? ` (${(100 * modes.explained[i]).toFixed(0)}%)` : '');
    sliders.push(slider(ui.controls, { label, min: -3, max: 3, step: 0.1, value: 0, format: (v) => `${v >= 0 ? '+' : ''}${v.toFixed(1)} SD`, onInput: (v) => { weights[i] = v; update(); } }));
  }
  button(ui.controls, 'Reset', () => sliders.forEach((s) => s.set(0)));
  ui.legend([['var(--slate)', 'mean shape'], ['var(--ink)', 'shape at these weights']]);

  function update() {
    // tangent vector v = mean_tangent + Σ wᵢ sᵢ cᵢ  (mean_tangent is ≈ 0 at the Fréchet mean)
    const v = mean.map((_, k) => (modes.offset ? [modes.offset[k][0], modes.offset[k][1]] : [0, 0]));
    for (let i = 0; i < nModes; i++) {
      const a = weights[i] * modes.sdev[i];
      modes.components[i].forEach((c, k) => { v[k][0] += a * c[0]; v[k][1] += a * c[1]; });
    }
    const t = frobenius(v);
    const sinc = t < 1e-9 ? 1 : Math.sin(t) / t;
    const P = mean.map((m, k) => [Math.cos(t) * m[0] + sinc * v[k][0], Math.cos(t) * m[1] + sinc * v[k][1]]);
    curPath.setAttribute('d', pathOf(P, SV));
    [...curDots.children].forEach((d, k) => { d.setAttribute('cx', (SV.cx + SV.s * P[k][0]).toFixed(1)); d.setAttribute('cy', (SV.cy + SV.s * P[k][1]).toFixed(1)); });
    const rows = weights.map((w, i) => [`w${i + 1}·s${i + 1} =`, fmt(w * modes.sdev[i], 3)]);
    rows.push(['‖v‖ = distance from mean =', fmt(t, 3)]);
    rows.push(['‖shape‖ =', fmt(frobenius(P), 3)]);
    readoutRow(ui.readout, rows);
  }
  update();
}
