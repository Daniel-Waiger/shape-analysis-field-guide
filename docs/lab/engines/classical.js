/* Phenotype Lab — classical engine ("fitted weights, JS").
   Reproduces the pipeline of scripts/run_lab.py from the `engine` block of a dataset JSON:
     centre → normalise (preshape) → rotate onto the mean (angle of Σ conj(z)·m) → log map
     → tangent vector [Re…, Im…] − pca_mean → project on the PCA components → standardise
     → logistic regression (sigmoid for one weight row, softmax otherwise).
   Pure functions, no DOM. Points are rows: [[x, y], …]. Chapters 02, 03, 06, 08. */

const EPS = 1e-12;

/** Centroid of [[x,y],…]. */
export function centroid(P) {
  let x = 0, y = 0;
  for (const p of P) { x += p[0]; y += p[1]; }
  return [x / P.length, y / P.length];
}
/** Frobenius norm of [[x,y],…] (square root of the summed squares). */
export function frobenius(P) {
  let s = 0;
  for (const p of P) s += p[0] * p[0] + p[1] * p[1];
  return Math.sqrt(s);
}
/** Σ_k conj(z_k)·w_k for z, w as [[x,y],…]: {re, im}. Chapter 03. */
export function hermitian(z, w) {
  let re = 0, im = 0;
  for (let k = 0; k < z.length; k++) {
    re += z[k][0] * w[k][0] + z[k][1] * w[k][1];
    im += z[k][0] * w[k][1] - z[k][1] * w[k][0];
  }
  return { re, im };
}
/** Geodesic distance arccos |⟨z, w⟩| between two preshapes. Chapter 04. */
export function geodesicDistance(z, w) {
  const s = hermitian(z, w);
  return Math.acos(Math.min(1, Math.hypot(s.re, s.im)));
}
export function rotate(P, theta) {
  const c = Math.cos(theta), s = Math.sin(theta);
  return P.map(([x, y]) => [x * c - y * s, x * s + y * c]);
}
/** Centre and scale to unit Frobenius norm. Returns the intermediate numbers too. */
export function preshape(P) {
  const c = centroid(P);
  const X0 = P.map((p) => [p[0] - c[0], p[1] - c[1]]);
  const size = frobenius(X0);
  const s = size > EPS ? size : 1;
  return { centroid: c, size, Z: X0.map((p) => [p[0] / s, p[1] / s]) };
}
export function sigmoid(t) { return 1 / (1 + Math.exp(-t)); }
export function softmax(v) {
  const m = Math.max(...v);
  const e = v.map((t) => Math.exp(t - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((t) => t / s);
}

/** Does this engine block describe a usable classical pipeline for N points? */
export function engineUsable(engine, nPoints) {
  if (!engine || !Array.isArray(engine.mean) || !Array.isArray(engine.components)) return { ok: false, reason: 'no engine block in the JSON' };
  if (nPoints && engine.mean.length !== nPoints) return { ok: false, reason: `the engine expects ${engine.mean.length} points, the dataset has ${nPoints}` };
  if (engine.tangent_layout && engine.tangent_layout !== 're_then_im') return { ok: false, reason: `unknown tangent layout "${engine.tangent_layout}"` };
  return { ok: true };
}

/**
 * Run the whole pipeline on raw points (any units). Returns every intermediate so the page
 * can narrate it: centroid, size, rotation angle, aligned preshape, tangent scores, logits, probabilities.
 */
export function classify(points, engine) {
  const N = points.length;
  const mean = engine.mean;                       // [[re, im]×N]
  const { centroid: c, size, Z } = preshape(points);
  // step 3: rotation onto the mean, θ = angle(Σ conj(z)·m)
  const h = hermitian(Z, mean);
  const theta = Math.atan2(h.im, h.re);
  const Zal = rotate(Z, theta);
  // step 4: log map at the mean: ρ = arccos Re⟨m, z̃⟩, v = (ρ / sin ρ)(z̃ − cos ρ · m)
  let cosRho = 0;
  for (let k = 0; k < N; k++) cosRho += mean[k][0] * Zal[k][0] + mean[k][1] * Zal[k][1];
  cosRho = Math.max(-1, Math.min(1, cosRho));
  const rho = Math.acos(cosRho);
  const f = rho < 1e-12 ? 1 : rho / Math.sin(rho);
  const v = Zal.map((p, k) => [f * (p[0] - cosRho * mean[k][0]), f * (p[1] - cosRho * mean[k][1])]);
  // tangent vector as 2N reals, re then im, minus the PCA mean
  const r = new Float64Array(2 * N);
  const pm = engine.pca_mean || [];
  for (let k = 0; k < N; k++) { r[k] = v[k][0] - (pm[k] || 0); r[N + k] = v[k][1] - (pm[N + k] || 0); }
  // project on the components
  const comps = engine.components;
  const nComp = engine.n_components ? Math.min(engine.n_components, comps.length) : comps.length;
  const scores = new Array(nComp);
  for (let i = 0; i < nComp; i++) { let s = 0; const row = comps[i]; for (let j = 0; j < 2 * N; j++) s += row[j] * r[j]; scores[i] = s; }
  // standardise
  const x = scores.map((s, i) => (s - engine.scaler_mean[i]) / (engine.scaler_scale[i] || 1));
  // logistic regression
  const coef = engine.lr_coef, b = engine.lr_intercept;
  const logits = coef.map((row, ci) => { let s = b[ci] || 0; for (let i = 0; i < nComp; i++) s += row[i] * x[i]; return s; });
  let probs;
  if (coef.length === 1 || engine.binary === true && coef.length === 1) {
    const p1 = sigmoid(logits[0]); probs = [1 - p1, p1];
  } else {
    probs = softmax(logits);
  }
  return { probs, centroid: c, size, theta, thetaDeg: (theta * 180) / Math.PI, preshape: Z, aligned: Zal, rho, tangent: r, scores, x, logits, nComponents: nComp };
}

/**
 * Self-check: run the engine on each worst-mistake shape of the engine's own method and compare
 * with the probability the script stored for the full-data model (`p_engine`). Returns a report;
 * `checked` is 0 when the JSON carries no `p_engine` (then nothing can be verified).
 */
export function selfCheck(dataset, tol = 1e-3) {
  const engine = dataset.engine;
  const method = dataset.methods?.[engine?.method] || dataset.methods?.landmarks_lr;
  const out = { ok: true, checked: 0, maxErr: 0, details: [] };
  if (!engine || !method || !Array.isArray(method.worst)) return out;
  for (const w of method.worst) {
    if (!Array.isArray(w.shape)) continue;
    const ref = w.p_engine;
    if (ref === undefined || ref === null) continue;
    const res = classify(w.shape, engine);
    let err;
    if (Array.isArray(ref)) err = Math.max(...ref.map((p, i) => Math.abs(p - res.probs[i])));
    else err = Math.abs(res.probs[w.pred] - ref);
    out.checked++;
    out.maxErr = Math.max(out.maxErr, err);
    out.details.push({ index: w.index, expected: ref, got: Array.isArray(ref) ? res.probs : res.probs[w.pred], err });
    if (err > tol) out.ok = false;
  }
  return out;
}
