/* Phenotype Lab — PointNet engine ("PointNet, ONNX").
   Loads onnxruntime-web lazily (jsdelivr, pinned; cdnjs serves the package's .wasm files empty), runs models/<id>.onnx with input "points" (1, N, 2)
   float32 = (points − centroid) / input_scale, then applies the linear probe stored in the JSON
   (standardise → w·e + b → sigmoid / softmax). Works on phones; no server. Chapter 08. */

import { centroid, frobenius, sigmoid, softmax } from './classical.js';

export const ORT_VERSION = '1.19.2';
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

let ortPromise = null;
/** Insert the onnxruntime-web script once and resolve with the global `ort`. */
export function loadOrt() {
  if (ortPromise) return ortPromise;
  ortPromise = new Promise((resolve, reject) => {
    if (typeof WebAssembly === 'undefined') { reject(new Error('WebAssembly is not available in this browser')); return; }
    if (window.ort) { resolve(window.ort); return; }
    const s = document.createElement('script');
    s.src = `${ORT_BASE}ort.min.js`;
    s.async = true;
    s.onload = () => {
      try {
        window.ort.env.wasm.wasmPaths = ORT_BASE;
        window.ort.env.wasm.numThreads = 1;   // no SharedArrayBuffer needed, so it runs on plain static hosting
        resolve(window.ort);
      } catch (e) { reject(e); }
    };
    s.onerror = () => { ortPromise = null; reject(new Error('onnxruntime-web could not be loaded from jsdelivr (offline?)')); };
    document.head.appendChild(s);
  });
  return ortPromise;
}

/** Why the engine cannot be offered for this dataset, or null when it can. */
export function onnxUnavailableReason(dataset) {
  const pp = dataset.methods?.pointnet_probe;
  if (!pp || pp.status === 'skipped') return pp?.reason ? `PointNet was skipped: ${pp.reason}` : 'PointNet was not run for this dataset';
  if (!pp.onnx) return 'the JSON names no ONNX model';
  if (!pp.probe || !pp.probe.coef) return 'the JSON carries no probe weights';
  if (typeof WebAssembly === 'undefined') return 'WebAssembly is not available in this browser';
  return null;
}

/**
 * Create the engine for one dataset. `run(points)` takes raw points in any units; because a dragged
 * shape has no real size, the cloud is first given the dataset's typical size (RMS radius = input_scale
 * pixels), then centred and divided by input_scale exactly as during training.
 */
export function createOnnxEngine(dataset) {
  const pp = dataset.methods?.pointnet_probe || {};
  const probe = pp.probe || {};
  const inputScale = typeof pp.input_scale === 'number' ? pp.input_scale : (typeof dataset.input_scale === 'number' ? dataset.input_scale : 1);
  const modelUrl = new URL(pp.onnx || '', document.baseURI).href;
  let session = null, loading = null, outputName = null, lastError = null;

  async function load() {
    if (session) return session;
    if (loading) return loading;
    loading = (async () => {
      const ort = await loadOrt();
      // Fetch the model ourselves so a missing file gives a clear message instead of a parser error.
      let res;
      try { res = await fetch(modelUrl); } catch (e) { throw new Error(`the model file could not be fetched (${e.message})`); }
      if (!res.ok) throw new Error(`model file missing: ${pp.onnx} (HTTP ${res.status}). Run scripts/run_lab.py to export it.`);
      const bytes = new Uint8Array(await res.arrayBuffer());
      try {
        session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
      } catch (e) {
        throw new Error(`ONNX Runtime could not open ${pp.onnx}: ${e.message || e}`);
      }
      const names = session.outputNames || [];
      outputName = names.includes('embedding') ? 'embedding' : names[names.length - 1];
      return session;
    })().catch((e) => { lastError = e; loading = null; throw e; });
    return loading;
  }

  /** Centre, give the typical size, divide by input_scale. Returns {data: Float32Array(N*2), centroid, scale, rms}. */
  function prepare(points) {
    const N = points.length;
    const c = centroid(points);
    const X0 = points.map((p) => [p[0] - c[0], p[1] - c[1]]);
    const fro = frobenius(X0) || 1;
    // typical size: RMS radius = input_scale px  ⇔  Frobenius norm = input_scale·√N
    const px = X0.map((p) => [(p[0] / fro) * inputScale * Math.sqrt(N), (p[1] / fro) * inputScale * Math.sqrt(N)]);
    const data = new Float32Array(N * 2);
    for (let k = 0; k < N; k++) { data[2 * k] = px[k][0] / inputScale; data[2 * k + 1] = px[k][1] / inputScale; }
    let rms = 0; for (let k = 0; k < N; k++) rms += data[2 * k] ** 2 + data[2 * k + 1] ** 2;
    return { data, centroid: c, size: frobenius(X0), rms: Math.sqrt(rms / N), N };
  }

  async function embed(data, N) {
    const ort = await loadOrt();
    const s = await load();
    const feeds = { [s.inputNames.includes('points') ? 'points' : s.inputNames[0]]: new ort.Tensor('float32', data, [1, N, 2]) };
    const out = await s.run(feeds);
    const t = out[outputName] || out[Object.keys(out)[0]];
    return Float32Array.from(t.data);
  }

  function applyProbe(e) {
    const m = probe.scaler_mean || [], sc = probe.scaler_scale || [];
    const x = Array.from(e, (v, i) => (v - (m[i] || 0)) / (sc[i] || 1));
    const coef = probe.coef, b = probe.intercept || [];
    const logits = coef.map((row, ci) => { let s = b[ci] || 0; for (let i = 0; i < row.length; i++) s += row[i] * x[i]; return s; });
    const probs = coef.length === 1 ? [1 - sigmoid(logits[0]), sigmoid(logits[0])] : softmax(logits);
    return { x, logits, probs };
  }

  let busy = false, queued = null;
  /** Run the network and the probe. Concurrent calls are coalesced: only the latest queued input runs next. */
  async function run(points) {
    if (busy) { return new Promise((resolve, reject) => { queued = { points, resolve, reject }; }); }
    busy = true;
    try {
      const prep = prepare(points);
      const t0 = performance.now();
      const e = await embed(prep.data, prep.N);
      const ms = performance.now() - t0;
      const pr = applyProbe(e);
      let norm = 0; for (const v of e) norm += v * v;
      return { ...pr, embedding: e, embedDim: e.length, embedNorm: Math.sqrt(norm), ms, centroid: prep.centroid, size: prep.size, rms: prep.rms, inputScale };
    } finally {
      busy = false;
      if (queued) { const q = queued; queued = null; run(q.points).then(q.resolve, q.reject); }
    }
  }

  /** Permutation test: shuffle the point order, re-embed, report max |Δ embedding|. */
  async function shuffleTest(points) {
    const prep = prepare(points);
    const N = prep.N;
    const order = Array.from({ length: N }, (_, i) => i);
    for (let i = N - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
    const shuffled = new Float32Array(N * 2);
    order.forEach((src, dst) => { shuffled[2 * dst] = prep.data[2 * src]; shuffled[2 * dst + 1] = prep.data[2 * src + 1]; });
    const [e1, e2] = [await embed(prep.data, N), await embed(shuffled, N)];
    let maxDiff = 0; for (let i = 0; i < e1.length; i++) maxDiff = Math.max(maxDiff, Math.abs(e1[i] - e2[i]));
    return { maxDiff, order };
  }

  /**
   * Self-check: embed each stored worst mistake (`shape`, in its original point order) and compare the
   * probe's probabilities with the script's `p_engine` vector (rounded to 4 decimals in the JSON).
   * `checked` is 0 when nothing can be verified.
   */
  async function selfCheck(tol = 1e-3) {
    const out = { ok: true, checked: 0, maxErr: 0, details: [] };
    for (const w of Array.isArray(pp.worst) ? pp.worst : []) {
      if (!Array.isArray(w.shape) || !Array.isArray(w.p_engine)) continue;
      const prep = prepare(w.shape);
      const e = await embed(prep.data, prep.N);
      const { probs } = applyProbe(e);
      const err = Math.max(...w.p_engine.map((p, i) => Math.abs(p - probs[i])));
      out.checked++; out.maxErr = Math.max(out.maxErr, err);
      out.details.push({ index: w.index, expected: w.p_engine, got: probs, err });
      if (err > tol) out.ok = false;
    }
    return out;
  }

  return { load, run, shuffleTest, selfCheck, prepare, get ready() { return !!session; }, get error() { return lastError; }, modelUrl, inputScale, embedDim: pp.embed_dim };
}
