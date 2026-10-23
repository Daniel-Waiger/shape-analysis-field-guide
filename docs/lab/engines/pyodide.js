/* Phenotype Lab — "Run it yourself" (shapekit in Pyodide, desktop only).
   Offered only on wide screens with a fine pointer. On first open it starts a Web Worker that loads
   Pyodide (jsdelivr, pinned), micropip-installs the pure-Python shapekit wheel from py/, fetches the
   dataset files from data/files/ (copies of data/), and runs the classical cross-validation pipeline with the reader's
   own parameters (components, C, spline M). Stop terminates the worker. About 20 MB on first load. */

export const PYODIDE_VERSION = '0.26.4';
/** The download on first use (Pyodide + numpy + scipy + scikit-learn), an estimate: it is not measured by the page. */
export const PYODIDE_SIZE_ESTIMATE = 'roughly 20 MB, an estimate';
const INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const WHEEL = 'py/shapekit-0.1.0-py3-none-any.whl';

/** Data files per dataset, relative to the repository's data/ directory (as the registry loads them). */
const DATA_FILES = {
  celegans: ['celegans_points.npy', 'celegans_labels.npy'],
  mef: ['mef_outlines_subset.npz'],
  brain: ['brain_landmarks.npz'],
  gorilla: ['gorilla_landmarks.npz'],
  leaves: ['leaves_outlines.npz'],
};

export function pyodideOffered() {
  try { return window.matchMedia('(min-width: 900px) and (pointer: fine)').matches; } catch (e) { return false; }
}

const PY_SCRIPT = String.raw`
import os, json, time, sys
import numpy as np
import sklearn
os.environ["SHAPEKIT_DATA"] = "/data"
from shapekit.lab_datasets import get
from shapekit.preshape import preshape, to_complex
from shapekit.shape_space import frechet_mean, log_map
from shapekit.tangent_pca import TangentPCA, tangent_to_real
from shapekit.bspline import BSplineCurve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

P = json.loads(PARAMS)
ds = get(P["id"])
t0 = time.time()
points, labels = ds.load()
print(f"loaded {ds.id}: {points.shape[0]} shapes x {points.shape[1]} points, classes {list(ds.classes)}")
print(f"scikit-learn {sklearn.__version__} in Pyodide (the page's JSON names the version scripts/run_lab.py used)")
if P["route"] == "spline":
    M = int(P["M"])
    points = np.stack([BSplineCurve(M, closed=ds.closed).fit(p).control_points_ for p in points])
    print(f"spline route: {M} control points per shape ({'closed' if ds.closed else 'open'} curve)")
Z = to_complex(preshape(points))
labels = np.asarray(labels).astype(int)
n_classes = int(labels.max()) + 1
chance = float(np.bincount(labels).max() / len(labels))

def fit(Ztr, ytr):
    mean = frechet_mean(Ztr)
    pca = TangentPCA(mean=mean).fit(Ztr)
    keep = int(min(P["components"], np.sum(pca.explained_variance_ > 1e-10)))
    comps = pca.components_[:keep]
    R = tangent_to_real(np.stack([log_map(mean, z) for z in Ztr]))
    S = (R - pca.mean_tangent_) @ comps.T
    sc = StandardScaler().fit(S)
    lr = LogisticRegression(max_iter=2000, C=float(P["C"])).fit(sc.transform(S), ytr)
    return mean, pca, comps, sc, lr, keep

def predict(model, Zte):
    mean, pca, comps, sc, lr, keep = model
    R = tangent_to_real(np.stack([log_map(mean, z) for z in Zte]))
    S = (R - pca.mean_tangent_) @ comps.T
    return lr.predict_proba(sc.transform(S))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
accs = []
cm = np.zeros((n_classes, n_classes), int)
for f, (tr, te) in enumerate(skf.split(np.zeros(len(labels)), labels)):
    model = fit(Z[tr], labels[tr])
    pred = predict(model, Z[te]).argmax(1)
    acc = float((pred == labels[te]).mean())
    accs.append(acc)
    for t, p in zip(labels[te], pred):
        cm[t, p] += 1
    print(f"fold {f+1}: {acc:.3f}  (components kept: {model[5]})")
print("")
pooled = float(np.trace(cm) / cm.sum())
print(f"pooled out-of-fold accuracy {pooled:.4f} (the page's kind of number)   mean of folds {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"chance {chance:.3f}   C = {P['C']}   components ≤ {P['components']}")
print("confusion matrix (rows = true, columns = predicted):")
for r in cm:
    print("  " + "  ".join(f"{v:5d}" for v in r))
print(f"done in {time.time() - t0:.1f} s")
`;

/** The worker body, inlined so the lab stays a handful of files. Runs classic (importScripts). */
function workerSource() {
  return `
const INDEX_URL = ${JSON.stringify(INDEX_URL)};
let pyodide = null;
function log(text) { postMessage({ type: 'log', text }); }
function status(text) { postMessage({ type: 'status', text }); }
async function fetchBytes(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error('HTTP ' + r.status + ' for ' + url);
  return new Uint8Array(await r.arrayBuffer());
}
onmessage = async (ev) => {
  const m = ev.data;
  try {
    if (m.type === 'init') {
      status('loading Pyodide ' + m.version + ' from jsdelivr (' + m.sizeEstimate + ' on first visit)…');
      importScripts(INDEX_URL + 'pyodide.js');
      pyodide = await loadPyodide({ indexURL: INDEX_URL, stdout: log, stderr: log });
      status('loading numpy, scipy, scikit-learn…');
      await pyodide.loadPackage(['numpy', 'scipy', 'scikit-learn', 'micropip']);
      status('installing shapekit wheel…');
      const probe = await fetch(m.wheelUrl, { method: 'HEAD' });
      if (!probe.ok) throw new Error('the shapekit wheel is not published yet: ' + m.wheelUrl + ' (HTTP ' + probe.status + '). Build it with scripts/build_lab_wheel.sh.');
      // numpy/scipy/scikit-learn are already loaded above; deps=False keeps micropip off PyPI (the wheel also lists matplotlib, unused here)
      pyodide.globals.set('WHEEL_URL', m.wheelUrl);
      await pyodide.runPythonAsync('import micropip\\nawait micropip.install(WHEEL_URL, deps=False)');
      status('fetching the dataset…');
      pyodide.FS.mkdirTree('/data');
      for (const f of m.dataFiles) {
        let bytes;
        try { bytes = await fetchBytes(m.dataBase + f); }
        catch (e) { throw new Error('data/files/' + f + ' could not be fetched (' + e.message + '). Run scripts/run_lab.py locally instead.'); }
        pyodide.FS.writeFile('/data/' + f, bytes);
        log('fetched data/' + f + ' (' + (bytes.length / 1024).toFixed(0) + ' kB)');
      }
      postMessage({ type: 'ready' });
    } else if (m.type === 'run') {
      if (!pyodide) throw new Error('Pyodide is not ready');
      status('running the classical pipeline…');
      pyodide.globals.set('PARAMS', JSON.stringify(m.params));
      await pyodide.runPythonAsync(m.script);
      postMessage({ type: 'done' });
    }
  } catch (e) {
    postMessage({ type: 'error', text: String(e && e.message ? e.message : e) });
  }
};`;
}

/**
 * Mount the collapsed panel into `container`. Returns a cleanup function.
 */
export function mountPyodidePanel(container, dataset) {
  const esc = (t) => String(t).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const el = (tag, attrs = {}, parent) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) { if (k === 'text') n.textContent = v; else if (k === 'html') n.innerHTML = v; else n.setAttribute(k, v); }
    if (parent) parent.appendChild(n);
    return n;
  };
  const details = el('details', { class: 'py-panel' }, container);
  const summary = el('summary', {}, details);
  el('span', { text: `Run it yourself (desktop, ${PYODIDE_SIZE_ESTIMATE})` }, summary);
  el('span', { class: 'small', text: 'optional · shapekit + scikit-learn in Pyodide, in this tab' }, summary);
  const body = el('div', { class: 'py-body' }, details);
  el('p', { class: 'small muted', html: 'This loads Python into the browser and re-runs the classical cross-validation (preshape → Fréchet mean → tangent PCA → standardize → logistic regression) with parameters you choose. It needs the wheel in <code>py/</code> and the dataset files in <code>data/files/</code> (copies of the repository’s <code>data/</code> folder, kept identical by a test); nothing is uploaded anywhere.' }, body);
  el('p', { class: 'py-why', html: '<strong>Why its numbers differ a little from the page.</strong> This panel runs the scikit-learn version bundled with Pyodide (printed in the log), while the page’s JSON was written with the version in the repository; it refits the Fréchet mean and the tangent PCA inside every fold, while the script fits them once on all shapes; it reports both the pooled out-of-fold accuracy (the page’s number) and the mean of the five fold accuracies; and it takes the points as the registry loads them, with no further start-point alignment' + (dataset.correspondence ? ' (the script\u2019s correspondence for this dataset: ' + esc(dataset.correspondence) + ')' : '') + '. Expect agreement to a few thousandths, not to the last digit.' }, body);
  const form = el('form', { class: 'py-form' }, body);
  const isCloud = dataset.representation === 'pointcloud';
  const engine = dataset.engine || {};
  const mk = (label, name, type, value, extra = {}) => {
    const l = el('label', {}, form); el('span', { text: label }, l);
    const i = el('input', { type, name, value, ...extra }, l); return i;
  };
  const routeL = el('label', {}, form); el('span', { text: 'route' }, routeL);
  const route = el('select', { name: 'route' }, routeL);
  el('option', { value: 'landmarks', text: dataset.representation === 'outline' ? 'resampled outline points' : 'landmarks' }, route);
  el('option', { value: 'spline', text: 'spline control points' }, route);
  const comps = mk('components (max)', 'components', 'number', engine.n_components || 20, { min: 1, max: 60, step: 1 });
  const C = mk('C (inverse regularization)', 'C', 'number', 1, { min: 0.001, step: 'any' });
  const M = mk('spline M', 'M', 'number', dataset.methods?.spline_lr?.M || 6, { min: 4, max: 60, step: 1 });
  const run = el('button', { type: 'submit', class: 'btn primary', text: 'Load Python and run' }, form);
  const stop = el('button', { type: 'button', class: 'btn', text: 'Stop / clear' }, form);
  const status = el('p', { class: 'py-status', role: 'status', 'aria-live': 'polite', text: isCloud ? 'This dataset is an unordered point cloud: the classical pipeline needs ordered points, so there is nothing to run here. Pick a landmark or outline dataset.' : 'Not started.' }, body);
  const log = el('pre', { class: 'py-log', 'aria-label': 'Python output' }, body);
  if (isCloud) { run.disabled = true; route.disabled = true; comps.disabled = true; C.disabled = true; M.disabled = true; }
  const syncM = () => { M.disabled = isCloud || route.value !== 'spline'; };
  route.addEventListener('change', syncM); syncM();

  let worker = null, ready = false, running = false;
  const setStatus = (t) => { status.textContent = t; };
  const append = (t) => { log.textContent += (log.textContent ? '\n' : '') + t; log.scrollTop = log.scrollHeight; };
  function kill() {
    if (worker) { worker.terminate(); worker = null; }
    ready = false; running = false; run.disabled = isCloud; run.textContent = 'Load Python and run';
  }
  function start() {
    kill();
    log.textContent = '';
    const blob = new Blob([workerSource()], { type: 'text/javascript' });
    worker = new Worker(URL.createObjectURL(blob));
    worker.onmessage = (ev) => {
      const m = ev.data;
      if (m.type === 'log') append(m.text);
      else if (m.type === 'status') setStatus(m.text);
      else if (m.type === 'ready') { ready = true; setStatus('Python is ready.'); runNow(); }
      else if (m.type === 'done') { running = false; run.disabled = false; run.textContent = 'Run again'; setStatus('Finished. Change a parameter and run again.'); }
      else if (m.type === 'error') { running = false; run.disabled = false; run.textContent = 'Retry'; setStatus('Stopped: ' + m.text); append('[error] ' + m.text); }
    };
    worker.onerror = (e) => { setStatus('The Python worker failed: ' + (e.message || 'unknown error')); run.disabled = false; running = false; };
    const files = dataset.data_files || DATA_FILES[dataset.id] || [];
    worker.postMessage({
      type: 'init', version: PYODIDE_VERSION, sizeEstimate: PYODIDE_SIZE_ESTIMATE,
      wheelUrl: new URL(WHEEL, document.baseURI).href,
      dataBase: new URL('data/files/', document.baseURI).href,
      dataFiles: files,
    });
    run.disabled = true; run.textContent = 'Loading…';
  }
  function params() {
    return { id: dataset.id, route: route.value, components: Math.max(1, parseInt(comps.value, 10) || 20), C: Math.max(1e-4, parseFloat(C.value) || 1), M: Math.max(4, parseInt(M.value, 10) || 6) };
  }
  function runNow() {
    if (!worker || !ready || running) return;
    running = true; run.disabled = true; run.textContent = 'Running…';
    append(`--- run: ${JSON.stringify(params())}`);
    worker.postMessage({ type: 'run', params: params(), script: PY_SCRIPT });
  }
  form.addEventListener('submit', (e) => { e.preventDefault(); if (isCloud) return; if (!worker) start(); else if (ready) runNow(); });
  stop.addEventListener('click', () => { kill(); log.textContent = ''; setStatus('Stopped and cleared.'); });
  details.addEventListener('toggle', () => { if (details.open && !worker && !isCloud) setStatus(`Press "Load Python and run" to start (${PYODIDE_SIZE_ESTIMATE} from jsdelivr on first use).`); });
  return () => kill();
}
