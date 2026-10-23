# Build spec: Phenotype Lab (chapter 10)

Read SPEC.md first (identity, layout, conventions, shapekit API). This file adds the Lab. The mockup it
implements is described at the end. If you must deviate, record it under "Deviations" at the bottom.

## What the Lab is

A page inside the site, `docs/lab/` (URL `…/lab/`), listed in the sidebar as "10 · Phenotype Lab" and linked
from chapter 09's Next line. It shows, for several labelled datasets, whether shape alone can classify the
phenotype, using the three method families the guide teaches:

| key | name | input | pipeline |
|---|---|---|---|
| `landmarks_lr` | Landmarks + logistic regression | ordered points with correspondence (landmarks or arc-length-resampled outline) | preshape → align to Fréchet mean → log map → tangent PCA (keep components with variance > 1e-10, cap 40) → StandardScaler → multinomial LogisticRegression |
| `spline_lr` | Spline + logistic regression | ordered outline (open or closed) | `BSplineCurve(M).fit` → control points → same geometry as above on the control points |
| `pointnet_probe` | PointNet + linear probe | any point set, order ignored | `shapekit.embed.train_pointnet` (seeded) → embeddings → linear probe; also the untrained baseline |

Everything is computed by **`scripts/run_lab.py`** (deterministic, seed 0), which writes `docs/lab/data/*.json`
and `docs/lab/models/*.onnx`. Nothing on the page is typed in by hand. Stratified 5-fold CV with shuffle,
`random_state=0`; chance = largest-class fraction. "Worst mistakes" = the 3 wrong predictions with the highest
confidence in the wrong class.

## Datasets and licences (the registry: `shapekit/lab_datasets.py`)

| id | title | representation | classes | source and licence | in repo? |
|---|---|---|---|---|---|
| `celegans` | C. elegans, live vs dead | landmarks (12) | dead 579 / live 798 | BBBC010, CC0; tutorial-prepared | yes (`data/celegans_*.npy`) |
| `mef` | Mouse fibroblasts, wild-type vs LMNA−/− | point cloud (120, unordered) | 600 / 600 | kukionfr Micropattern_MEF_LMNA_Image, MIT © 2020 Kyu Sang Han | yes (`data/mef_outlines_subset.npz`) |
| `brain` | Brain landmarks, schizophrenia vs control | landmarks (13) | con 14 / scz 14 | `schizophrenia` dataset of the R package `shapes` (Dryden), GPL-2; data from Bookstein (1996) | yes, fetched by `scripts/fetch_datasets.py` from the CRAN tarball, stored as `data/brain_landmarks.npz` **with `data/brain_landmarks.LICENSE` (GPL-2 notice)** |
| `gorilla` | Gorilla skulls, female vs male | landmarks (8) | f 30 / m 29 | same package (`gorf.dat`, `gorm.dat`), GPL-2; O'Higgins & Dryden 1993 | yes, same mechanism (`data/gorilla_landmarks.npz`) |
| `leaves` | Leaf outlines, species | closed outline (150 points, ordered) | 15 species (verify counts) | figshare 14551005 "Outlines of Swedish leaves", CC BY 4.0 (Leroi, Marsland, Salili-James 2022), derived from Söderkvist (2001) Swedish Leaf dataset (author's Zenodo copy CC BY 4.0) | yes if the CSV carries species labels (`data/leaves_outlines.npz` + attribution); if it does not, fall back to Zenodo 19663514 images → contours (CC BY 4.0), documenting the extraction |
| `rbc` | Red blood cells, sickle-cell disease | closed outline from masks | circular / elongated / other | erythrocytesIDB v2, Zenodo 18299474, **CC BY-NC-ND 4.0** | **never**. `fetch_datasets.py --rbc` downloads the 594 MB zip to `data/_local/` (git-ignored) and extracts outlines locally; `run_lab.py` then writes `rbc.json` with `"shapes_shown": false` and no coordinates of any kind (no examples, no mean, no modes, no scatter, no neighbours, no engine weights); only accuracies, confusion matrices and counts. If the data is absent, `rbc.json` is written with `"status": "fetch_locally"` and the page shows how to fetch it |

Every dataset entry in the registry carries: `id, title, question, representation, class names, source
{name, citation, url, license, redistributed: bool}, loader()`. `ATTRIBUTION.md` gains a "Lab datasets" section
with the same facts, and the names found today: MEF images © 2020 Kyu Sang Han (MIT); leaves outlines by Armand
Leroi, Stephen Marsland and Arianna Salili-James (CC BY 4.0) from Oskar J. O. Söderkvist's Swedish Leaf dataset
(CC BY 4.0 on Zenodo; the original Linköping page states no terms); erythrocytesIDB by Marrero Fernández et al.,
Universitat de les Illes Balears (CC BY-NC-ND 4.0; related paper IEEE JBHI 19(4):1514–1525, 2015).

## JSON contract (`docs/lab/data/<id>.json`)

```jsonc
{
  "id": "celegans", "title": "…", "question": "…", "representation": "landmarks|outline|pointcloud",
  "n_points": 12, "n": 1377, "status": "ready|fetch_locally", "shapes_shown": true,
  "source": {"name": "…", "citation": "…", "url": "…", "license": "CC0", "redistributed": true},
  "classes": [{"id": 0, "name": "dead", "n": 579}, …],          // colours: class order → coral, teal, gold, lavender, slate, … (palette in lab.js)
  "computed_at": "2026-10-22T…Z", "script": "scripts/run_lab.py --dataset celegans",
  "examples": {"0": [[[x,y]×n_points]×5], "1": […]},             // aligned preshapes, image orientation
  "pca": {"explained": [10 floats], "mean": [[x,y]…], "modes": [{"sd": s, "plus": [[x,y]…], "minus": […]}×2],
          "scatter": [[pc1, pc2, class]×≤400]},
  "methods": {
    "landmarks_lr": {"accuracy": 0.802, "accuracy_sd": 0.0, "chance": 0.58, "cm": [[535,44],[229,569]],
                     "worst": [{"index": 1194, "true": 1, "pred": 0, "p": 0.95, "shape": [[x,y]…]}×3]},
    "spline_lr": {… same, plus "M": 6},
    "pointnet_probe": {"accuracy": …, "untrained": …, "cm": …, "worst": …, "onnx": "models/celegans.onnx",
                       "embed_dim": 256, "epochs": …, "seconds": …}
  },
  "engine": {                                                    // classical pipeline for live classification in JS
    "method": "landmarks_lr", "mean": [[re, im]×N], "n_components": 20,
    "pca_mean": [2N floats], "components": [[2N floats]×20], "tangent_layout": "re_then_im",
    "scaler_mean": [20], "scaler_scale": [20], "lr_coef": [[20]×C], "lr_intercept": [C]
  },
  "neighbours": {"shapes": [[[x,y]…]×≤300], "labels": [int×≤300], "indices": […]}   // aligned preshapes for k-NN display
}
```
`docs/lab/data/index.json`: `[{"id","title","status","representation","n","classes":[names],"best_accuracy","chance","license"}]`.
Floats rounded to 4 decimals (coordinates 4, weights 6). Omit `pca`, `examples`, `neighbours`, `engine` and every
`worst[].shape` when `shapes_shown` is false. Binary classes use `lr_coef` with C rows regardless (multinomial).

## The page (`docs/lab/index.html`, `docs/lab/lab.css`, `docs/lab/lab.js`, `docs/lab/engines/*.js`)

Same identity as the site (reuse `../style.css` tokens; Fraunces/Inter/JetBrains Mono; light/dark; mobile first;
16 px gutters; no horizontal scroll). Vanilla JS modules, no frameworks. Screens, matching the mockup:

1. **Lab home**: intro, dataset cards from `index.json` (status pill: ready / fetch locally / proposed), method
   chips, leaderboard table (accuracy per method, chance), footnotes on CV. Route `#/`.
2. **Dataset page** `#/<id>`: the four panels: (1) raw shapes by class (`examples`), (2) aligned cloud (PCA scatter,
   SVG), (3) mode 1 (mean ± 1 SD) with explained-variance tiles, (4) results: accuracy, confusion matrix with the
   asymmetry highlighted, the three worst mistakes drawn with their p. Method tabs switch panel 4 (and the engine
   used on screen 3). For `shapes_shown: false`: panels 1–3 are replaced by a licence note and the fetch command;
   panel 4 shows numbers only. For `status: fetch_locally`: the whole page is the fetch instructions.
3. **Classify** `#/<id>/classify`: draggable landmarks (pointer + touch; keyboard nudge) over the faint mean; load
   buttons (mean, each worst mistake, random of each class); probability bars per class; a 5-step "what happened to
   your shape" stepper with live readouts (centroid, size, rotation angle, top 2 tangent scores); nearest 3 training
   shapes by geodesic distance from `neighbours` with their labels; an engine selector.
4. **Phone**: the same routes, stacked; cards ≥ 44 px targets; dataset chips scroll horizontally.

**Engines** (`docs/lab/engines/`):
- `classical.js`: preshape (centre, Frobenius-normalise), complex align to `engine.mean` (angle of Σ conj(z)·m),
  log map, tangent vector (re then im) − `pca_mean`, project on `components`, standardise, multinomial LR
  (softmax). Must reproduce `run_lab.py`'s probabilities for the worst-mistake shapes to 1e-3 (the JSON carries
  them; the page self-checks on load and shows a quiet warning if off).
- `onnx.js`: PointNet via onnxruntime-web (cdnjs, pinned exact version), loading `models/<id>.onnx` lazily on
  first use; input `(1, N, 2)` centred points scaled by the stored `input_scale`; output embedding → the probe
  weights (`pointnet_probe.probe`: `coef`, `intercept`, `scaler_mean`, `scaler_scale`, written by run_lab.py).
  Works on phones. A "shuffle the points" button re-runs and reports max |Δembedding|.
- `pyodide.js` (desktop only, `min-width: 900px` and not touch-primary): a "Run it yourself" panel that loads
  Pyodide (cdn.jsdelivr.net/pyodide, pinned), `micropip.install`s the pure-Python wheel `docs/lab/py/shapekit-0.1.0-py3-none-any.whl`
  (built by `scripts/build_lab_wheel.sh`), loads the dataset files from `data/files/` (byte-identical copies of `data/`, enforced by tests/test_lab_site_files.py), and runs the
  classical pipeline with user-chosen M (spline), number of components and C, printing the CV accuracy. Clearly
  labelled as optional and ~20 MB.

Accessibility and honesty: real buttons/inputs; every number traceable (hover/tap shows "computed by … at …");
"[not run]" rather than blanks; no number is ever typed into the HTML.

## Scripts

- `scripts/fetch_datasets.py [--brain --gorilla --leaves --rbc --all]`: downloads with checksums printed,
  writes the npz files + per-dataset LICENSE/ATTRIBUTION notices; `--rbc` only into `data/_local/` (git-ignored).
  Must run with `python -I`-safe code (no imports from the download directories).
- `scripts/run_lab.py [--dataset id] [--skip-pointnet] [--epochs N] [--out DIR] [--force]`: everything above;
  writes JSON + ONNX; prints a results table; ≤ 15 min total on 4 CPUs (PointNet epochs tuned per dataset size;
  brain/gorilla are too small for PointNet: skip it there and say so in JSON:
  `"pointnet_probe": {"status": "skipped", "reason": "n < 100"}`). Determinism: the classical methods are
  deterministic for a given seed on any machine; the PointNet results are deterministic for a given seed
  **and** torch thread count (`--threads`, default 4), torch build and CPU (a different thread count changes
  the reduction order, the trained weights, the chosen epoch and the ONNX file). The JSON records `seed`,
  `timing.threads` and the exact command in `script`; only the wall-clock `timing` block is never reproducible.
- `scripts/build_lab_wheel.sh`: `python -m build --wheel` (or `pip wheel .`) → `docs/lab/py/`.
- Tests: `tests/test_lab.py`: registry loads every redistributed dataset with the stated shapes; `run_lab.py --dataset celegans --skip-pointnet` writes valid JSON matching the contract; the classical engine math in Python (a reference `shapekit.lab_engine.classify(shape, engine_dict)`) reproduces the LR probabilities; ONNX export round-trips (skipped without onnxruntime).

## Chapter 10 text (`docs/chapters/10-phenotype-lab.md`)

Written last, from the real JSON: what the lab is, how to read the four panels, the findings per dataset in plain
language (e.g. the live-worm asymmetry), what the three methods agree and disagree on, the limits (n = 28 for
brain), how to add your own masks, credits. Same conventions as the other chapters. Sidebar entry "10 · Phenotype
Lab" opens `lab/` (not a markdown route).

## Mockup (for reference)

Four artboards: Lab home (dataset cards, method chips, leaderboard); Dataset (four panels); Classify (drag
landmarks, probability bars, stepper, nearest worms, engine pills); Phone (stacked, on-device badge).

## Deviations

(record here)

### Computation (`scripts/run_lab.py`, `shapekit/lab_pipeline.py`, `lab_engine.py`, `lab_export.py`)

- **Module split.** The computation lives in `shapekit/lab_pipeline.py` (preprocessing, the three methods,
  display blocks, ONNX export), the JSON rounding/stripping/index in `shapekit/lab_export.py`, and the
  browser-engine reference in `shapekit/lab_engine.py` (`classify(shape_xy, engine)`, plus `explain()` which
  returns every intermediate of the stepper: centroid, size, angle, aligned preshape, ρ, tangent vector,
  scores, standardised scores, logits, probabilities). `scripts/run_lab.py` is only the CLI.
- **Binary `lr_coef` has two rows, in symmetric form.** scikit-learn ≥ 1.5 fits one row `w` for two classes
  (`p1 = sigmoid(w·x + b)`). The engine stores `[-w/2, w/2]` and `[-b/2, b/2]`, whose softmax is exactly
  the same `p1`, so `classical.js` has one code path (`engine.binary_rows: "symmetric"` marks it). The same
  holds for `pointnet_probe.probe.coef`.
- **`worst[].shape` is rounded to 6 decimals, not 4.** Those shapes are the inputs of the page's engine
  self-check; 4-decimal coordinates passed through the low-variance modes moved probabilities by up to
  8e-4 on the worms, too close to the 1e-3 tolerance. With 6 decimals the largest deviation is ~5e-5.
  Every other coordinate (examples, pca, neighbours) stays at 4 decimals; engine/probe weights at 6.
- **`spline_lr` only for ordered curves.** Brain and gorilla landmarks are anatomical points, not a curve,
  so their `spline_lr` is `{"status": "not_applicable", "reason": "anatomical landmarks are not an ordered
  curve"}`. M = 6 (open) for the worm midline, M = 12 (closed) for leaves and red cells. The spline engine
  classifies the `(M, 2)` control polygon (`engine.input: "control_points"`, `engine.M`, `engine.closed`);
  its `worst[]` entries carry the control points as `shape` and the sampled curve (60 points) as `curve`
  for drawing. The page's drag-the-landmarks screen therefore uses the `landmarks_lr` engine (the
  top-level `engine`); using the spline engine on screen would need a least-squares B-spline fit in JS.
- **Point clouds (mef).** `landmarks_lr`, `spline_lr` and the top-level `engine` are
  `{"status": "not_applicable", "reason": "points are unordered…"}`; `pca`, `examples` and `neighbours` are
  computed from the points ordered by angle around the centroid and say so in `pca.note` /
  `neighbours.note` (display only, never used for classification).
- **PointNet protocol** (superseded in Round 2, see below). Hyper-parameters as in chapter 08 /
  `make_figures_embed.py` (SupCon τ 0.07, Adam 3e-4, weight decay 1e-3, batch 64, rotation + jitter
  augmentation, embed_dim 256, 4 torch threads, seed 0; epochs: celegans 40, mef 60, leaves 60, rbc 40, typed
  once in `lab_pipeline.POINTNET_HP` / `EPOCHS` and echoed into the JSON strings). Round 1 chose the epoch on the
  scored 25 % hold-out; Round 2 chooses it on a 20 % slice of the training part and reports both numbers
  (`accuracy` on the untouched hold-out, `accuracy_select` on the slice). `accuracy_cv_val` is the tutorial's
  5-fold probe on the hold-out embeddings (and `accuracy_sd` its fold sd, ddof = 0, documented in
  `accuracy_sd_def` as *not* an uncertainty of `accuracy`); `untrained` / `untrained_select` /
  `untrained_cv_val` are the same numbers for the untrained twin. `cm` and `worst` are on the hold-out
  (`worst[].index` is the index in the full dataset, `shape` the preshape in original point order; for
  PointNet `p` and `p_engine` are both the hold-out probe's probabilities, there is no `p_cv`, which
  `worst_note` says). Extra keys: `history_select_acc` (per epoch, including epoch 0), `permutation_max_delta`
  (max |Δembedding| after shuffling one cloud's points), `onnx_info` (opset, verified max |Δ| vs torch),
  `probe` (`scaler_mean, scaler_scale, coef, intercept, classes`), `input_scale`, `input_normalisation`,
  `n_points`, `n_fit`, `n_select`, `n_val`. For celegans the 12 landmarks are fed as an unordered cloud (the
  fair comparison; `note` says so).
- **ONNX** is exported with the TorchScript exporter (`dynamo=False`; the dynamo exporter needs
  `onnxscript`, not installed), opset 17, input `points (batch, n_points, 2)` with both axes dynamic,
  output `embedding`; verified with onnxruntime to < 1e-4 at export time.
- **Classical CV detail.** The Fréchet mean and the tangent PCA are fitted once on all shapes (no labels
  involved); only the scaler and the regression are refit inside each fold (`cross_val_predict`, which
  also yields the probabilities for `worst`). `accuracy` is the pooled CV accuracy, `fold_accuracies` the
  five folds, `accuracy_sd` their sd. Brain (n = 28, 22 modes) is reported as it comes out (below chance):
  that is the n = 28 limit chapter 10 should discuss, not something to tune away.
- **Gorilla has no CV mistakes** (accuracy 1.0), so its `worst` list is empty and there is nothing to
  self-check.
- **Extra JSON keys** beyond the contract (all additive): top-level `closed`, `chance`, `seed`,
  `correspondence`, `preprocessing` (notes such as the outline resampling and, for rbc, the extraction
  summary stored by `fetch_datasets.py` in the npz), `cyclic_shift_alignment` (closed outlines),
  `example_indices`, `timing` (wall-clock seconds per method and `threads`; never inside a method block);
  `pca.sdev`, `pca.n_components`, `pca.scatter_indices`, `pca.tangent_layout`; `neighbours.aligned_to`; per
  method `status`, `protocol`, `size_information`, `fold_accuracies`, `accuracy_sd_def`, `n`, `n_components`,
  `cv`, `classifier`, `note`, `worst_note`, `cm_layout`, and `worst[].p_cv` (the full CV probability
  vector; `p` is the CV probability of the wrong class and `p_engine` the full-data pipeline's vector).
  `index.json` rows also carry `best_method`, `best_protocol`, `shapes_shown`, `methods` (status per
  method) and `computed_at`; datasets with no JSON yet are listed as `"status": "proposed"`.
- **`shapes_shown: false` (rbc)** is enforced by `lab_export.strip_coordinates`: `pca`, `examples`,
  `example_indices`, `neighbours`, `engine`, every method's `engine`/`probe`/`onnx_info`, every
  `worst[].shape/curve`, and no `.onnx` file is written (`"onnx": null`). Counts, accuracies, confusion
  matrices and the worst-mistake indices/probabilities remain.
- **`--dataset` is repeatable** and `--out DIR` redirects the output (used by the tests);
  `--threads` (default 4) sets the torch thread count (a warning is printed for any other value). The script
  exits non-zero if any dataset failed (an engine self-check above 1e-3 counts as a failure) but still writes
  the others, `index.json` and the `data/files/` copies.
- `scripts/build_lab_wheel.sh` removes the `build/` and `shapekit.egg-info/` folders setuptools leaves
  behind (unless they existed before). The wheel is about 65 KB and includes the `lab_*` modules.

### Page (`docs/lab/`, front end)

- **Layout.** The Lab is a standalone page with a top bar (back link, site title, Chapters / Phenotype Lab /
  Glossary pills, an "on-device" badge, the theme toggle) instead of the guide's chapter sidebar, as in the
  mockup. On phones the bar shows a back button, "Phenotype Lab" with the dataset as subtitle, and the
  horizontally scrolling dataset chips underneath. It reuses `../style.css` tokens and the `sa-theme` key;
  `lab.css` adds `--gold-text`, `--lavender-text`, `--slate-text` and tinted backgrounds for the pills.
- **Sidebar entry and progress.** `app.js` gains a chapter entry `{ id: '10', href: 'lab/', external: true }`:
  the sidebar, the landing card ("Chapter 10 · interactive"), the pager of chapter 09 and the glossary, the
  ←/→ keys and the `#10` route all open `lab/`. The progress counter now counts 14 items; the Lab page
  records itself in the shared `sa-visited` list under `lab`, which `app.js` maps to `10`.
- **Class colours beyond five.** The palette is coral, teal, gold, lavender, slate in class order; datasets
  with more classes (leaves: 15) get evenly spaced hues after that. Class lists longer than four are
  summarised as "15 classes (27–30 each)" on cards and in the leaderboard; the dataset page's panel 1
  still names every class in its own colour.
- **Leaderboard columns** are filled from each dataset's JSON (`methods.*.accuracy`), fetched lazily after
  `index.json`; `index.json` itself only gives the best accuracy and chance. Methods with
  `status: skipped | not_applicable` show "n/a" with the reason on hover; missing ones show "[not run]".
- **Home cards and the proposed list.** The six registry datasets are listed by name in `lab.js` (names and
  representations only, no numbers) so a dataset without a JSON still appears as "proposed"; cards show the
  first example of up to three classes once the JSON is loaded, a representation glyph before that.
- **Confusion-matrix sentence** is generated from the matrix (largest off-diagonal cell, both directions,
  "asymmetry" wording only when one rate is more than twice the other); no dataset-specific prose is in
  the page. That text belongs to chapter 10.
- **Spline mistakes** are drawn from `worst[].curve` (60 sampled points) with the control polygon faint
  behind; the classify screen always uses the top-level (landmarks) `engine`, as the computation notes say.
- **Self-check** compares `classical.js` against `worst[].p_engine` (the full-data vector; scalar accepted
  too) for the engine's own method; it is reported in the "Fitted weights, JS" note (max |Δp| ≈ 6e-5 on
  the shipped data) and raises a quiet gold banner above 1e-3. With no mistakes (gorilla) or no `p_engine`
  it says so instead of claiming a check. `lr_coef` with one row is read as a sigmoid, two or more rows as a
  softmax (the shipped symmetric two-row form therefore needs no special case).
- **ONNX runtime comes from jsdelivr**, not cdnjs: cdnjs lists onnxruntime-web 1.19.2 but serves its
  `.wasm` files with an empty body (verified with curl; the page then fails with "BufferSource argument is
  empty"). Script and wasm are both pinned to `onnxruntime-web@1.19.2/dist/`; threads are set to 1 so the
  page works on static hosting without cross-origin isolation.
- **PointNet input size.** A dragged shape has no size, so before the network sees it the cloud is given the
  dataset's typical size (RMS radius = `input_scale` pixels) and then divided by `input_scale`, i.e. the
  network sees an average-sized cell. The stepper says so. The ONNX engine is therefore not self-checked
  against `p_engine` (which used the real pixel size); the permutation test ("Shuffle the points") is.
- **Pyodide panel** runs in a Web Worker (so "Stop / clear" can really stop it) started from an inline
  blob; Pyodide 0.26.4 from jsdelivr, packages numpy/scipy/scikit-learn/micropip, the wheel from `py/`,
  data files fetched from `data/files/` (copies of `data/`) into the Pyodide filesystem and read through the registry
  (`SHAPEKIT_DATA=/data`). For celegans the files are `celegans_points.npy` + `celegans_labels.npy`, not an
  npz (the JSON may override the list with `data_files`). (Earlier builds fetched `../../data/`, which GitHub Pages does not serve; fixed.) Note that the old path was only served when the
  *repository root* is served; on GitHub Pages (rooted at `docs/`) the panel reports that the data is not
  served from this location. The panel is disabled for point clouds (no ordered points to run the classical
  pipeline on). User inputs: route (landmarks / spline control points), components, C, spline M. The
  wheel is installed with `deps=False` (its metadata lists matplotlib, which the pipeline does not need
  and which micropip would otherwise try to resolve on PyPI); numpy, scipy and scikit-learn come from
  Pyodide's own package index. Verified end to end: celegans, 10 components, C = 1 → 0.810 ± 0.016.
- **Licence-restricted datasets** (rbc, `shapes_shown: false`): panels 1–3 are one licence panel with the
  fetch and run commands and the JSON's `shapes_note`; panel 4 shows the accuracy, the C × C confusion
  matrix and the worst-mistake indices with their p, no drawings; the classify route explains that nothing
  can be classified in the browser and repeats the fetch instructions.
- **Nearest neighbours** are recomputed on every drag over `neighbours.shapes` (≤ 300 × N ≤ 150 inner
  products); the classical engine runs synchronously per animation frame; ONNX runs are coalesced so at
  most one inference is in flight and the latest drag position wins.

## Round 2 (after the review): contract changes

These apply to `scripts/run_lab.py`, the JSON and the page alike.

1. **Closed outlines are put in correspondence over cyclic shifts.** For `representation: outline` with
   `closed: true`, `prepare_points` canonicalises each outline's start point label-free: start from the
   current mean, re-root every outline at the cyclic shift (same direction) that minimises its Procrustes
   distance to the mean, recompute the Fréchet mean, repeat (3 rounds). Recorded in `preprocessing` and
   as `"correspondence": "arc length + cyclic-shift alignment to the mean"`. Applies to leaves and rbc.
2. **PointNet model selection never touches the scored hold-out.** `train_pointnet` gets
   `select_fraction=0.2`: the 75 % training part is split again (stratified, seeded) into fit / select; the
   best epoch is chosen on *select*; `accuracy` is the probe (fitted on fit embeddings) scored on the
   untouched 25 % hold-out; `accuracy_select` is the number the selection saw. JSON per PointNet method:
   `"protocol": "stratified 75/25 hold-out, seed 0; epoch chosen on a 20 % slice of the training part"`.
3. **Size is removed for every method, and measured on its own.** PointNet clouds are scaled to unit RMS
   radius per cloud (`size_information: "removed"` on every method). New method `size_lr`
   ("Size only"): log centroid size → StandardScaler → LogisticRegression, same 5-fold CV, same output
   fields (accuracy, accuracy_sd, chance, cm, fold_accuracies; no worst/shape). Every method block also
   carries `"protocol"` ("stratified 5-fold CV, seed 0, pooled out-of-fold" for the classical ones) and
   `"size_information"`.
4. `index.json` rows gain `"best_method"` and `"best_protocol"`; `best_accuracy` stays the max over ready
   methods except `size_lr` (a baseline, never a headline).
5. `run_lab.py` never overwrites a `ready` JSON with a stub (needs `--force`); `--skip-pointnet` writes to a
   separate `--out` unless `--force`; the `script` field records the exact flags; a failed engine
   self-check exits non-zero; `seconds` move under `timing` (excluded from determinism tests); diagnostics
   (`permutation_max_delta`, `max_abs_diff_vs_torch`) keep 3 significant digits; `input_scale` 6 decimals;
   `accuracy_sd` documented as ddof=0; `EPOCHS["rbc"] = 40` and every JSON regenerated by the shipped code.
6. Regeneration order, documented in README: `fetch_datasets.py` → `run_lab.py` (copies the redistributable
   `data/` files to `docs/lab/data/files/` itself) → `build_lab_wheel.sh`; dependencies in a `lab` extra.

## Round 2 deviations (page)

- **Chapter 10 is both a route and a page.** The `CHAPTERS` entry in `app.js` carries `file:
  '10-phenotype-lab.md'` *and* `href: 'lab/'`: the sidebar entry, the landing card, chapter 09's pager and
  the ←/→ keys open `lab/`, while `#10` renders the chapter text in the guide shell (with a callout
  "Open the Phenotype Lab →" under the lede and `aria-current` on the sidebar entry). `postProcess` maps a
  plain `10-phenotype-lab.md` link to `lab/`, an anchored one (`…md#section`) to `#10/section`, `../lab/`
  to `lab/`, and any `../../…` link (outside `docs/`, e.g. `ATTRIBUTION.md`) to the GitHub blob URL. The lab
  links back with "Read chapter 10" in its top nav, hero, dataset pager and footer.
- **"Size only" is shown as a baseline everywhere methods are listed**: a muted, dotted-border leaderboard
  column labelled "(baseline)", a tab labelled "(baseline)" on the dataset page with a one-line
  explanation, and a chip on the home page. The bold "best" cell comes from `index.json`'s `best_method`
  (fallback: the maximum over the three shape methods) and is never the baseline. Two footnotes under the
  leaderboard are built from the JSON: the unique `protocol` string(s) per method, and a size note that
  reads `size_information` ("every shape method sees size-normalized shapes" only when every shape block
  says `removed`; the baseline's own value, `only`, is quoted).
- **Protocols come from the JSON.** Panel 4's header and the hover title of every accuracy show the
  method's `protocol`; nothing says "5-fold" or "seed 0" unless the JSON does. PointNet shows
  `accuracy_select` ("selection split"), `untrained` and, when present, `accuracy_cv_val ± accuracy_sd`
  labelled "inner cross-validated probe on the held-out part"; the "± across N folds" line is only for
  the CV methods (N from `n_folds`). Confusion-matrix row totals are the matrix's own row sums; when they
  differ from the class sizes the caption, the row titles and the generated sentence say "held-out" and
  give the held-out total; worst mistakes are captioned "on the held-out split" (PointNet) or "out of
  fold" (CV methods), on the dataset page and on the classify screen. The classify screen adds "the engine
  on this page was fitted on all shapes, so it has seen this one" only when the full-data classical
  engine now gets a loaded mistake right.
- **Generated sentences are hedged.** The lopsidedness sentence appears only when one error rate (as a
  share of its row) is more than twice the other, and says whether the errors lean toward the larger class
  (prevalence) or against it; the overlap sentence is qualified "on these two components"; "watch the bars
  move"; the shuffle result states the page's 1e-4 tolerance; the self-check names its 1e-3 tolerance as
  the page's; "the N most confident mistakes" uses the real N; the Pyodide size is "roughly 20 MB, an
  estimate"; the engine note counts the engine's numbers instead of "a few hundred".
- **Class colors beyond five** keep the mid hue for strokes, but text uses per-theme variables
  `--cx5 … --cxK` written into a `<style>` element by `ensureExtraColors`: the lightness of each hue is
  searched (down from 45 % in light, up from 50 % in dark) until the contrast against the darkest/lightest
  tinted background the text can sit on reaches 4.6:1. Filled controls (CTA, primary button, active nav
  pill, pressed engine pill, the last stepper numeral) use a new `--teal-fill` (#0F766E) with white text
  (≥ 5:1 in both themes) instead of `--teal`.
- **Wide tables**: `scrollable()` turns the leaderboard wrapper and every confusion-matrix wrapper into a
  focusable, labelled scroll region with an edge fade and a "scroll sideways" hint that appear only while
  the content overflows (ResizeObserver). Matrices with more than five classes use a compact mode (24 px
  cells, column heads written vertically, binomials abbreviated to "U. glabra" with the full name in the
  cell title). The `.panels` grid no longer stretches panel 1 to panel 2's height.
- **Touch and keyboard.** Every control is at least 44 px tall on narrow or coarse-pointer devices (chips,
  tabs, pills, buttons, inputs, the theme toggle, table and breadcrumb links). Dragging is handled on the
  stage SVG: a press picks the nearest landmark within 22 screen pixels (the invisible hit circles are
  also sized to 22 screen px via `getScreenCTM`), so a finger need not land on a 7 px dot and crowded
  clouds (mef 120 points, leaves 100) are draggable. Stages with more than 20 points use a roving
  tabindex (one tab stop; `[` and `]` move between points, arrows nudge). The "mean shape" SVG label and
  panel 3's in-figure labels became HTML legends so they never collide with the drawing.
- **Species names** are rendered with a lowercase epithet (`speciesCase`, binomials only; "Populus sp.")
  in the UI; the JSON keeps the source's labels. Titles wrap tokens containing "/" (LMNA−/−) in a
  `nowrap` span. Backticks in JSON strings render as `<code>`. Dates and thousands use `en-US`.
- **Data line** on every dataset page: source name linked to `source.url`, the citation, the license
  linked to `source.license_url` when the JSON has it, else to a small lookup of license names → canonical
  license URLs (a table of URLs, not of results), "modified: …" from `preprocessing`, a link to
  `ATTRIBUTION.md` on GitHub, and the script/date. The license-restricted panel (rbc) no longer repeats
  `shapes_note` and words the restriction as a precaution (ND/NC), not as a legal claim.
- **ONNX engine** is unchanged in arithmetic: it always produced a unit-RMS-radius cloud, which is now
  exactly the training preprocessing (`input_scale` = 1, `size_information: removed`); the stepper and the
  engine note describe it as "size removed, as in training" when the JSON says so. Because the
  preprocessing now matches, the ONNX route is self-checked too: once the model has loaded, `selfCheck()`
  embeds each stored PointNet mistake (`shape`, original point order) and compares the probe's
  probabilities with `p_engine` (4 decimals in the JSON); the result is written into the PointNet engine
  note and a gold banner appears above 1e-3. Shipped data: max |Δp| 3.0e-5 (celegans), 2.7e-5 (mef),
  5.9e-5 (leaves).
- **Pyodide panel** prints the scikit-learn version it runs, the pooled out-of-fold accuracy (the page's
  kind of number) and the mean of folds, and explains the remaining differences (library version, mean
  and PCA refit per fold, no further start-point alignment; the dataset's `correspondence` string is
  quoted). It still runs on the data as the registry loads it.
- **Guide CSS**: `.table-wrap` is `position: relative` so KaTeX's absolutely positioned MathML copies
  inside table cells stay inside the scroll clip (they widened chapter 10 to 485 px on phones).
- **Wording**: the footer and the home lede say "no result is typed in by hand" (parameters of the page's
  own checks, such as tolerances, are named as the page's). American spelling in all lab strings
  (`labeled`, `standardize`, `colors`, `centered`, `license`, `gray`, `neighbors`); JSON keys such as
  `neighbours` are untouched. `README.md`, `ATTRIBUTION.md` and the chapters were not edited by the page
  round.
- **Test**: `tests/test_lab_js_syntax.py` runs `node --check` on every `.js` under `docs/` (skipped when
  node is absent).

## Round 2 deviations

Implemented by the computation round that followed the review; every number below was regenerated by the
shipped code (`python scripts/run_lab.py`, 8.7 min for all six datasets on an idle 4-CPU machine, seed 0,
4 threads; `bash scripts/build_lab_wheel.sh`; `python -m pytest -q`: 133 passed).

- **Cyclic-shift alignment is label-free *and* data-driven; it is applied to rbc and deliberately not to
  leaves.** `prepare_points` always runs `align_cyclic_shifts` (3 rounds of "re-root every outline at the
  shift closest to the Fréchet mean, recompute the mean") and records `fraction_near_zero`: the share of
  outlines whose best shift is within 10 % of the perimeter of where the source started. When at least 50 %
  are already there (`START_LANDMARK_MAJORITY`) the source's start point is a landmark and is kept. Measured:
  rbc 24 % (uniform: the contour tracer starts at the top-most pixel row, meaningless for a cell) → applied,
  2224 of 2261 outlines shifted, mean geodesic distance to the mean 0.147 → 0.123, `landmarks_lr` 0.786 →
  **0.917**, `spline_lr` 0.794 → **0.920**; leaves 69 % (every outline starts at its lowest image point, the
  leaf base, in photographs that share an orientation) → kept, numbers unchanged (0.793 / 0.825). Forcing the
  alignment on the leaves gives 0.739 / 0.750 (any subset of the shifts hurts: 0.754 with only the small ones,
  0.761 with only the large ones), because whole species are shifted to match a generic mean (all 30 *Quercus
  robur* flipped base ↔ tip, all 29 *Acer platanoides* moved by 10–40 points): the source's start is real
  correspondence and the mean-driven one blurs it. The JSON carries the decision (`correspondence`,
  `cyclic_shift_alignment.applied`, the fractions, the criterion) and the `preprocessing` note says which way
  it went. `lab_engine.classify` does not search shifts: the shipped `worst[].shape`, `neighbours` and
  `examples` are the re-rooted outlines, and the engine dict records `correspondence`.
- **PointNet selection** (`train_pointnet(select_fraction=0.2, normalize_size=True)`): the 75 % training part
  is split again, stratified, seed 0, into fit (80 %) / select (20 %); the network trains on fit only, the
  probe is fitted on the fit embeddings, the epoch is chosen on select, `accuracy` is that probe on the
  untouched 25 % hold-out and `accuracy_select` is the number the selection saw. The chapter-08 path
  (`select_fraction=0`, `make_figures_embed.py`) is unchanged and verified identical to the previous code
  (same split, losses, accuracies and embeddings on a 300-cell run). Results, hold-out / select / untrained
  (hold-out, epoch 0) / `accuracy_cv_val`, best epoch of N: celegans 0.878 / 0.865 / 0.873 / 0.887, 10 of 40;
  mef 0.917 / 0.939 / 0.920 / 0.887, 4 of 60; leaves 0.955 / 0.970 / 0.936 / 0.900, 60 of 60; rbc 0.908 /
  0.915 / 0.905 / 0.915, 38 of 40. Round-1 hold-out numbers (epoch chosen on the same hold-out) were 0.887,
  0.953, 0.982, 0.935. Two honest consequences: on mef the select slice picked epoch 4 and the trained
  network does not beat its untrained twin on the hold-out (0.917 vs 0.920); on the leaves the best epoch is
  the last one (60), so more epochs might help, but the budget and "every JSON by the shipped code" were
  kept. Round 1's `history_val_acc` is replaced by `history_select_acc`; the hold-out is never evaluated
  during training.
- **Size removed for PointNet** (`input_normalisation: "per_cloud_rms"`): every cloud is centred and divided
  by its own RMS radius before the shared `input_scale`, which is therefore 1.0 (computed, not assumed;
  rounded to 6 decimals). The ONNX input contract is "centred, divided by the cloud's own RMS radius, then by
  input_scale"; the page's existing ONNX preprocessing (give the shape RMS radius = `input_scale` px, divide by
  `input_scale`) produces exactly that, so the ONNX engine is now also checkable against `p_engine` (the real
  size no longer matters): verified with onnxruntime, max |Δp| 3.0e-5 (celegans), 2.7e-5 (mef), 5.0e-5
  (leaves), hold-out accuracies and confusion matrices reproduced exactly. The flag travels with the weights
  (`PointNet.per_cloud_norm` buffer). Untrained-twin baselines dropped accordingly (celegans 0.881 → 0.873).
- **`size_lr`** ("Size only"): log centroid size of the points the classical route uses (landmarks, the
  100-point outline, or the raw cloud for mef) → StandardScaler → LogisticRegression, same folds; fields as the
  other CV blocks plus `feature`, `size_median_by_class`, `size_information: "only"`; no `worst`, no engine.
  celegans 0.747 (reproduces the content review's 0.7466), mef 0.779, brain 0.786 (n = 28; size beats shape),
  gorilla 1.000, leaves 0.343, rbc 0.724 (= chance: the extraction's area filter is per database). `index.json`
  `best_accuracy` / `best_method` / `best_protocol` exclude it (`lab_export.BASELINE_METHODS`).
- **Headlines after Round 2:** celegans 0.878 (pointnet), mef 0.917 (pointnet), brain 0.393 (landmarks),
  gorilla 1.000 (landmarks), leaves 0.955 (pointnet), rbc **0.920 (spline_lr)**: on the red cells the classical
  route now beats PointNet (0.908) once the start points are in correspondence and size is removed.
- **`protocol` / `size_information` / `accuracy_sd_def`** on every method block; the PointNet `accuracy_sd`
  is kept for the page but its definition string says it is the sd of the 5-fold probe inside the hold-out,
  not an uncertainty of `accuracy`.
- **`timing`**: a top-level block (`seconds` per method and `total`, `threads`, a note) replaces the per-method
  `seconds`; nothing else in a JSON changes between runs on the same machine.
- **Diagnostics**: `permutation_max_delta` and `onnx_info.max_abs_diff_vs_torch` keep 3 significant digits
  (shipped: 2e-7 … 1e-5, no longer "0.0"); `input_scale` 6 decimals.
- **run_lab.py guards**: a ready JSON is never replaced by a `fetch_locally` stub without `--force` (the row
  prints "kept"); `--skip-pointnet` / `--epochs` refuse the default output folder without `--force`
  (`argparse` error, nothing written); `script` records the exact flags (`--skip-pointnet`, `--epochs N`,
  `--seed S` and `--threads T` when non-default, `--force`); a self-check above 1e-3 raises
  `SelfCheckError`, the dataset counts as failed and the exit code is 1; `--threads` other than 4 prints a
  warning; `torch.use_deterministic_algorithms(True, warn_only=True)` is set. The script copies every
  top-level `data/` file to `docs/lab/data/files/` (refusing names that look like restricted data) so the
  Pyodide panel's copies can never drift (`tests/test_lab_site_files.py` derives its list from `data/`).
- **rbc provenance** (content review 8 and 20): `fetch_datasets.py --rbc` stores its extraction summary in
  the npz (`preprocessing`: 80 fields, one IDB2 field without a circular mask, 9 components dropped by the
  area filter, "the class counts are this extraction's") and `run_dataset` copies it into the JSON's
  `preprocessing` (`lab_pipeline.dataset_notes`). The npz was re-extracted by the shipped code
  (points byte-identical to the previous extraction) and rbc was trained with `EPOCHS["rbc"] = 40`.
- **fetch_datasets.py**: the rbc md5 is enforced with and without `--skip-download`; `download()` verifies a
  finished download and deletes it on a mismatch; HTTP 416 on a resumed `.part` promotes and verifies the
  complete part; a zip is validated entirely before anything is written; the unpack is marked complete by
  `data/_local/erythrocytesIDB_v2/.unpacked-ok` (an interrupted unpack is redone); the `.gitignore` guard is a
  real line match (`gitignore_covers`); `io` / `os` imports removed. All tested with a local Range-capable
  HTTP server (`tests/test_lab_fetch.py`).
- **Tests added**: wheel members vs the tree (`tests/test_lab_wheel.py`); shipped JSON/ONNX/index consistency
  and the ONNX-reproduces-the-probe check (`tests/test_lab_shipped.py`); the licence guard end to end on a
  synthetic `shapes_shown=False` dataset through `run_dataset` + `finalize`, classical and PointNet/ONNX
  branches; overwrite guards, partial-run refusal, self-check failure exit, `script` flags, data-file copy
  refusal; `align_cyclic_shifts` on synthetic outlines with random starts; the `size_lr` number.
- **Cleanups**: `resolve_classes` no longer searches npz files or a meta file (every registered dataset names
  its classes; the npz lookup lives on only as `dataset_notes`); `__all__` lists everything public; the
  hyper-parameters are typed once (`POINTNET_HP`, `VAL_FRACTION`, `SELECT_FRACTION`) and the JSON strings are
  built from them; point clouds get their display geometry from `shape_geometry` (no classifier, no
  assertion); `worst_note` states the two meanings of `p_engine` (classical: full-data engine; PointNet:
  the hold-out probe). Not changed, on purpose: the top-level `engine` still duplicates
  `methods.landmarks_lr.engine` (it is the contract the page reads; ~250 KB on the largest file) and
  matplotlib stays a core dependency (the figure scripts need it; the Pyodide panel installs the wheel with
  `deps=False`). The `lab` extra in `pyproject.toml` lists torch, onnx, onnxruntime, pyreadr, xarray, rdata,
  Pillow, contourpy and build.
- **Budget**: 8.7 min measured for the full run (classical ~3 s; PointNet celegans 19 s, mef 87 s, leaves
  158 s, rbc 257 s), inside the 15-min target; the fit part is now 60 % of each dataset, which is why the
  PointNet steps are faster than in Round 1.
