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

## Round 3: raw images (option 3: samples in the repo + an images fetch option)

**Which images may be redistributed.** Only datasets whose licence for the *images* is verified:
`celegans` (BBBC010, CC0) and `mef` (Micropattern_MEF_LMNA_Image, MIT © 2020 Kyu Sang Han; the MIT
notice must accompany the copies). `leaves`: the photographs' only licence statement is a Zenodo record
whose uploader is unverified, so **no leaf photographs in the repo** unless that is resolved (fetch only).
`rbc`: CC BY-NC-ND, **never** in the repo (fetch only; it is already fetched into `data/_local/`).
`brain`, `gorilla`: coordinate data, no images exist.

**Samples** (`docs/lab/data/images/<id>/`, small, web-ready):
- `celegans`: one image from a live well (B07, ampicillin-treated) and one from a dead well (A23,
  untreated), the brightfield channel the tutorial used (`w2`), contrast-stretched to 8-bit PNG,
  ≤ 900 px wide; plus the matching BBBC010 ground-truth masks if available; each with an overlay computed
  by `shapekit` from the real pixels: worm mask → skeleton → 12 arc-length landmarks (the chapter 01
  pipeline on real data). Provenance per file (BBBC010 file name, well, channel).
- `mef`: one wild-type and one LMNA−/− field (`xy001c1`, actin channel c1, nuclei c2 optional as a
  second colour), 8-bit PNG ≤ 900 px; overlay = the outlines that `mef_data.npy` (the tutorial's file,
  keys `wildtype_xy001c1_*` / `lmna_xy001c1_*`) holds for that field, **only if** they register on the
  image (verify visually; if they do not, draw outlines traced by `shapekit.contours` from the tutorial's
  `xy001c1_instances.h5` mask for the wild-type field and say so).
- Each folder has `manifest.json`:
  `{"id","images":[{"file","alt","caption","width","height","channel","source_file","class",
  "overlays":[{"kind":"outline|landmarks","points":[[x,y],…],"closed":bool,"label":str}],
  "processing": "what was done to the pixels"}], "credit":{"title","authors","url","license",
  "license_url","notice_file"}}` with overlay coordinates in the PNG's pixel frame (x right, y down).
- Licence files next to the samples: `docs/lab/data/images/mef/LICENSE` (full MIT text, Han's copyright)
  and `docs/lab/data/images/celegans/CREDIT.md` (CC0 statement + the citation BBBC010 requests).
- Budget: ≤ 2.5 MB for all samples together.

**Fetch option** (`scripts/fetch_datasets.py --images celegans|mef|leaves|rbc|all`): downloads the original
image sets into `data/_local/images/<id>/` (git-ignored), checksums printed, never committed:
BBBC010 images + ground truth from bbbc.broadinstitute.org; a bounded MEF subset (`--max-files N`,
default 40, from the GitHub repository's raw files or its zip); the Swedish Leaf per-species zips from the
Linköping page (`--species` subset allowed); for rbc, point to the already-extracted erythrocytesIDB tree.
Each prints the licence and the attribution it requires.

**Script**: `scripts/make_sample_images.py` builds the samples and manifests from the fetched originals
(deterministic), and also writes `docs/figures/01_real_pipeline.png` (the chapter 01 pipeline on one real
BBBC010 worm: image → ground-truth mask → skeleton → 12 landmarks → spline), in the SPEC visual identity.

**Page**: every dataset page gets a "0 · From pixels" panel before panel 1: the sample images with a
toggle for the overlay (outline/landmarks), caption, processing note and the credit line with licence
link; for datasets without redistributable images it says why and gives the fetch command.

**Tests**: manifests valid; every image file exists, is a PNG within budget; overlay points inside the
image bounds; licence/credit files present; no `rbc` or `leaves` image under `docs/`.

## Round 3 deviations (page)

- **Which datasets have images is registry metadata in `lab.js`**, not a probe: `REGISTRY[].images` is
  `manifest` (celegans, mef), `unverified` (leaves), `nc_nd` (rbc) or `none` (brain, gorilla). Only the
  `manifest` entries ever request `data/images/<id>/manifest.json`, so the other pages make no request that
  could 404. The manifest is fetched in parallel with the dataset JSON and awaited before the page renders,
  so panel 0 never pushes the page down; `<img>` carries the manifest's `width`/`height` and
  `loading="lazy"`. A missing or empty manifest shows a one-line note (the file, the HTTP error, and
  `scripts/make_sample_images.py` / `fetch_datasets.py --images <id>`) instead of breaking the page.
- **Panel 0 is a full-width panel** at the top of the `.panels` grid (also above the license panel on rbc,
  above the fetch box on a `fetch_locally` page and on a dataset without JSON). Images sit in an auto-fit
  grid (`minmax(min(100%, 360px), 1fr)`): two columns on desktop (celegans' four images as 2 × 2), stacked on phones.
- **Registration**: the overlay is an absolutely positioned SVG (`inset: 0`, 100 % × 100 %) over the
  `<img>`, `viewBox="0 0 width height"` (the PNG's pixel frame), `preserveAspectRatio="none"` (the box has
  the image's aspect ratio already, so this only guards against sub-pixel rounding). Strokes use
  `vector-effect: non-scaling-stroke`, so line widths stay constant on screen at any image size; landmarks
  are zero-length round-capped path segments sized in image pixels (max(width, height) / 120), so the 15
  dense worms of B07 stay readable on a phone and the dots grow in the larger view. Each path has a dark halo under the class
  color so it reads on brightfield and on fluorescence. Verified in Chromium: an overlay point's screen
  position (via `getScreenCTM`) equals the image box's `left + x·w/W, top + y·h/H` to < 1e-13 px at
  1280 × 800 and 390 × 844, also after resizing a loaded page between the two, and in the larger view.
- **Overlay colors by class**: `image.class` (or an overlay's own `class`) is matched to `classes[]` by id
  or by a loose name key (case, the Unicode minus and punctuation ignored: `LMNA−/−` = `lmna-/-`,
  `wild-type` = `wildtype`); an unmatched class is drawn in gold and the legend says "class not matched".
  The legend lists the classes present in class order and the overlay kinds present (outline line,
  landmark dots).
- **Toggle**: one per image, off by default (the raw pixels first), `aria-pressed`, with a constant label
  that names what the manifest draws ("Show outlines and landmarks" for celegans, "Show outlines" for mef)
  rather than the literal "Show outline/landmarks". The overlay SVG is `hidden` while off and carries
  `role="img"` with the overlays' labels while on.
- **Larger view**: each image is a button ("View larger: <alt>") that opens one shared `<dialog>`
  (`showModal`; Escape or Close or a backdrop click closes it; focus returns to the image button). The
  dialog has its own overlay toggle, starting in the state of the image's toggle, and the processing note.
  The image is sized `min(100%, 1.5 × PNG width, (100dvh − 190px) × aspect)`, so it always fits the
  viewport.
- **Caption block** per image: class name in its text color, the caption, the `processing` note and
  `source_file` / `channel` in small text. **Credit line**: title → `url`, authors, license →
  `license_url` (fallback: the page's license-URL lookup), and the notice file linked on GitHub
  (`…/blob/main/docs/lab/data/images/<id>/<notice_file>`), followed by the fetch command for the full
  image set. Credit and caption links are inline text links, so they are not padded to 44 px; the toggle
  and image buttons are ≥ 44 px on phones.
- **No-image wording**: brain/gorilla "This dataset is landmark coordinates; there are no images." (plus:
  the R package ships the coordinates only); leaves: the outlines are redistributed under their own
  license, "the photographs' license could not be verified, so they are not redistributed", plus
  `python scripts/fetch_datasets.py --images leaves`; rbc: CC BY-NC-ND 4.0, web-ready samples would be
  modified copies, which the no-derivatives term does not allow, so as a precaution no image is in the
  repository, plus `--rbc` first and then `python scripts/fetch_datasets.py --images rbc`. The commands are
  the ones this spec defines; the page does not check that the installed script already has `--images`.
- **Footer shift fixed**: `.lab-main` now has `min-height: calc(100dvh - 64px)`, so replacing "Loading…" with
  the page no longer drags the footer up into the first screen and back (a 0.096 layout shift on every
  desktop page, home included, that predates Round 3). What remains is the web-font swap (≤ 0.019).

## Round 3 deviations (data and images)

Implemented by `scripts/fetch_datasets.py --images …` and `scripts/make_sample_images.py`; tested by
`tests/test_lab_images.py` (`python -m pytest -q`: 149 passed then; 163 after the Round 3 fixes).

- **Fetch flags.** `--images` takes one or more of `celegans mef leaves rbc all` (`--images celegans mef`);
  `--all` keeps its old meaning (the four coordinate datasets) and does not fetch images. `--species` is a
  list of Linköping species *numbers* (`--species 1`, `--species 5,7`, `--species all`), default `5,7` (the
  two smallest zips, 74 + 99 MB), not a count. `--max-files` counts TIFFs and fetches whole fields (c1 + c2),
  rounding an odd number down; a 150 MB cap stops the MEF subset early whatever `--max-files` says (the
  default 40 files is ~107 MB). Every image set refuses to run unless `.gitignore` covers `data/_local`,
  prints size/md5/sha256 per file and writes `data/_local/images/<id>/SOURCE.txt` (licence notice +
  checksums). The zips are unpacked with the existing `safe_extract_zip` (path-escape guard) behind an
  `.unpacked-ok` marker; the BBBC010 zips also carry `.svn/` metadata, which is unpacked but not counted.
- **Checksums.** BBBC and Linköping publish none: BBBC010's three zips are pinned to the md5s of the
  2026-10-23 download (and to the server's sizes); leaf zips are pinned by size, and species 1 also by md5
  (the only one downloaded here). MEF files are Git LFS objects: `raw.githubusercontent.com` serves the LFS
  *pointer*, whose sha256 and size every file from `media.githubusercontent.com` is checked against (a
  mismatch deletes the file). Both are read at a pinned commit (`7cbe667…`). The GitHub API listing
  (`api.github.com/…/git/trees/<commit>?recursive=1`) is tried first; through this session's proxy it
  answered 403, so the script fell back to probing the pointers of the candidate paths in a fixed order
  (the tutorial's two fields, then `xy001`, `xy002`, … round-robin over the 12 folders): same selection.
- **The tutorial's MEF files are fetched too.** `--images mef` also downloads Shape Analysis 101's
  `mef_data.npy` and `xy001c1_instances.h5` (commit `bfccc0c…`, md5-pinned) into
  `data/_local/images/mef/tutorial/`, because the overlays need them. `mef_data.npy` is a pickled dict: it
  is only read by `make_sample_images.py`'s restricted unpickler (numpy `_reconstruct` / `ndarray` /
  `dtype` only). The tutorial's two fields were identified by sha256: `wildtype/raw_images/xy001c*.tif` =
  `Control/LMNA++1/xy001c*.tif`, `lmna_deficient/raw_images/xy001c*.tif` = `Circle/LMNA--1/xy001c*.tif`
  (so the tutorial's two classes come from different micropattern conditions; the captions say so).
- **Fetched in this round**: `--images celegans` (3 zips, 72 MB: 200 TIFFs, 100 foreground masks, 1,407
  per-worm masks), `--images mef --max-files 8` (4 fields, 21.3 MB, + the 2 tutorial files) and
  `--images leaves --species 1` (leaf1.zip, 251 MB, 75 TIFFs). Nothing was unreachable except the GitHub
  API (above). `--images rbc` downloads nothing and reports the existing erythrocytesIDB tree.
- **BBBC010 channel.** The BBBC010 page says "Channel 1 = brightfield; channel 2 = GFP", but in the files
  `w2` is visibly the brightfield image (bright well, dark worms) and `w1` the fluorescence one; the samples
  use `w2` as the contract says and the processing note records the discrepancy. The tutorial clone's
  TIFFs and `B07_06_ground_truth.png` are byte-identical to the BBBC010 downloads.
- **C. elegans overlays** come from all 27 per-worm ground-truth masks of the two wells (15 in B07, 12 in
  A23): `outline` (longest traced contour, resampled to 100 points, closed) and `landmarks` (12, open) per
  worm, also attached to the foreground-mask PNGs (same frame). Only each mask's largest 8-connected
  component is used: `B07_11_ground_truth.png` contains one stray pixel, which made the greedy skeleton
  walk jump 114 px; with it removed all 27 worms pass the check that no walk step exceeds 3 px (a worm
  failing it would keep its outline and lose its landmarks, with the reason in `processing`). The images are
  696 × 520, so no resize (scale 1); the fallback with only the tutorial clone (`--tutorial-dir`) gives
  the same PNGs and figure but overlays for worm B07_06 only, and says so.
- **MEF registration was measured, not only looked at.** Every `mef_data.npy` outline point (row, col →
  x, y) was sampled on the distance map to the edges of an Otsu mask of the σ = 1 px smoothed actin image:
  median 1.5 px for both fields (p90 31.5 px wild-type, which includes thin protrusions the threshold
  misses; 6.5 px LMNA−/−), against 106 / 143 px with the axes swapped; and 0.5 px to the edges of
  `xy001c1_instances.h5` (wild-type). They register (criterion: median ≤ 3 px, recorded in each image's
  `registration` block), so the h5-tracing fallback (implemented) is not used. Visually confirmed on full-
  resolution overlays. `mef_data.npy` lists most cells twice a few px apart, and one wild-type cell also as
  "with protrusion" (4), "without" (5) and "protrusion only" (9); outlines lying on an already kept lower
  key (median point-to-loop distance < 3 px) are not drawn: 23 → 11 (wild-type), 18 → 9 (LMNA−/−), keys
  listed in `processing`. The points are unordered, and the page draws `landmarks` as a connected path, so
  MEF overlays are `kind: "outline"` only, with the points put in loop order for drawing (nearest neighbour
  + 2-opt; deterministic).
- **MEF pixels**: actin channel only (nuclei not shown, to keep the overlay colours readable), 1st–99.7th
  percentile stretch (the brightest rounded LMNA−/− cells clip), Lanczos resize 1280 × 1024 → 900 × 720
  (scale 0.703125; overlay x' = (x + 0.5)·s − 0.5). PNG sizes: celegans 129 + 123 + 3 + 3 KB, mef 303 +
  194 KB; with manifests, LICENSE and CREDIT.md the folder is 0.92 MB (budget 2.5 MB).
- **Manifest extras** (additive to the contract): top-level `title`, `generated_by`, `coordinates`;
  per image `class_id`, `source_md5`, `source_archive` and `well` (celegans), `scale` and `registration`
  (mef); per overlay `class` (an id); `credit.citation` (celegans), `credit.copyright` and
  `credit.overlay_credit` (mef: the outlines are the tutorial's, BSD-3). `class` is the registry's class
  name (`dead`/`live`, `wild-type`/`LMNA−/−`). Point lists are written one per line (114 KB instead of
  296 KB for celegans). No timestamps, so the files are byte-identical between runs (tested).
- **`docs/figures/01_real_pipeline.png`** (1600 × 520) uses worm B07_06 (the tutorial's example worm), a
  square crop around it, the `make_figures.py` identity helpers, and an 11 pt caption (the shared
  `caption()` helper is 10.5 pt). It is listed in `docs/figures/README.md` as made by
  `make_sample_images.py`, not `make_figures.py`.

## Round 3b: the MEF confound (verified)

Field-by-field registration of the tutorial's `mef_data.npy` outlines on the original images
(kukionfr/Micropattern_MEF_LMNA_Image @ 7cbe667) matched 12 of 12 sampled wild-type fields to
`Control/LMNA++1/xyNNN` (unpatterned) and 12 of 12 sampled LMNA−/− fields to `Circle/LMNA--1/xyNNN`
(circle micropattern), each compared with the 12 folders' images at the same xy number, no other folder
registering (evidence table: scratchpad `mefv/full.jsonl`; after the Round 3 fixes 14 of 14 per class, see
below). [Inference] the whole tutorial dataset is these two folders (86 + 86 fields not sampled). So in this
dataset genotype is confounded with substrate (circle pattern vs free spreading) and with whatever else
differs between the two folders (culture dish, day, imaging session); the repository does not say what its
folder suffixes 1 and 2 mean, so this is not called a replicate confound. Outline-only evidence (after the
Round 3 fixes, circularity on loop-ordered points): circularity median 0.88 (LMNA−/−) vs 0.32 (WT);
circularity + axis ratio alone classify at 0.94 (5-fold CV), about PointNet's 0.917 hold-out (different
protocols). (Round 3b as first written quoted 0.11 and 0.92, from angle-ordered points.)

Consequences (all must be reflected):
- Registry: `LabDataset` gains `caveats: tuple[str, ...]`; MEF's caveat states the confound with the
  folder names and the 24/24 evidence, and that the lab's MEF accuracy cannot be attributed to genotype.
  MEF's `question` becomes "Can a classifier tell these two groups of fibroblasts apart from outlines alone,
  and what is it actually seeing?" Its `title` stays but gains "(genotype confounded with substrate)" in the
  page display. `run_lab.py` writes `caveats` into the JSON and `index.json` (`has_caveat: true`).
- A new display-only number in mef.json: `roundness_baseline` = 5-fold CV accuracy of circularity + axis
  ratio (StandardScaler + LogisticRegression, seed 0), computed by run_lab.py, shown next to the methods
  with the label "roundness only (two hand-made numbers)". It is a diagnostic, not a method.
- Page: a prominent caveat box on the MEF dataset page (above panel 0) and a marker + footnote on the
  leaderboard row and the home card; the MEF sample image captions say which original folder each field
  came from.
- Text: chapters 00 (dataset description), 08 (results on MEF), 10 (MEF findings and the lesson), ATTRIBUTION
  (MEF provenance: the two folders), README map row if it mentions genotype classification.
- Deconfounded comparison (same substrate, both genotypes) is feasible from the original repository but
  needs segmentation; it is out of scope for now and is mentioned as an open item.

## Round 3 fixes (after the Round 3 content and QA reviews)

- **Circularity on loop-ordered points.** `shapekit.contours.order_loop(points)` (new, exported from
  `shapekit`, documented, tested on a twisted five-armed star where angle order fails: angle order gives
  6.8× the true perimeter, `order_loop` recovers the outline exactly) is the nearest-neighbor walk + 2-opt
  that `make_sample_images.py` used for drawing (`cycle_order`, now removed there), vectorized over `k` but
  with the same first-improvement scan, so it returns the same order (checked on 70 clouds; the sample
  manifests' overlay points are byte-identical). `roundness_baseline` now measures circularity on
  `order_loop` of each cloud and records `"point_order": "nearest-neighbor + 2-opt loop"`; the display blocks
  (PCA, examples, neighbors) keep angle order. `run_lab.py --dataset mef --force`: every method block, the
  engine and the display blocks are identical to the previous file; only `caveats`, `roundness_baseline`,
  `computed_at` and `timing` changed (`index.json`: only `computed_at`). New roundness numbers: accuracy
  **0.940** (was 0.920), fold SD 0.0057, folds 0.9292 / 0.9417 / 0.9417 / 0.9417 / 0.9458, matrix
  [[562, 38], [34, 566]], median circularity wild-type **0.321** (was 0.1135), LMNA−/− 0.8817; axis ratio
  unchanged (1.6438 / 1.173). The roundness diagnostic now scores a little above PointNet's 0.917 hold-out;
  the texts call the three numbers level (different protocols), not a ranking.
- **Confound wording.** "Confounded with replicate" is dropped everywhere: the matched folders are
  `Control/LMNA++1` and `Circle/LMNA--1`, both suffixed 1, and the repository README does not define the
  suffix. The texts say genotype is confounded with substrate and with whatever else differs between the
  two source folders (culture dish, day, imaging session).
- **Evidence: 28 fields, not 26.** The QA re-verification added fields xy005 and xy077, each in *both*
  classes, so the sample is 14 field numbers × 2 classes = 28 fields (14 + 14), and 86 + 86 = 172 are
  unsampled. I fetched xy005c1 and xy077c1 from all 12 folders (sha256 against the LFS pointers) and ran the
  original registration script (`scratchpad/mefv/full.py`, output `mefv/extra.jsonl`), so all 28 rows
  use one method against all 12 candidates. With the 99.5th-percentile clip: matching folder 1.0 to 6.8 px
  (wild-type xy005 6.80 px; the QA's 6.40 px is its own edge definition and is quoted separately in
  ATTRIBUTION as an independent check, 2.0 to 6.4 px against 57 px or more in four folders), every other
  folder ≥ 29.9 px; plain Otsu: wild-type xy054 647 px (one saturated 1,856-pixel bright spot sets the
  threshold at 1631 against 335 with the clip), the other 27 fields 1.2 to 7.4 px; inside/outside median
  intensity ratio 1.3 to 6.8 in the matching folder (1.34 on wild-type xy005; 1.38 to 6.78 on the original
  24), 0.9 to 1.1 elsewhere. Cells of this subset from verified fields: 196 (95 wild-type, 101 LMNA−/−).
- **"100 fields per folder" is verified, not inferred.** The pointer clone of the repository at `7cbe667`
  (`data/_local/images/mef_verify/repo`, `git ls-tree`) lists exactly `xy001c1.tif` … `xy100c1.tif` in each
  of the 12 folders, so the texts state it with that source rather than hedging it to the sampled numbers.
- **MEF image license.** ATTRIBUTION, the MEF manifest credit (`license_note`, shown on the page's credit
  line) and the fetch notice add "[Unverified] whether the author intended the MIT software license to
  cover the image data". `docs/lab/data/images/mef/LICENSE` stays the verbatim upstream MIT text (the
  script refuses to write anything else). The paper the README names is cited as a courtesy from PubMed
  (PMID 33424024): Phillip, Han, Chen, Wirtz & Wu, Nature Protocols 16, 754–774 (2021),
  doi:10.1038/s41596-020-00432-x; the README calls it a 2020 paper.
- **Chapter 01.** "human-corrected" is kept only for BBBC010's foreground masks (the page's own words);
  the per-worm masks are "ground-truth per-worm masks" in the chapter, `docs/figures/README.md` and the
  regenerated figure caption. scikit-image 0.26 `skeletonize` gives 104 pixels on `B07_06_ground_truth.png`
  (1,212 pixels, one component), `thin` 102, `shapekit` 114 with 2 tips; the chapter now says the mask is
  not the cause.
- **Wide figures on phones** (`docs/app.js` `markWideFigure`, `docs/style.css` `.fig-wide`): a guide figure
  whose natural size is ≥ 1200 px wide and ≥ 2.5 × as wide as tall is wrapped, once loaded, in a focusable
  sideways-scrolling frame (`role="region"`, labeled) with a visible hint; below 700 px the image keeps a
  760 px width (about the desktop scale) instead of shrinking to 358 px. Wider screens are unchanged. This
  affects `01_pipeline.png`, `01_real_pipeline.png`, `01_representations.png`, `03_votes.png` and
  `08_temperature.png`, not only the new figure; verified at 390 × 844 and 1280 × 800, no page-level
  horizontal scroll.
- **Page.** Footnote and caveat box say "different protocols" when the best method's protocol differs
  from the roundness baseline's; the box names the point order from the JSON and hedges "would be expected
  to impose [Inference]"; `#/mef/classify` has a one-line caveat with a link to the dataset page; the
  enlarged view's heading is the caption's first sentence, with the full caption below the image; the
  overlay's accessible name is a summary ("15 worm outlines and landmarks", "11 cell outlines").
- **Not changed, on purpose.** Chapter 10 keeps "one to two minutes" for the MEF PointNet run: this re-run
  took 78 s in total (the previous one 130 s), so "about two minutes" would now be wrong. The C. elegans
  processing note stays visible small print, not collapsed (the QA review listed its length as a nit only). The
  `licence` spellings left in test names and in the `RuntimeError("licence-restricted …")` message matched by
  `tests/test_lab.py` predate Round 3 and are not reader-facing text.
