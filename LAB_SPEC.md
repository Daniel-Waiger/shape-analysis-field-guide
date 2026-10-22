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
  (built by `scripts/build_lab_wheel.sh`), loads the dataset's npz from `../../data/` (fetch), and runs the
  classical pipeline with user-chosen M (spline), number of components and C, printing the CV accuracy. Clearly
  labelled as optional and ~20 MB.

Accessibility and honesty: real buttons/inputs; every number traceable (hover/tap shows "computed by … at …");
"[not run]" rather than blanks; no number is ever typed into the HTML.

## Scripts

- `scripts/fetch_datasets.py [--brain --gorilla --leaves --rbc --all]`: downloads with checksums printed,
  writes the npz files + per-dataset LICENSE/ATTRIBUTION notices; `--rbc` only into `data/_local/` (git-ignored).
  Must run with `python -I`-safe code (no imports from the download directories).
- `scripts/run_lab.py [--dataset id] [--skip-pointnet] [--epochs N]`: everything above; writes JSON + ONNX;
  prints a results table; ≤ 15 min total on 4 CPUs (PointNet epochs tuned per dataset size; brain/gorilla are
  too small for PointNet: skip it there and say so in JSON: `"pointnet_probe": {"status": "skipped", "reason": "n < 100"}`).
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
