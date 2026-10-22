# Attribution

This guide is a re-telling, not an original research contribution. Here is what came from where.

## Course material

- **Shape Analysis 101** — Johannes Hugger and Virginie Uhlmann (2021). *Shape Analysis 101 Tutorial.*
  <https://github.com/uhlmanngroup/shapeanalysis101>, BSD 3-Clause (Uhlmann Group @ EMBL-EBI).
  The chapter sequence, the two datasets, the worked results quoted in the chapters, and the
  methods implemented in `shapekit/` follow this tutorial. `shapekit/` is a clean re-implementation
  written for this guide, with the tutorial's `spline_models/` used as the reference for the B-spline
  interpolation filter (`spline_curve_model.py`, by Virginie Uhlmann: the Uhlmann group's Spline Fitting
  Toolbox v1.0) and the spline shape space (`spline_shape_space.py`, by Virginie Uhlmann and Anna Song). The tutorial's license text is reproduced in `LICENSE`.
- The spline shape-space construction follows: Song, A., Uhlmann, V., Fageot, J., Unser, M. (2020).
  *Dictionary learning for two-dimensional Kendall shapes.* SIAM Journal on Imaging Sciences 13(1), 141–175.

## Data

- `data/celegans_points.npy`, `data/celegans_labels.npy` — the tutorial's `point_models.npy` and `labels.npy`
  (1,377 worms × 12 landmarks; 0 = dead, 1 = live), derived by the tutorial authors from the Broad
  Bioimage Benchmark Collection set **BBBC010** (Ljosa, Sokolnicki & Carpenter 2012, *Nature Methods* 9:637).
  Files copied unchanged under the BSD-3 license above.
- `data/mef_outlines_subset.npz` — 1,200 of the tutorial's 5,377 MEF outlines (`mef_data.npy`), chosen
  with a fixed seed (600 wild-type, 600 LMNA−/−), converted from (row, col) to (x, y) and stored as float32.
  The tutorial derived these from the **Micropattern_MEF_LMNA_Image** dataset
  (<https://github.com/kukionfr/Micropattern_MEF_LMNA_Image>), MIT © 2020 Kyu Sang Han; the outlines
  are redistributed here under that MIT licence and the tutorial's BSD-3 licence.

## Lab datasets (chapter 10, `shapekit/lab_datasets.py`)

The Phenotype Lab uses six labelled datasets. The two above (`celegans`, `mef`) plus the four below. What
is in the repository was fetched and transformed by `scripts/fetch_datasets.py`, which prints the checksum
of every download and writes a notice file next to each dataset; nothing was typed in by hand.

| id | data in repo | source | licence | what was derived |
|---|---|---|---|---|
| `celegans` | `data/celegans_*.npy` | BBBC010 via Shape Analysis 101 (see above) | CC0 (images), BSD-3 (landmarks) | copied unchanged: 1,377 worms × 12 landmarks, 579 dead / 798 live |
| `mef` | `data/mef_outlines_subset.npz` | Micropattern_MEF_LMNA_Image via Shape Analysis 101 (see above) | MIT © 2020 Kyu Sang Han; BSD-3 (tutorial) | 1,200 of 5,377 outlines (600 / 600), 120 unordered points each |
| `brain` | `data/brain_landmarks.npz` | `schizophrenia.dat` of the R package **shapes** 1.2.8 by Ian L. Dryden (<https://cran.r-project.org/package=shapes>); Bookstein's schizophrenia data, Bookstein, F. L. (1996) *Bull. Math. Biol.* 58:313–365, provided to the package by Fred Bookstein | **GPL-2** (the package's licence; notice in `data/brain_landmarks.LICENSE`, full text in `LICENSES/GPL-2.txt`) | R's 13 × 2 × 28 array rearranged to (28, 13, 2); labels 0 = control (subjects 1–14), 1 = schizophrenia (15–28), verified against `schizophrenia$group`; values unchanged |
| `gorilla` | `data/gorilla_landmarks.npz` | `gorf.dat` and `gorm.dat` of the same package; data from Paul O'Higgins; O'Higgins, P. & Dryden, I. L. (1993) *J. Hum. Evol.* 24:183–205; Dryden & Mardia (2016) ch. 1 | **GPL-2** (notice in `data/gorilla_landmarks.LICENSE`) | 8 × 2 × 30 and 8 × 2 × 29 arrays rearranged to (59, 8, 2); labels 0 = female (30), 1 = male (29); values unchanged |
| `leaves` | `data/leaves_outlines.npz` | **Outlines of Swedish leaves**, Armand Leroi, Stephen Marsland and Arianna Salili-James, figshare DOI 10.6084/m9.figshare.14551005 (<https://figshare.com/articles/dataset/Outlines_of_Swedish_leaves/14551005>), traced from Oskar J. O. Söderkvist's (2001) Swedish Leaf dataset (Linköping University; <https://www.cvl.isy.liu.se/en/research/datasets/swedish-leaf/>, which states no terms; the author's Zenodo copy, record 19663514, is CC BY 4.0) | **CC BY 4.0** | the CSV carries the species labels (15 species, 440 outlines, 27–30 per species). Each outline (200 points in the file, the closing vertex repeated) was resampled to 150 points by arc length; coordinates otherwise unchanged (scaled to ±1.5 by the authors). Details in `data/leaves_outlines.ATTRIBUTION.md` |
| `rbc` | **nothing** (`data/_local/`, git-ignored) | **erythrocytesIDB** version 2 (October 2017), Marrero Fernández, P. D., Coello Said, G., Delgado Font, W. E., Herold García, S., Fernández García, K., Montoya Padrón, A., González-Hidalgo, M. & Jaume-i-Capó, A., Universitat de les Illes Balears, Zenodo record 18299474 (<https://zenodo.org/records/18299474>); related paper *IEEE J. Biomed. Health Inform.* 19(4):1514–1525 (2015) | **CC BY-NC-ND 4.0** | `scripts/fetch_datasets.py --rbc` downloads the 594 MB zip locally and traces one closed outline (150 arc-length points) per connected component of the per-class field masks of erythrocytesIDB2 and erythrocytesIDB3 (labels circular / elongated / other). Nothing derived is committed or published: the lab reports only accuracies, confusion matrices and counts for this dataset |

## Code the deep-learning chapter was checked against

- `shapekit/embed.py` follows the PointNet design of Qi et al. (2017), *PointNet*, arXiv:1612.00593,
  as implemented in <https://github.com/fxia22/pointnet.pytorch> (MIT).
- The supervised contrastive loss follows Khosla et al. (2020), arXiv:2004.11362, as implemented in
  <https://github.com/HobbitLong/SupContrast> (BSD 2-Clause).
- The `Center` transform discussed in chapter 09 is PyTorch Geometric's `transforms.Center` (MIT).

## Everything else

Text, figures, animations, interactive demos and tests were written for this repository and are MIT-licensed.

## What was and was not copied

A token-level scan of this repository against the tutorial (notebooks, modules, README) and against the two
upstream code bases above found no verbatim prose beyond bibliographic citations and marked quotations. Two
code fragments that mirrored their sources line for line, the two sweeps of the recursive spline filter and
one line of the contrastive loss, were rewritten (`shapekit/bspline.py` now solves the banded system directly
and writes the recursive variant with `scipy.signal.lfilter`; `shapekit/embed.py` uses `logsumexp`). The two
data files are copied unchanged under the tutorial's BSD-3 license. Every chapter ends with its credits and
every `shapekit` module names its sources in its header.
