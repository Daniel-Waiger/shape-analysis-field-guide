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
  Bioimage Benchmark Collection set **BBBC010** (<https://bbbc.broadinstitute.org/BBBC010>; Ljosa, Sokolnicki & Carpenter 2012, *Nature Methods* 9:637; CC0).
  Files copied unchanged under the BSD-3 license above.
- `data/mef_outlines_subset.npz` — 1,200 of the tutorial's 5,377 MEF outlines (`mef_data.npy`), chosen
  with a fixed seed (600 wild-type, 600 LMNA−/−), converted from (row, col) to (x, y) and stored as float32.
  The tutorial derived these from the **Micropattern_MEF_LMNA_Image** dataset
  (<https://github.com/kukionfr/Micropattern_MEF_LMNA_Image>), MIT © 2020 Kyu Sang Han; the outlines
  are redistributed here under that MIT license and the tutorial's BSD-3 license; Han's copyright and
  permission notice is reproduced in `data/mef_outlines_subset.LICENSE`.

## Lab datasets (chapter 10, `shapekit/lab_datasets.py`)

The Phenotype Lab uses six labeled datasets. The two above (`celegans`, `mef`) plus the four below. What
is in the repository was fetched and transformed by `scripts/fetch_datasets.py`, which prints the checksum
of every download and writes a notice file next to each dataset; nothing was typed in by hand.

| id | data in repo | source | license | what was derived |
|---|---|---|---|---|
| `celegans` | `data/celegans_*.npy` | BBBC010 (<https://bbbc.broadinstitute.org/BBBC010>) via Shape Analysis 101 (see above) | CC0 (images), BSD-3 (landmarks) | copied unchanged: 1,377 worms × 12 landmarks, 579 dead / 798 live |
| `mef` | `data/mef_outlines_subset.npz` | Micropattern_MEF_LMNA_Image via Shape Analysis 101 (see above) | MIT © 2020 Kyu Sang Han (notice in `data/mef_outlines_subset.LICENSE`); BSD-3 (tutorial) | 1,200 of 5,377 outlines (600 / 600), 120 unordered points each |
| `brain` | `data/brain_landmarks.npz` | `schizophrenia.dat` of the R package **shapes** 1.2.8 by Ian L. Dryden (<https://cran.r-project.org/package=shapes>); Bookstein's schizophrenia data, Bookstein, F. L. (1996) *Bull. Math. Biol.* 58:313–365, provided to the package by Fred Bookstein | **GPL-2** (the package's license, <https://www.gnu.org/licenses/old-licenses/gpl-2.0.txt>; notice in `data/brain_landmarks.LICENSE`, full text in `LICENSES/GPL-2.txt`). [Unverified] whether the data providers granted the package's license to the data themselves; the package's license is the only statement found | R's 13 × 2 × 28 array rearranged to (28, 13, 2); labels 0 = control (subjects 1–14), 1 = schizophrenia (15–28), verified against `schizophrenia$group`; values unchanged |
| `gorilla` | `data/gorilla_landmarks.npz` | `gorf.dat` and `gorm.dat` of the same package; data from Paul O'Higgins; O'Higgins, P. & Dryden, I. L. (1993) *J. Hum. Evol.* 24:183–205; Dryden & Mardia (2016) ch. 1 | **GPL-2** (same package, same URL; notice in `data/gorilla_landmarks.LICENSE`; same [Unverified] hedge) | 8 × 2 × 30 and 8 × 2 × 29 arrays rearranged to (59, 8, 2); labels 0 = female (30), 1 = male (29); values unchanged |
| `leaves` | `data/leaves_outlines.npz` | **Outlines of Swedish leaves**, Armand Leroi, Stephen Marsland and Arianna Salili-James, figshare DOI 10.6084/m9.figshare.14551005 (<https://figshare.com/articles/dataset/Outlines_of_Swedish_leaves/14551005>), extracted (figshare's word) from the photographs of Oskar J. O. Söderkvist's (2001) Swedish Leaf dataset (Linköping University; <https://www.cvl.isy.liu.se/en/research/datasets/swedish-leaf/>, which states no terms; a Zenodo record listing Söderkvist as creator, 19663514, CC BY 4.0, deposited 2026-04-20, uploader not identified, is the only license statement found for the photographs) | **CC BY 4.0** (<https://creativecommons.org/licenses/by/4.0/>) | the CSV carries the species labels (15 species, 440 outlines, 27–30 per species). Changes to the authors' data: each outline (200 points in the file, the closing vertex repeated; the figshare description says 150) was resampled to 150 points by arc length, and the lab resamples again to 100 points for the classical methods and re-roots each outline at the cyclic shift that best matches the mean; coordinates otherwise unchanged (scaled to ±1.5 by the authors). Details in `data/leaves_outlines.ATTRIBUTION.md` |
| `rbc` | **nothing** (`data/_local/`, git-ignored) | **erythrocytesIDB** version 2 (October 2017), Marrero Fernández, P. D., Coello Said, G., Delgado Font, W. E., Herold García, S., Fernández García, K., Montoya Padrón, A., González-Hidalgo, M. & Jaume-i-Capó, A., Universitat de les Illes Balears, Zenodo record 18299474 (<https://zenodo.org/records/18299474>); related paper *IEEE J. Biomed. Health Inform.* 19(4):1514–1525 (2015) | **CC BY-NC-ND 4.0** | `scripts/fetch_datasets.py --rbc` downloads the 594 MB zip locally and traces one closed outline (150 arc-length points) per connected component of the per-class field masks of erythrocytesIDB2 and erythrocytesIDB3 (80 fields; labels circular / elongated / other). The count is the lab's own extraction, not a dataset count: 9 components were dropped by an area filter (8 in IDB2, 1 in IDB3) and one IDB2 field has no circular mask, so "circular" is probably under-counted. The license forbids sharing adapted material (ND) and commercial use (NC), so, as a precaution, nothing containing coordinates is committed or published: the lab reports only counts, accuracies, confusion matrices, fold accuracies, worst-mistake indices with probabilities and the per-epoch validation history; no coordinates, means, modes, scatter, neighbors or weights |

## Code the deep-learning chapter was checked against

- `shapekit/embed.py` follows the PointNet design of Qi et al. (2017), *PointNet*, arXiv:1612.00593,
  as implemented in <https://github.com/fxia22/pointnet.pytorch> (MIT).
- The supervised contrastive loss follows Khosla et al. (2020), arXiv:2004.11362, as implemented in
  <https://github.com/HobbitLong/SupContrast> (BSD 2-Clause).
- The `Center` transform discussed in chapter 09 is PyTorch Geometric's `transforms.Center` (MIT).

## Everything else

Text, figures, animations, interactive demos and tests were written for this repository and are MIT-licensed. The data files under `data/` keep the licenses in the table above (BSD-3 and MIT, GPL-2, CC BY 4.0), not the repository's MIT.

## What was and was not copied

A token-level scan of this repository against the tutorial (notebooks, modules, README) and against the two
upstream code bases above found no verbatim prose beyond bibliographic citations and marked quotations. Two
code fragments that mirrored their sources line for line, the two sweeps of the recursive spline filter and
one line of the contrastive loss, were rewritten (`shapekit/bspline.py` now solves the banded system directly
and writes the recursive variant with `scipy.signal.lfilter`; `shapekit/embed.py` uses `logsumexp`). Six
data files are copied from other sources: the two tutorial files unchanged under its BSD-3 license (the MEF
outlines also under Kyu Sang Han's MIT), the brain and gorilla landmarks under GPL-2 (`data/*.LICENSE`,
`LICENSES/GPL-2.txt`) and the leaf outlines under CC BY 4.0 (`data/leaves_outlines.ATTRIBUTION.md`); the
red-cell data is never copied into the repository. Every chapter ends with its credits and
every `shapekit` module names its sources in its header.
