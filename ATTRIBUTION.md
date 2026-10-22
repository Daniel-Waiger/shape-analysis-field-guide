# Attribution

This guide is a re-telling, not an original research contribution. Here is what came from where.

## Course material

- **Shape Analysis 101** — Hugger, J., Uhlmann, V. (2021). *Shape Analysis 101 Tutorial.*
  <https://github.com/uhlmanngroup/shapeanalysis101>, BSD 3-Clause (Uhlmann Group @ EMBL-EBI).
  The chapter sequence, the two datasets, the worked results quoted in the chapters, and the
  methods implemented in `shapekit/` follow this tutorial. `shapekit/` is a clean re-implementation
  written for this guide, with the tutorial's `spline_models/` used as the reference for the B-spline
  interpolation filter and the spline shape space. The tutorial's license text is reproduced in `LICENSE`.
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
  (<https://github.com/kukionfr/Micropattern_MEF_LMNA_Image>). [Unverified] The license of that
  image dataset was not checked for this guide; the outlines are redistributed here on the strength of
  the tutorial's BSD-3 license only.

## Code the deep-learning chapter was checked against

- `shapekit/embed.py` follows the PointNet design of Qi et al. (2017), *PointNet*, arXiv:1612.00593,
  as implemented in <https://github.com/fxia22/pointnet.pytorch> (MIT).
- The supervised contrastive loss follows Khosla et al. (2020), arXiv:2004.11362, as implemented in
  <https://github.com/HobbitLong/SupContrast> (BSD 2-Clause).
- The `Center` transform discussed in chapter 09 is PyTorch Geometric's `transforms.Center` (MIT).

## Everything else

Text, figures, animations, interactive demos and tests were written for this repository and are MIT-licensed.
