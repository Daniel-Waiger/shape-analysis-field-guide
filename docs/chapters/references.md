# References

Papers, datasets and upstream code behind this guide. Entries are grouped by what they contributed. A detail marked **[Unverified]** was not checked against the original source during the writing of this guide; the core citation (authors, year, venue) is given as the field commonly cites it.

## The tutorial this guide adapts

- **Hugger, J. and Uhlmann, V. (2021).** *Shape Analysis 101.* EMBL-EBI Uhlmann group. Four Jupyter notebooks plus `spline_models/` and `shape_embeddings/`. BSD 3-Clause license, © 2021 Uhlmann Group @ EMBL-EBI. <https://github.com/uhlmanngroup/shapeanalysis101>. The snapshot read for this guide was commit `bfccc0c` (21 Nov 2023). Its README also cites a 2021 Hugger & Uhlmann preprint for the embedding notebook; the preprint's title and venue are **[Unverified]** and nothing in the code refers to it beyond that citation.
- **Plain-language walkthrough** of that tutorial, including the catalog of bugs summarized in [Chapter 09](09-field-notes.md): `GLOBIAS-2026-GROUP-5/docs/shapeanalysis101_explained.md` (internal document; the source of every number quoted as "results on these datasets").

## Shape statistics (Chapters 02 to 06)

- **Kendall, D. G. (1984).** Shape manifolds, Procrustean metrics, and complex projective spaces. *Bulletin of the London Mathematical Society* 16:81–121. The paper that defines shape space as the preshape sphere modulo rotation and gives its dimension ($2N - 4$ for $N$ planar points).
- **Dryden, I. L. and Mardia, K. V. (2016).** *Statistical Shape Analysis, with Applications in R*, 2nd edition. Wiley, Chichester. The tutorial cites its Result 8.2 for the Fréchet mean as the top eigenvector of $\sum z_i z_i^{H}$. **[Unverified]** that the result number is unchanged between the 1998 and 2016 editions.
- **Klingenberg, C. P. (2020).** Walking on Kendall's shape space: understanding shape spaces and their coordinate systems. *Evolutionary Biology* 47:334–352. A biologist-facing explanation of tangent spaces and Procrustes coordinates; the natural next read after [Chapter 06](06-tangent-space-and-pca.md). Title **[Unverified]** word for word.

## Splines (Chapter 07)

- **Unser, M. (1999).** Splines: a perfect fit for signal and image processing. *IEEE Signal Processing Magazine* 16(6):22–38. The source for the cubic B-spline bump, the recursive interpolation filter with pole $-2 + \sqrt 3$, and the fact that the Gram matrix entries are the degree-7 B-spline at integers.
- **Unser, M., Aldroubi, A. and Eden, M. (1993).** B-spline signal processing: Part I, theory; Part II, efficient design and applications. *IEEE Transactions on Signal Processing* 41(2):821–833 and 834–848. The original derivation of the two-sweep filter. Page ranges **[Unverified]**.
- **Song, A., Uhlmann, V., Fageot, J. and Unser, M. (2020).** Dictionary learning for two-dimensional Kendall shapes. *SIAM Journal on Imaging Sciences* 13(1):141–175. Cited by the tutorial for the spline shape space of notebook 2B (the Gram-matrix inner product and its Fréchet mean, alignment and tangent PCA). Author list and exact title **[Unverified]**; the tutorial's treatment of *open* curves with the periodic Gram matrix was not checked against this paper.

## Learned embeddings (Chapter 08)

- **Qi, C. R., Su, H., Mo, K. and Guibas, L. J. (2017).** PointNet: deep learning on point sets for 3D classification and segmentation. *CVPR 2017*; arXiv:1612.00593. Shared per-point MLP, max-pool, and the T-Net with its orthogonality regularizer.
- **Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C. and Krishnan, D. (2020).** Supervised contrastive learning. *NeurIPS 2020*; arXiv:2004.11362. The SupCon loss of [Chapter 08](08-learned-embeddings.md). Full author list **[Unverified]** in order.
- **Fey, M. and Lenssen, J. E. (2019).** Fast graph representation learning with PyTorch Geometric. *ICLR 2019 Workshop on Representation Learning on Graphs and Manifolds*; arXiv:1903.02428. The library whose `Center` transform (points × 2 convention) was copied into the tutorial; see [Chapter 09](09-field-notes.md), story 1.

## Datasets

- **BBBC010: *C. elegans* live/dead assay.** Ljosa, V., Sokolnicki, K. L. and Carpenter, A. E. (2012). Annotated high-throughput microscopy image sets for validation. *Nature Methods* 9:637. <https://bbbc.broadinstitute.org/BBBC010>. Brightfield images of worms; wells A23 (dead) and B07 (live) are the two used by the tutorial. The 1377 twelve-landmark worms in `data/celegans_points.npy` derive from this set via the tutorial's `point_models.npy`; the exact landmarking procedure is **[Inference]** (segment lengths vary by about 2% per worm, consistent with arc-length resampling).
- **MEF_LMNA micropattern images.** `kukionfr/Micropattern_MEF_LMNA_Image` on GitHub: fluorescence images of wild-type and lamin-A/C-deficient (LMNA−/−) mouse embryonic fibroblasts, with instance segmentations. <https://github.com/kukionfr/Micropattern_MEF_LMNA_Image>. Author, license and associated publication **[Unverified]**. The tutorial's `mef_data.npy` (5377 cells × 120 unordered outline points) was built from 200 of these images; `data/mef_outlines_subset.npz` in this guide is a 1200-cell subset.

## Upstream code the tutorial borrowed from

- **HobbitLong/SupContrast.** Reference PyTorch implementation of the supervised contrastive loss by Yonglong Tian and collaborators. <https://github.com/HobbitLong/SupContrast>. License **[Unverified]** (BSD-2 as commonly reported). The tutorial's `SupConLoss` is adapted from its `losses.py`; its training script calls `model.train()` every epoch, a line lost in the tutorial's split loop.
- **fxia22/pointnet.pytorch.** PyTorch port of PointNet by Fei Xia. <https://github.com/fxia22/pointnet.pytorch>. License **[Unverified]** (MIT as commonly reported). Source of `STN3d` (renamed `SpatialTransformer3d` in the tutorial), `PointNetfeat`, and the 2 × points input convention; the T-Net regularizer `feature_transform_regularizer` was not ported.
- **PyTorch Geometric.** <https://github.com/pyg-team/pytorch_geometric>, MIT license. Source of the `Center` transform (points × 2 convention).

## Software used by `shapekit`

- NumPy, SciPy, Matplotlib, scikit-learn; optionally PyTorch for [Chapter 08](08-learned-embeddings.md). Versions are pinned in `requirements.txt` for the reasons given in [Chapter 09](09-field-notes.md), story 7.
