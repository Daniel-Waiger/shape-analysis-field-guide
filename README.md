# Shape Analysis Field Guide

*From outlines to shape space, for biologists.*

<p align="center">
  <img src="docs/figures/hero.gif" width="720"
       alt="A live worm morphing along a geodesic through the mean shape to a dead worm, looping">
  <br><sub>A live worm (teal) walking the shortest path in shape space through the mean (ink) to a dead worm (coral). Chapter 04 explains the walk.</sub>
</p>

**Shape** is what is left of an outline after you throw away *where* it is, *how big* it is and *which way* it
points. This guide teaches, in plain language, how to turn microscope outlines into numbers that respect that
definition, how to average and compare them, how to find the few ways a population really varies, and how a
neural network can learn a shape "fingerprint". Every idea comes with a picture, a small hand-checkable
example, a formula with a word-by-word reading, and the function in `shapekit/` that implements it.

It adapts the EMBL-EBI tutorial *Shape Analysis 101* (Hugger & Uhlmann, 2021) and its two datasets:
1,377 *C. elegans* worms (dead or alive, 12 landmarks each) and 1,200 mouse fibroblast outlines
(wild-type or lamin-deficient, 120 unordered points each; in every field checked, 28 of 200 with the rest
[Inference], genotype is confounded with the growth surface and with whatever else differs between the two source
folders, so results on this set cannot be attributed to genotype; see chapter 10). Chapter 10's Phenotype Lab adds four more
(brain and gorilla landmarks, leaf outlines, red blood cells). See [ATTRIBUTION.md](ATTRIBUTION.md).

## Read it

Two ways, same text:

- **On GitHub**: start at [docs/chapters/00-start-here.md](docs/chapters/00-start-here.md). Formulas and
  figures render in place; the animations are GIFs.
- **As a site**, at **<https://daniel-waiger.github.io/shape-analysis-field-guide/>** (nicer: dark mode,
  keyboard navigation, and six *interactive* demos where you drag landmarks, solve a rotation, walk a
  geodesic, sweep PCA modes, bend a spline and watch a contrastive loss pull points apart). To run the same
  site locally:

  ```bash
  cd docs && python3 -m http.server 8000     # then open http://localhost:8000
  ```

  It is served by GitHub Pages from the `/docs` folder of `main` (see [docs/README-site.md](docs/README-site.md)).

## The map

| # | Chapter | What you walk away with |
|---|---|---|
| 00 | [Start here](docs/chapters/00-start-here.md) | What "shape" means, the two datasets, how to run things |
| 01 | [From pixels to outlines](docs/chapters/01-from-pixels-to-outlines.md) | Masks → contours → ordered points; landmarks vs point clouds vs curves |
| 02 | [Preshape](docs/chapters/02-preshape.md) | Removing position and size; the preshape sphere |
| 03 | [Rotation, solved in one line](docs/chapters/03-rotation-and-complex-numbers.md) | Complex numbers turn "find the best rotation" into one formula |
| 04 | [Kendall shape space](docs/chapters/04-kendall-shape-space.md) | Geodesic vs chord vs Procrustes distance; walking between shapes |
| 05 | [Mean shapes](docs/chapters/05-mean-shapes.md) | The Fréchet mean as a top eigenvector; why iterating alignment lands on it |
| 06 | [Tangent space and PCA](docs/chapters/06-tangent-space-and-pca.md) | Flattening the sphere; modes of variation; dead vs live worms; sampling new shapes |
| 07 | [Shapes as smooth curves](docs/chapters/07-splines.md) | Cubic B-splines, local control, fitting by a two-pass filter, the overlap matrix Φ |
| 08 | [Learned embeddings](docs/chapters/08-learned-embeddings.md) | PointNet, permutation invariance, supervised contrastive loss, linear probes |
| 09 | [Field notes](docs/chapters/09-field-notes.md) | Nine real bugs from the source tutorial, why they happened, the habits that prevent them |
| 10 | [Phenotype Lab](docs/chapters/10-phenotype-lab.md) | Six labeled datasets, three methods side by side, a classifier you can drag; what the real numbers say, including one below chance and one where the label is confounded (the fibroblasts: genotype travels with substrate) |
| — | [Glossary](docs/chapters/glossary.md) · [Cheat sheet](docs/chapters/cheatsheet.md) · [References](docs/chapters/references.md) | |

Each chapter ends with **Try it** (a runnable snippet), **Check yourself** (three questions with hidden answers)
and a link onward. Reading time is about 10–15 minutes per chapter.

## Run it

```bash
pip install -e .                 # numpy, scipy, matplotlib, scikit-learn
pip install -e ".[embed]"        # + torch, for chapter 08
python -m pytest -q              # 163 tests (about 20 s); the deep-learning ones skip without torch
python scripts/make_figures.py   # regenerates docs/figures/ (a few minutes on CPU)
python scripts/make_figures_embed.py   # chapter 08/09 figures; trains a small PointNet on CPU
python scripts/export_site_data.py     # numbers the interactive demos use
python scripts/fetch_datasets.py --images celegans mef   # original images into data/_local/images/ (git-ignored)
python scripts/make_sample_images.py   # the Lab's sample images + overlays, and docs/figures/01_real_pipeline.png
```

### Regenerate the Phenotype Lab (chapter 10)

Everything the Lab page shows is written by three scripts, in this order:

```bash
pip install -e ".[lab]"          # torch, onnx, onnxruntime, pyreadr/xarray (or rdata), Pillow, contourpy, build
python scripts/fetch_datasets.py --brain --gorilla --leaves   # 1. the redistributable datasets (checksummed)
python scripts/fetch_datasets.py --rbc                        #    optional, CC BY-NC-ND: 594 MB into data/_local/
python scripts/run_lab.py                                     # 2. docs/lab/data/*.json, models/*.onnx, data/files/
bash scripts/build_lab_wheel.sh                               # 3. docs/lab/py/shapekit-*.whl for the Pyodide panel
```

1. `fetch_datasets.py` downloads with pinned MD5s and writes `data/*.npz` plus a license notice per
   dataset; `--rbc` stays under the git-ignored `data/_local/` (its license forbids redistribution).
2. `run_lab.py` computes every number (seed 0, 4 torch threads), writes the JSON contract, the ONNX
   models and `index.json`, copies the `data/` files to `docs/lab/data/files/` for the in-browser Python
   panel, and self-checks every engine. Measured: 8.7 min for all six datasets on an idle 4-CPU machine
   (PointNet is all of it; `--skip-pointnet --out /tmp/lab` runs the classical methods in seconds). It never
   replaces a ready `rbc.json` with a "fetch locally" stub unless you pass `--force`.
3. `build_lab_wheel.sh` builds the pure-Python wheel the "Run it yourself" panel installs.

The PointNet numbers are reproducible for a given seed, thread count, torch build and CPU; the classical
numbers on any machine. `tests/test_lab_shipped.py` checks the shipped files against the code.

`shapekit/` is a small, readable, tested re-implementation of every method in the guide. The convention
throughout: a shape is an array of shape `(N, 2)`, points as rows, `(x, y)` as columns; a dataset is `(K, N, 2)`.

```python
from shapekit.datasets import load_celegans
from shapekit.preshape import preshape, to_complex
from shapekit.shape_space import frechet_mean, geodesic_distance
from shapekit.tangent_pca import TangentPCA

points, labels = load_celegans()            # (1377, 12, 2), 0 = dead, 1 = live
Z = to_complex(preshape(points))            # position and size removed, one complex number per landmark
mean = frechet_mean(Z)                      # the average worm
pca = TangentPCA(mean).fit(Z)               # the main ways worms differ
print(pca.explained_variance_ratio_[:4])    # about 0.66, 0.13, 0.09, 0.07: mostly bending
```

## Where things live

```
docs/chapters/   the text               docs/figures/   generated figures and GIFs
docs/demos/      interactive demos      docs/data/      precomputed numbers for the demos
docs/lab/        the Phenotype Lab      docs/lab/data/  its JSON (numbers, shapes, engines) + data/files/ copies
                 (chapter 10, a page)   docs/lab/models/, docs/lab/py/   ONNX PointNets, the Pyodide wheel
shapekit/        the library            tests/          pytest
scripts/         figure, data and lab   data/           the datasets (small; license notices next to them),
                 makers                                 data/_local/ (git-ignored, fetched, never published)
```

## Credits

This guide stands on other people's work, and says so:

- **Johannes Hugger and Virginie Uhlmann** (Uhlmann group, EMBL-EBI) wrote the course this guide adapts,
  [*Shape Analysis 101*](https://github.com/uhlmanngroup/shapeanalysis101) (2021, BSD-3): its chapter
  sequence, its two datasets and the worked results quoted here are theirs. Virginie Uhlmann wrote the
  spline toolbox it ships (`spline_curve_model.py`), and Virginie Uhlmann and Anna Song wrote its spline
  shape space, implementing Song, Uhlmann, Fageot & Unser (2020).
- The statistical theory is Kendall's (1984), as set out by Dryden & Mardia (2016). Splines follow Unser,
  Aldroubi & Eden (1993). PointNet is Qi et al. (2017); supervised contrastive learning is Khosla et al.
  (2020). Reference implementations read while writing `shapekit.embed`: fxia22/pointnet.pytorch (MIT) and
  HobbitLong/SupContrast (BSD-2).
- Data: BBBC010 (Ljosa, Sokolnicki & Carpenter 2012; CC0) for the worms; the Micropattern_MEF_LMNA_Image
  dataset (Kyu Sang Han, MIT) for the cells, both as prepared by the tutorial (BSD-3). For the Phenotype Lab:
  Bookstein's schizophrenia landmarks and O'Higgins' gorilla skulls from Ian Dryden's R package `shapes`
  (GPL-2); *Outlines of Swedish leaves* by Leroi, Marsland and Salili-James (figshare, CC BY 4.0), extracted
  from Söderkvist's Swedish Leaf dataset; and erythrocytesIDB (Universitat de les Illes Balears, CC BY-NC-ND
  4.0), which is fetched locally and never redistributed.

Nothing from these sources is reproduced verbatim except six data files (two from the tutorial, brain and
gorilla under GPL-2, leaves under CC BY 4.0; see [ATTRIBUTION.md](ATTRIBUTION.md)) and bibliographic
citations: the methods were re-implemented, the text was written fresh, and the figures and demos are new.
Where a passage quotes the tutorial, it is in quotation marks. Each chapter ends with its own credits, each
`shapekit` module names its sources in its header, and [ATTRIBUTION.md](ATTRIBUTION.md) has the full list.

## Honesty notes

- Numbers quoted in chapters 00 to 09 are results on the tutorial's two datasets, computed by the scripts here or
  taken from the source tutorial's saved outputs; the numbers in chapter 10 are written by `scripts/run_lab.py`
  into `docs/lab/data/`. None of them is a general truth.
- Anything inferred rather than measured is marked **[Inference]**; outside facts not checked against a
  source are marked **[Unverified]**.
- The deep-learning chapter's trained model is a short CPU run meant to show the mechanism, not a benchmark.

License: MIT for this guide; adapted material is BSD-3; the data files keep their own licenses (GPL-2, CC BY 4.0, MIT; see [LICENSE](LICENSE) and [ATTRIBUTION.md](ATTRIBUTION.md)).
