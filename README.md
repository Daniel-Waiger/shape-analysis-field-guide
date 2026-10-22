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
(wild-type or lamin-deficient, 120 unordered points each). See [ATTRIBUTION.md](ATTRIBUTION.md).

## Read it

Two ways, same text:

- **On GitHub**: start at [docs/chapters/00-start-here.md](docs/chapters/00-start-here.md). Formulas and
  figures render in place; the animations are GIFs.
- **As a site** (nicer: dark mode, keyboard navigation, and six *interactive* demos where you drag landmarks,
  solve a rotation, walk a geodesic, sweep PCA modes, bend a spline and watch a contrastive loss pull points
  apart):

  ```bash
  cd docs && python3 -m http.server 8000     # then open http://localhost:8000
  ```

  To publish it, enable GitHub Pages from the `/docs` folder (private repos need a paid plan; see
  [docs/README-site.md](docs/README-site.md)).

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
| — | [Glossary](docs/chapters/glossary.md) · [Cheat sheet](docs/chapters/cheatsheet.md) · [References](docs/chapters/references.md) | |

Each chapter ends with **Try it** (a runnable snippet), **Check yourself** (three questions with hidden answers)
and a link onward. Reading time is about 10–15 minutes per chapter.

## Run it

```bash
pip install -e .                 # numpy, scipy, matplotlib, scikit-learn
pip install -e ".[embed]"        # + torch, for chapter 08
python -m pytest -q              # ~50 fast tests; the deep-learning ones skip without torch
python scripts/make_figures.py   # regenerates docs/figures/ (a few minutes on CPU)
python scripts/make_figures_embed.py   # chapter 08/09 figures; trains a small PointNet on CPU
python scripts/export_site_data.py     # numbers the interactive demos use
```

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
shapekit/        the library            tests/          pytest
scripts/         figure + data makers   data/           the two datasets (small)
```

## Honesty notes

- Numbers quoted in the chapters are results on these two datasets, computed by the scripts here or taken from
  the source tutorial's saved outputs; they are not general truths.
- Anything inferred rather than measured is marked **[Inference]**; outside facts not checked against a
  source are marked **[Unverified]**.
- The deep-learning chapter's trained model is a short CPU run meant to show the mechanism, not a benchmark.

License: MIT for this guide; adapted material is BSD-3 (see [LICENSE](LICENSE)).
