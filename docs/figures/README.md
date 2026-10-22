# Figures

Static figures for the chapters in `docs/chapters/`. All of them are generated,
never hand-edited; regenerate with `python scripts/make_figures.py` from the
repository root (deterministic, seed 0, about 10 s; `--only NAME` for one
figure, `--list` for the names). Palette and conventions are in `SPEC.md`:
class 0 (dead worm / wild-type cell) is coral, class 1 (live worm / LMNA-/-) is
teal, means are ink, "before" states slate, highlights gold, the tangent plane
and splines lavender. Shapes are drawn with equal aspect and the y axis
inverted, so worms look as they do in the image.

| File | Shows | Made by |
|---|---|---|
| `hero.gif` | A live worm (teal) morphing along the geodesic to the mean worm (ink) and on to a dead worm (coral), then back; loops | `scripts/make_figures.py` |
| `og-card.png` | 1200 × 630 still of the same three worms with the title: the image link previews (WhatsApp, Slack, X…) show for the site, referenced by the Open Graph tags in `docs/index.html` | `scripts/make_og_card.py` |
| `01_pipeline.png` | Synthetic cell image → threshold mask → traced contours (one dropped by `min_len`) → 20 arc-length landmarks → closed cubic B-spline | `scripts/make_figures.py` |
| `01_arclength_resampling.png` | Ordered skeleton pixels with 1 px and √2 px steps (diagonals in gold) next to the same midline resampled to 12 equal arc-length steps | `scripts/make_figures.py` |
| `01_representations.png` | One worm as 12 ordered landmarks, as 120 unordered points, and as a continuous spline curve | `scripts/make_figures.py` |
| `02_preshape_steps.png` | Six worms where the camera found them, then centred, then centred and scaled to norm 1 | `scripts/make_figures.py` |
| `02_sphere.png` | A shaded 3-D sphere with all 1,377 worms placed by their first two tangent-PCA scores, the mean, and one great-circle arc to worm 899 | `scripts/make_figures.py` |
| `03_rotation_search.gif` | Worm B (818) turning over worm A (87) while the mismatch E(θ) is traced; ends with the minimum at θ* = −1.4923 rad marked and B drawn there in gold | `scripts/make_figures.py` |
| `03_votes.png` | The two worms, the 12 vote arrows conj(b_k)·a_k, and the votes added head to tail with their sum in gold; the sum's angle is θ* | `scripts/make_figures.py` |
| `04_distances.png` | The great circle through the mean and worm 899 drawn to scale: arc (geodesic 0.35815) and chord (Procrustes 0.35624), with the formulas and numbers | `scripts/make_figures.py` |
| `04_geodesic_morph.gif` | The mean worm morphing into a U-shaped worm (1236) along the geodesic; all 11 steps t = 0 … 1 light up below | `scripts/make_figures.py` |
| `05_means.png` | Fréchet mean (ink), plain average of the aligned worms (slate, norm 0.9645) and the GPA result (gold) overlaid, with a zoom on one tip | `scripts/make_figures.py` |
| `05_power_iteration.gif` | GPA from three random starting worms, every round rotated onto the Fréchet mean, with the distance to the mean per round on a log axis | `scripts/make_figures.py` |
| `06_tangent_plane.png` | Sphere, the lavender tangent sheet at the mean, a worm on the sphere, its log image on the sheet, and the log and exp arrows | `scripts/make_figures.py` |
| `06_scree.png` | Variance explained by modes 1–10, tangent PCA (ink) against flat PCA on the aligned coordinates (faint) | `scripts/make_figures.py` |
| `06_modes.png` | Modes 1–4 at ±1 SD around the mean | `scripts/make_figures.py` |
| `06_mode_sweep.gif` | Mode 1 swept −2 SD → +2 SD → −2 SD with a slider readout | `scripts/make_figures.py` |
| `06_pca_scatter.png` | Mode-1 against mode-2 scores of all worms, coloured by class | `scripts/make_figures.py` |
| `06_dead_vs_live.png` | Per-class Fréchet mean and first two modes at ±1 SD, dead (coral) beside live (teal), same axes | `scripts/make_figures.py` |
| `06_sampling.png` | Six worms sampled from a Gaussian on the first three tangent modes with the data's own standard deviations | `scripts/make_figures.py` |
| `07_bspline_basis.png` | The cubic B-spline bump with its values at integers, its shifted copies, and their sum (exactly 1) | `scripts/make_figures.py` |
| `07_spline_fit.png` | Twelve landmarks fitted with M = 6 (least squares) and M = 12 (exact interpolation) control points, with control polygons | `scripts/make_figures.py` |
| `07_local_control.gif` | One control point moving in a circle; only the four spline segments it touches (gold) move | `scripts/make_figures.py` |
| `08_pointnet.png` | PointNet architecture diagram | `scripts/make_figures_embed.py` |
| `08_maxpool_invariance.png` | Shuffled point order gives the same embedding | `scripts/make_figures_embed.py` |
| `08_temperature.png` | Softmax of the same similarities at three temperatures | `scripts/make_figures_embed.py` |
| `08_embedding_pca.png` | PCA of the embeddings coloured by class | `scripts/make_figures_embed.py` |
| `09_center_axis_bug.png` | Correct centring versus the wrong-axis result for one cell | `scripts/make_figures_embed.py` |

Worm indices used by `make_figures.py` are chosen so the figures reproduce the
numbers quoted in the chapters: rotating worm 818 onto worm 87 gives
θ* = −1.4922899425 rad; worm 899 lies 0.35815 (geodesic) and 0.35624 (chord)
from the Fréchet mean; worm 622 is the arched live worm used in chapters 01 and
07; worm 2 the nearly straight dead worm of the hero animation.
