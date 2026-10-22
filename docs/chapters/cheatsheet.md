# Cheat sheet

Every formula in the guide on one page, grouped by chapter. Each entry gives the formula, a short symbol legend and the `shapekit` function that implements it. Axis convention throughout: points are rows, `(N, 2)`; a dataset is `(K, N, 2)`; complex form is `(N,)` or `(K, N)`.

## 01 · From pixels to outlines

**Arc length along an ordered polyline**

$$
s_0 = 0, \qquad s_i = s_{i-1} + \lVert \mathbf{p}_i - \mathbf{p}_{i-1} \rVert
$$

- $\mathbf{p}_i$: the $i$-th ordered point (skeleton pixel or contour vertex); $s_i$: distance travelled to reach it.
- Resampling asks for $N$ values of $s$ equally spaced from $0$ to $s_{\text{end}}$ and interpolates linearly between points.
- `shapekit.contours.resample_by_arclength(points, N)`; ordering by `shapekit.contours.order_skeleton(mask)`; tracing by `shapekit.contours.mask_to_contours(mask, min_len)`.

## 02 · Preshape

**Centering, size, preshape**

$$
X_c = X - \bar{X}, \qquad \lVert X_c \rVert = \sqrt{\sum_{k=1}^{N} \left(x_k^2 + y_k^2\right)}, \qquad X_p = \frac{X_c}{\lVert X_c \rVert}
$$

- $X$: the $N \times 2$ landmark matrix; $\bar X$: its centroid (mean of the rows).
- $\lVert \cdot \rVert$: the Frobenius norm, the size being removed; $X_p$: the preshape, which lies on the unit hypersphere.
- `shapekit.preshape.center(X)`, `shapekit.preshape.normalize(X)`, `shapekit.preshape.preshape(X)`.

## 03 · Rotation with complex numbers

**Points as complex numbers; rotation**

$$
z_k = x_k + i\,y_k, \qquad e^{i\theta} z_k = (x_k\cos\theta - y_k\sin\theta) + i\,(x_k\sin\theta + y_k\cos\theta)
$$

- $z_k$: landmark $k$ as one complex number; $e^{i\theta} = \cos\theta + i\sin\theta$ rotates it by $\theta$ about the origin.
- `shapekit.preshape.to_complex(X)`, `shapekit.preshape.from_complex(z)`.

**Optimal rotation angle (closed form)**

$$
\theta^{\ast} = \operatorname{angle}\!\left(\sum_{k=1}^{N} \overline{z_k}\, w_k\right) = \operatorname{angle}\!\left(z^{H} w\right)
$$

- $z$: the shape being rotated; $w$: the target; $\overline{z_k}$: complex conjugate; $z^{H}$: conjugate transpose.
- Each term is one landmark's "vote" for a rotation; the angle of the sum is the consensus. Rotating $z$ by $\theta^{\ast}$ minimizes $\lVert w - e^{i\theta} z \rVert$.
- `shapekit.procrustes.optimal_rotation_angle(z, w)`, `shapekit.procrustes.align(z, w)`, `shapekit.procrustes.align_dataset(Z, ref)`.

## 04 · Kendall shape space

**Three distances between preshapes $z$ and $w$**

$$
\rho = \arccos\lvert z^{H} w \rvert, \qquad d_P = \lVert w - e^{i\theta^{\ast}} z \rVert, \qquad d_C = \sqrt{2 - 2\lvert z^{H} w \rvert}, \qquad d_P = d_C
$$

- $\rho$: geodesic distance, the arc on the sphere (0 to $\pi/2$); $d_P$: Procrustes distance, straight line after best rotation; $d_C$: chord distance. The last two are identical; the arc is always slightly longer.
- `shapekit.shape_space.geodesic_distance(z, w)`, `procrustes_distance(z, w)`, `chord_distance(z, w)`.

**Geodesic path (slerp)**

$$
\gamma(t) = \frac{\sin\!\big((1-t)\rho\big)\, e^{i\theta^{\ast}} z + \sin(t\rho)\, w}{\sin\rho}, \qquad t \in [0, 1]
$$

- $t$: fraction of the way from $z$ (after alignment) to $w$; every $\gamma(t)$ is itself a unit-norm preshape.
- `shapekit.shape_space.geodesic_path(z, w, n_steps=10)` returns $n_{\text{steps}} + 1$ shapes including both ends.

## 05 · Mean shapes

**Fréchet mean**

$$
S = \sum_{i=1}^{K} z_i z_i^{H}, \qquad S\,m = \lambda_{\max}\, m, \qquad m \leftarrow \frac{m - \bar m}{\lVert m - \bar m \rVert}
$$

- $S$: an $N \times N$ complex matrix summing every worm's outer product; $m$: its top eigenvector, re-centered and re-normalized. It minimizes $\sum_i \rho(m, z_i)^2$ (Dryden & Mardia 2016, Result 8.2).
- `shapekit.shape_space.frechet_mean(Z)`.

**Plain average of aligned shapes** (nearly the same direction, but shorter: norm 0.9645 on the worms)

$$
\bar z = \frac{1}{K} \sum_{i=1}^{K} e^{i\theta_i^{\ast}} z_i
$$

**Generalized Procrustes analysis = power iteration**

$$
m^{(0)} = z_{\text{random}}, \qquad m^{(j+1)} = \operatorname{normalize}\!\left(\frac{1}{K}\sum_i e^{i\theta_i^{\ast}} z_i\right) \;\propto\; \operatorname{normalize}\!\left(S\, m^{(j)}\right)
$$

- Each round aligns every shape to the current mean and re-averages; this is multiplying by $S$ and rescaling, so it converges to the Fréchet mean (within $10^{-5}$ in 1 to 2 rounds on the worms).
- `shapekit.procrustes.gpa_mean(Z, n_iter=20, seed=0, return_history=False)`.

## 06 · Tangent space and PCA

**Log map (sphere to sheet) and exp map (sheet to sphere)**

$$
\operatorname{Log}_m(z) = \frac{\rho}{\sin\rho}\left(e^{i\theta^{\ast}} z - \cos\rho\; m\right), \qquad
\operatorname{Exp}_m(v) = \cos t\; m + \frac{\sin t}{t}\, v, \quad t = \lVert v \rVert
$$

- $m$: the mean; $\rho$: geodesic distance from $m$ to $z$; $v$: a tangent vector, perpendicular to $m$, with $\lVert v \rVert = \rho$.
- `shapekit.shape_space.log_map(mean, z)`, `shapekit.shape_space.exp_map(mean, v)`.

**Tangent PCA**

$$
C = \frac{1}{K-1}(V - \bar V)^{T}(V - \bar V), \qquad C c_i = \lambda_i c_i, \qquad \text{explained}_i = \frac{\lambda_i}{\sum_j \lambda_j}, \qquad a_i = (v - \bar V)\cdot c_i
$$

- $V$: $K \times 2N$ matrix of tangent vectors as real numbers (real parts, then imaginary parts); $c_i$: principal component $i$; $\lambda_i$: its variance; $a_i$: a shape's score along it. Only $2N - 4$ components carry variance.
- `shapekit.tangent_pca.TangentPCA(mean).fit(Z)`, `.transform(Z)`, `.explained_variance_ratio_`.

**Modes of variation and sampling**

$$
\text{mode}_i^{\pm} = \operatorname{Exp}_m(\pm\, s_i c_i), \quad s_i = \sqrt{\lambda_i}; \qquad
z_{\text{new}} = \operatorname{Exp}_m\!\Big(\sum_{i=1}^{q} w_i s_i c_i\Big), \quad w_i \sim \mathcal N(0,1)
$$

- $s_i$: the standard deviation along mode $i$ (not $\sqrt{\lambda_i (K-1)}$, which is a singular value and about $\sqrt{K-1}$ times too large).
- `TangentPCA.mode_shape(i, n_sd=1.0)`, `TangentPCA.sample(n, n_modes, seed)`.

Results on the *C. elegans* landmarks: explained 66.1 / 13.3 / 8.7 / 7.4 / 2.2 / 1.0 %; flat PCA 59.4 / 19.5 / 9.4 / 6.8 %; dead mode 1 = 78.2 %, live mode 1 = 66.8 %; mode-1 bend amplitude about ±0.015 (dead) vs ±0.12 (live).

## 07 · Splines

**Cubic B-spline bump**

$$
\beta(x) = \begin{cases} \tfrac{2}{3} - x^2 + \tfrac{\lvert x\rvert^3}{2} & \lvert x\rvert < 1 \\ \tfrac{(2-\lvert x\rvert)^3}{6} & 1 \le \lvert x\rvert \le 2 \\ 0 & \text{otherwise}\end{cases}
\qquad \beta(0) = \tfrac{2}{3},\ \beta(\pm 1) = \tfrac{1}{6},\ \beta(\pm 2) = 0,\ \sum_k \beta(x - k) = 1
$$

- `shapekit.bspline.cubic_bspline(x)`.

**Curve, and its values at whole numbers (the 1-4-1 blur)**

$$
\mathbf r(t) = \sum_k c_k\, \beta(t - k), \qquad \mathbf r(k) = \frac{c_{k-1} + 4 c_k + c_{k+1}}{6}
$$

- $c_k$: control points; $t$: curve parameter. Each point of the curve depends on at most 4 control points (local control).
- `shapekit.bspline.BSplineCurve(M, closed).sample(n_points)`.

**Interpolation by the recursive filter**

$$
c^{+}_k = s_k + p\, c^{+}_{k-1}, \qquad c^{-}_k = p\left(c^{-}_{k+1} - c^{+}_k\right), \qquad c_k = 6\, c^{-}_k, \qquad p = -2 + \sqrt 3 \approx -0.268
$$

- $s_k$: measured points; $p$: the pole, the root of $x^2 + 4x + 1 = 0$ inside $(-1, 1)$; memory fades by 0.268 per step. Exact inverse of the 1-4-1 blur. Open curves pad each end twice ($M$ points → $M + 4$ coefficients); closed curves wrap around.
- `shapekit.bspline.interpolation_filter(samples, closed=False)`, `shapekit.bspline.BSplineCurve(M, closed).fit(points)`.

**Gram matrix and the curve inner product**

$$
\Phi_{kl} = \int \beta(t-k)\,\beta(t-l)\,dt, \qquad \langle z, w\rangle_\Phi = z^{H}\Phi w
$$

- $\Phi_{kl}$ depends only on $\lvert k - l \rvert$: 0.4794, 0.2363, 0.0238, 0.0002, then 0 (= 2416, 1191, 120, 1 over 5040); rows sum to 1.
- `shapekit.bspline.gram_matrix(M)`.

**Chapters 02 to 06 with $\Phi$ inserted**

| Quantity | Landmarks | Curves |
|---|---|---|
| size | $\sqrt{z^{H} z}$ | $\sqrt{z^{H}\Phi z}$ |
| rotation | $\operatorname{angle}(z^{H} w)$ | $\operatorname{angle}(z^{H}\Phi w)$ |
| geodesic | $\arccos\lvert z^{H} w\rvert$ | $\arccos\lvert z^{H}\Phi w\rvert$ |
| Fréchet mean | top eigenvector of $\sum z z^{H}$ | top eigenvector of $\sum z z^{H}\Phi$ |
| tangent PCA | on $V$ | on $V\sqrt{\Psi}$, $\Psi = \Phi \oplus \Phi$ [Inference] |

## 08 · Learned embeddings

**PointNet**

$$
\mathbf g = \max_{j=1\dots N} f_\theta(\mathbf p_j), \qquad \mathbf e = h_\phi(\mathbf g), \qquad \mathbf z = \frac{\mathbf u}{\lVert \mathbf u\rVert},\ \mathbf u = \text{head}(\mathbf e)
$$

- $\mathbf p_j$: point $j$; $f_\theta$: shared per-point MLP (2 → 64 → 128 → 1024); $\max$: per feature over points (permutation invariant); $h_\phi$: MLP to the embedding $\mathbf e$ (1024 → 1024 → 2048); head: 2048 → 128, then unit length so that $\mathbf z_i\cdot\mathbf z_j$ is cosine similarity.
- `shapekit.embed.PointNet(embed_dim, proj_dim)`.

**Softmax with temperature**

$$
P(p \mid i) = \frac{\exp(\mathbf z_i\cdot\mathbf z_p / \tau)}{\sum_{a\ne i}\exp(\mathbf z_i\cdot\mathbf z_a/\tau)}
$$

- $\tau$: temperature (0.07 in the tutorial). For two others with similarities 0.9 and 0.1: $P = 1/(1+e^{-0.8/\tau})$ = 0.690 at $\tau=1$, 0.935 at $\tau=0.3$, 0.99999 at $\tau=0.07$.

**Supervised contrastive loss**

$$
L = \frac{1}{B}\sum_{i=1}^{B}\left[-\frac{1}{\lvert P(i)\rvert}\sum_{p\in P(i)} \log P(p\mid i)\right]
$$

- $B$: batch size; $P(i)$: the positives of anchor $i$ (same class, not itself). If an anchor has no positive the term is $0/0$.
- Benchmarks: all vectors identical in a batch of 4 (two per class) gives $\ln 3 = 1.099$; no information in a batch of 200 gives $\ln 199 = 5.293$.
- `shapekit.embed.SupConLoss(temperature)`.

**Linear probe**

$$
P(\text{class 1}\mid \mathbf e) = \sigma(\mathbf w\cdot\mathbf e + b), \qquad \sigma(x) = \frac{1}{1 + e^{-x}}
$$

- Logistic regression on frozen embeddings, scored by stratified 5-fold cross-validation.
- `shapekit.embed.linear_probe_accuracy(features, labels)`.

Results on the tutorial's saved one-epoch CPU runs (full MEF set): untrained probe 0.835; loss after 1 epoch 5.269 / 5.276 (≈ ln 199); probe after 1 epoch 0.741 / 0.822.

## Constants worth memorizing

| Constant | Value | Where |
|---|---|---|
| $\beta(0), \beta(\pm1), \beta(\pm2)$ | $2/3,\ 1/6,\ 0$ | 07 |
| spline pole $p$ | $-2 + \sqrt 3 = -0.2679$ | 07 |
| $\Phi$ at lags 0, 1, 2, 3 | $0.4794,\ 0.2363,\ 0.0238,\ 0.0002$ | 07 |
| max geodesic distance between preshapes | $\pi/2 = 1.571$ | 04 |
| shape dimensions for $N$ planar landmarks | $2N - 4$ (= 20 for $N = 12$) | 06 |
| "learned nothing" SupCon loss, batch $B$ | $\ln(B - 1)$ | 08 |

---

*Credits: terms and formulas follow *Shape Analysis 101* by Johannes Hugger and Virginie Uhlmann (Uhlmann group, EMBL-EBI, 2021; BSD-3) and Dryden & Mardia (2016); see [ATTRIBUTION.md](../../ATTRIBUTION.md).*
