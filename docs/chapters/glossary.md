# Glossary

Every term used in the guide, in alphabetical order, with the chapter that explains it. One or two sentences each; the chapters carry the detail.

**Adam** — An optimizer that adapts the step size for each weight separately; the tutorial used it with learning rate 1e-4 and weight decay 1e-4. [08](08-learned-embeddings.md)

**Alignment** — Rotating one preshape onto another so that they match as closely as possible; solved in closed form by the optimal rotation angle. [03](03-rotation-and-complex-numbers.md)

**Anchor** — In the supervised contrastive loss, the cell whose loss is being computed; all other cells in the batch are its positives or negatives. [08](08-learned-embeddings.md)

**Arc length** — Distance measured along a curve rather than in a straight line; landmarks are placed at equal arc-length intervals so that they are evenly spread. [01](01-from-pixels-to-outlines.md)

**Augmentation** — Random perturbations (flips, jitter) applied to training data so that the network learns to ignore them; never actually switched on in the tutorial. [08](08-learned-embeddings.md), [09](09-field-notes.md)

**Axis convention** — The rule for which array axis indexes points and which indexes coordinates; `shapekit` uses points as rows, `(N, 2)`. Mixing conventions caused the tutorial's worst bug. [09](09-field-notes.md)

**B-spline, cubic** — A smooth curve built as a sum of shifted copies of one bump function, each weighted by a control point. [07](07-splines.md)

**Basis function** — One of the shifted bumps whose weighted sum makes the spline curve; `cubic_bspline(x)` evaluates it. [07](07-splines.md)

**Batch** — The group of training examples processed together in one step; the contrastive loss compares every cell in the batch with every other. [08](08-learned-embeddings.md)

**BatchNorm** — A layer that rescales activations to a stable range using batch statistics in training mode and frozen running statistics in eval mode; the mode matters. [08](08-learned-embeddings.md), [09](09-field-notes.md)

**Binary mask** — An image where each pixel is either object or background. See also instance mask. [01](01-from-pixels-to-outlines.md)

**Centering** — Subtracting the centroid from every point so that the shape sits at the origin; removes position. [02](02-preshape.md)

**Centroid** — The mean position of a shape's points, its center of mass. [02](02-preshape.md)

**Chord distance** — The straight-line distance between two preshapes through the inside of the sphere, $\sqrt{2 - 2\lvert z^{H} w \rvert}$; equal to the Procrustes distance. [04](04-kendall-shape-space.md)

**Closed curve / open curve** — A closed curve returns to its start (a cell outline); an open curve has two ends (a worm midline). Splines handle the two with different end rules. [07](07-splines.md)

**Complex number** — A pair of real numbers $x + iy$ that can stand for a 2-D point; multiplying by $e^{i\theta}$ rotates it. [03](03-rotation-and-complex-numbers.md)

**Conjugate transpose** ($z^{H}$) — Flip a complex vector and negate the imaginary parts; $z^{H} w$ is the sum of $\overline{z_k} w_k$ over matching points. [03](03-rotation-and-complex-numbers.md)

**Contour** — The ordered loop of points around an object's boundary, traced from a mask. [01](01-from-pixels-to-outlines.md)

**Control point** — One of the weights (an $(x, y)$ position) that shapes a spline; the magnet on the flexible ruler. [07](07-splines.md)

**Correspondence** — Knowing that point $k$ of one shape matches point $k$ of another; landmarks have it, point clouds do not. [01](01-from-pixels-to-outlines.md), [08](08-learned-embeddings.md)

**Cosine similarity** — The dot product of two unit-length vectors, equal to the cosine of the angle between them; $1$ means identical direction. [08](08-learned-embeddings.md)

**Covariance matrix** — The table of how every pair of coordinates co-varies across the dataset; PCA finds its eigenvectors. [06](06-tangent-space-and-pca.md)

**Cross-validation, stratified K-fold** — Cut the data into K parts with equal class ratios, train on K−1 and test on the rest, rotate, average. [08](08-learned-embeddings.md)

**Dead code** — Code that nothing calls; it cannot fail until someone calls it. [09](09-field-notes.md)

**Deep copy** — A copy that duplicates the underlying data rather than pointing at it; `copy.deepcopy(model.state_dict())`. [09](09-field-notes.md)

**Degrees of freedom** — The number of independent ways a thing can vary; 12 planar landmarks have 24, of which 20 are shape. [06](06-tangent-space-and-pca.md)

**Eigenvector, eigenvalue** — A direction that a matrix only stretches, and the stretch factor. The top eigenvector of $\sum z z^{H}$ is the Fréchet mean; the eigenvectors of the covariance are the principal components. [05](05-mean-shapes.md), [06](06-tangent-space-and-pca.md)

**Embedding** — A fixed-length vector (2048 numbers in the tutorial) that a network produces as the fingerprint of one shape. [08](08-learned-embeddings.md)

**Epoch** — One full pass over the training data. The tutorial's saved runs used one. [08](08-learned-embeddings.md)

**Exp map** (exponential map) — Walks from the mean across the sphere along a tangent vector, turning a point on the flat sheet back into a preshape. [06](06-tangent-space-and-pca.md)

**Explained variance** — The fraction of total variance carried by one principal component; 66.1% for mode 1 of the worms. [06](06-tangent-space-and-pca.md)

**Fréchet mean** — The shape that minimizes the sum of squared geodesic distances to all shapes; computed as the top eigenvector of $\sum z z^{H}$. [05](05-mean-shapes.md)

**Frobenius norm** — The square root of the sum of squares of all coordinates; the "size" that is divided out to make a preshape. [02](02-preshape.md)

**Gaussian** (normal distribution) — The bell curve; sampling from it in the tangent space generates synthetic shapes. [06](06-tangent-space-and-pca.md)

**Generalized Procrustes analysis (GPA)** — Align all shapes to a reference, average, align to the new average, repeat; it is power iteration and converges to the Fréchet mean. [05](05-mean-shapes.md)

**Geodesic distance** — The length of the shortest arc between two preshapes on the sphere, $\arccos\lvert z^{H} w \rvert$. [04](04-kendall-shape-space.md)

**Geodesic path** (slerp) — The smooth morph from one shape to another along that arc. [04](04-kendall-shape-space.md)

**Gram matrix** ($\Phi$) — The table of overlaps between pairs of spline bumps, $\Phi_{kl} = \int \beta(t-k)\beta(t-l)\,dt$; inserting it into $z^{H} \Phi w$ compares whole curves. [07](07-splines.md)

**Hypersphere** — The set of all vectors of length 1 in many dimensions; every preshape lives on it. [02](02-preshape.md)

**Instance mask** — An image where each object has its own integer label; cells 1 to 11 in the MEF example. [01](01-from-pixels-to-outlines.md)

**Interpolation** — Finding control points so that the spline passes exactly through the measured points; done by the recursive filter. [07](07-splines.md)

**Kendall shape space** — The preshape sphere with shapes that differ only by rotation identified as one point; a curved space. [04](04-kendall-shape-space.md)

**k-means** — An unsupervised method that splits points into k clusters by nearest center; used to see whether embeddings separate by themselves. [08](08-learned-embeddings.md)

**Landmark** — A point placed on a shape such that the same-numbered point on another shape is its counterpart. [01](01-from-pixels-to-outlines.md)

**Learning rate** — The step size an optimizer takes along the gradient; 1e-4 in the tutorial. [08](08-learned-embeddings.md)

**Linear probe** — A logistic regression trained on frozen embeddings to measure how well they separate the classes. [08](08-learned-embeddings.md)

**Local control** — Moving one control point changes only the nearby part of a spline, because each bump is only four units wide. [07](07-splines.md)

**Log map** (logarithm map) — Projects a preshape onto the flat sheet at the mean, recording direction and geodesic distance. [06](06-tangent-space-and-pca.md)

**Logistic regression** — A weighted sum of features squashed through an S-curve into a probability; the simplest classifier. [08](08-learned-embeddings.md)

**Loss** — The number a network is trained to make small; here the supervised contrastive loss. [08](08-learned-embeddings.md)

**Marching squares** — The algorithm behind `find_contours`: it traces the level line between background and object through each 2×2 block of pixels. [01](01-from-pixels-to-outlines.md)

**Max-pool** — For each feature, keep the largest value over all points; the step that makes PointNet blind to point order. [08](08-learned-embeddings.md)

**Mean shape** — See Fréchet mean. The plain point-by-point average of aligned shapes is nearly the same but shorter. [05](05-mean-shapes.md)

**MLP** (multilayer perceptron) — A stack of "matrix multiply, add bias, ReLU" layers; PointNet uses one per point and one after pooling. [08](08-learned-embeddings.md)

**Mode of variation** — A principal component made visible: the mean shape walked $\pm 1$ SD along that component. [06](06-tangent-space-and-pca.md)

**Negative** — In the contrastive loss, a cell of a different class from the anchor. [08](08-learned-embeddings.md)

**Partition of unity** — The shifted spline bumps always sum to exactly 1, so shifting all control points shifts the whole curve. [07](07-splines.md)

**PCA** (principal component analysis) — Finds the directions of greatest variance in a dataset, in order; run on the tangent sheet for shapes. [06](06-tangent-space-and-pca.md)

**Permutation invariance** — The property that shuffling the input points leaves the output unchanged. [08](08-learned-embeddings.md)

**Pickle** — Python's serialization format; loading a pickled file can run code, so load only files you trust. [09](09-field-notes.md)

**Point cloud** — An unordered set of points on a shape, with no correspondence between shapes; the MEF cells are 120-point clouds. [01](01-from-pixels-to-outlines.md), [08](08-learned-embeddings.md)

**PointNet** — A network for point clouds: shared per-point MLP, max-pool, then an MLP on the pooled vector. [08](08-learned-embeddings.md)

**Pole** — The memory constant $p = -2 + \sqrt{3} \approx -0.268$ of the spline interpolation filter; each sweep's memory fades by this factor per step. [07](07-splines.md)

**Positive** — In the contrastive loss, another cell of the same class as the anchor. [08](08-learned-embeddings.md)

**Power iteration** — Repeatedly multiply a vector by a matrix and rescale; it converges to the top eigenvector. GPA is power iteration in disguise. [05](05-mean-shapes.md)

**Preshape** — A shape after centering and scaling to unit norm; position and size removed, rotation still present. [02](02-preshape.md)

**Principal component** — One of the directions PCA finds; a 24-number vector on the tangent sheet. [06](06-tangent-space-and-pca.md)

**Procrustes distance** — The straight-line distance between two preshapes after the best rotation, $\lVert w - e^{i\theta} z \rVert$; identical to the chord distance. [04](04-kendall-shape-space.md)

**Projection head** — A small network that maps the embedding to a shorter normalized vector for the loss; discarded after training. [08](08-learned-embeddings.md)

**Recursive filter** — The two-sweep algorithm (forward, backward, times 6) that un-blurs samples into spline control points exactly. [07](07-splines.md)

**ReLU** — Replace negative numbers with zero; the non-linearity between MLP layers. [08](08-learned-embeddings.md)

**Resampling** — Replacing the raw, unevenly spaced skeleton pixels with N points at equal arc-length intervals. [01](01-from-pixels-to-outlines.md)

**Rotation** — Turning a shape about the origin; in complex form, multiplication by $e^{i\theta}$. [03](03-rotation-and-complex-numbers.md)

**Scree plot** — A bar or line chart of explained variance per component, used to decide how many modes matter. [06](06-tangent-space-and-pca.md)

**Seed** — The starting number for a random generator; fixing it makes shuffles, splits and samples reproducible. [09](09-field-notes.md)

**Shape** — What is left of a configuration of points after position, size and rotation are removed. [00](00-start-here.md)

**Skeleton** — The one-pixel-wide center line left after peeling away an object's outer layers; used as the worm's midline. [01](01-from-pixels-to-outlines.md)

**Softmax** — Turns a list of scores into probabilities by exponentiating and normalizing; the heart of the contrastive loss. [08](08-learned-embeddings.md)

**√(n−1) pitfall** — Scaling samples by a singular value $\sqrt{\lambda(n-1)}$ instead of a standard deviation $\sqrt{\lambda}$, which throws them off the sphere. [06](06-tangent-space-and-pca.md)

**State dict** — A PyTorch model's weights as a dictionary of tensors; `model.state_dict()` returns live references, not a copy. [09](09-field-notes.md)

**Supervised contrastive loss** (SupCon) — For each anchor, the negative log softmax probability of each positive among all other cells in the batch, averaged. [08](08-learned-embeddings.md)

**Tangent space** — The flat sheet touching the sphere at the mean; where PCA is done. [06](06-tangent-space-and-pca.md)

**Temperature** ($\tau$) — The divisor applied to similarities before the softmax; small $\tau$ makes the loss pickier. [08](08-learned-embeddings.md)

**T-Net** — PointNet's mini-network that predicts a small matrix to apply to all input points before the per-point MLP. [08](08-learned-embeddings.md)

**Train / eval mode** — PyTorch's two modes, switched by `model.train()` and `model.eval()`; they change BatchNorm's behavior and must be set explicitly. [09](09-field-notes.md)

**Weight decay** — A small pull of all weights toward zero during training, to limit overfitting. [08](08-learned-embeddings.md)
