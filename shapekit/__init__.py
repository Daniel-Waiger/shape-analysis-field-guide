"""shapekit: a small, readable toolkit for 2-D statistical shape analysis.

Axis convention, used by every function in this package:

    points are rows, (x, y) are columns.
    one shape      -> array of shape (N, 2)
    a dataset      -> array of shape (K, N, 2)

The same shape can be written as N complex numbers ``z = x + i y``, which
is what the rotation, distance and mean functions take: ``(N,)`` for one
shape and ``(K, N)`` for a dataset.  Convert with :func:`to_complex` and
:func:`from_complex`.

Pixel coordinates follow the image: x = column (grows to the right),
y = row (grows downward), so plot shapes with an inverted y axis to see
them as they appear in the image.

Chapters refer to the field guide in ``docs/chapters``.
"""

from .preshape import center, frobenius_norm, from_complex, normalize, preshape, to_complex
from .procrustes import (
    align,
    align_dataset,
    canonical_orientation,
    gpa_mean,
    hermitian_product,
    optimal_rotation_angle,
    rotate,
)
from .shape_space import (
    chord_distance,
    exp_map,
    frechet_mean,
    geodesic_distance,
    geodesic_path,
    log_map,
    procrustes_distance,
)
from .tangent_pca import TangentPCA, real_to_tangent, tangent_to_real
from .bspline import BSplineCurve, basis_matrix, cubic_bspline, gram_matrix, interpolation_filter
from .contours import (
    arc_length,
    mask_to_contours,
    mask_to_landmarks,
    order_loop,
    order_skeleton,
    resample_by_arclength,
    skeletonize,
    skeleton_endpoints,
)
from .datasets import load_celegans, load_mef_subset

__version__ = "0.1.0"

__all__ = [
    "center", "frobenius_norm", "from_complex", "normalize", "preshape", "to_complex",
    "align", "align_dataset", "canonical_orientation", "gpa_mean", "hermitian_product",
    "optimal_rotation_angle", "rotate",
    "chord_distance", "exp_map", "frechet_mean", "geodesic_distance", "geodesic_path",
    "log_map", "procrustes_distance",
    "TangentPCA", "real_to_tangent", "tangent_to_real",
    "BSplineCurve", "basis_matrix", "cubic_bspline", "gram_matrix", "interpolation_filter",
    "arc_length", "mask_to_contours", "mask_to_landmarks", "order_loop", "order_skeleton",
    "resample_by_arclength", "skeletonize", "skeleton_endpoints",
    "load_celegans", "load_mef_subset",
]
