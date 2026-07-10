"""Image alignment: transforms, an undo stack, and pluggable methods.

Alignment methods register themselves under a name and are looked up through
:func:`get_aligner` / :func:`available_aligners`, which is how the alignment UI
populates its method dropdown. Importing this package registers the built-in
methods.
"""

from tktomo.align.apply import apply_projection_transform
from tktomo.align.base import (
    Aligner,
    available_aligners,
    get_aligner,
    register_aligner,
)
from tktomo.align.feature import (
    FeatureAlignmentResult,
    estimate_feature_transform,
    estimate_rigid,
    forward_points,
)
from tktomo.align.transform import Transform, TransformHistory, apply_transform

# Import for side effect: register built-in methods in the registry.
from tktomo.align import methods as _methods  # noqa: F401,E402

__all__ = [
    "Aligner",
    "FeatureAlignmentResult",
    "Transform",
    "TransformHistory",
    "apply_projection_transform",
    "apply_transform",
    "available_aligners",
    "estimate_feature_transform",
    "estimate_rigid",
    "forward_points",
    "get_aligner",
    "register_aligner",
]
