"""Reconstruction / reprojection backends behind a common protocol.

Backends register under a name; the UIs and pipelines select one via
:func:`get_backend`. Importing this package registers the built-in TomoPy
backend (whose heavy imports are deferred until first use).
"""

from tktomo.recon.backend import (
    ReconBackend,
    available_backends,
    get_backend,
    register_backend,
)
from tktomo.recon import tomopy_backend as _tomopy_backend  # noqa: F401

__all__ = [
    "ReconBackend",
    "available_backends",
    "get_backend",
    "register_backend",
]
