"""Load / save projection stacks from HDF5 (NeXus NXtomo-friendly).

The loader is deliberately format-tolerant: it accepts explicit dataset paths and
otherwise probes a short list of conventional locations (including the NXtomo
``/entry/data/data`` + ``rotation_angle`` convention). Angles are normalised to
**radians** on load to match TomoPy.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from tktomo.io.data import ProjectionData

# Conventional dataset locations probed when explicit paths are not given.
_DATA_CANDIDATES = (
    "/entry/data/data",
    "/entry/instrument/detector/data",
    "/exchange/data",  # DXchange / TomoPy convention
    "/data",
)
_ANGLE_CANDIDATES = (
    "/entry/data/rotation_angle",
    "/entry/sample/rotation_angle",
    "/exchange/theta",
    "/angles",
)


def _import_h5py():
    try:
        import h5py  # noqa: PLC0415

        import hdf5plugin  # noqa: F401,PLC0415  # registers compression filters
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Reading HDF5 requires 'h5py' (and 'hdf5plugin' for compressed data). "
            "Install with: pip install h5py hdf5plugin"
        ) from exc
    return h5py


def _first_present(h5file, candidates: tuple[str, ...]) -> str | None:
    for path in candidates:
        if path in h5file:
            return path
    return None


def load_projections(
    path: str | os.PathLike[str],
    *,
    data_path: str | None = None,
    angle_path: str | None = None,
    angles_in_degrees: bool | None = None,
) -> ProjectionData:
    """Load a projection stack from an HDF5/NeXus file.

    Parameters
    ----------
    path:
        HDF5 file to read.
    data_path, angle_path:
        Explicit dataset locations. If omitted, conventional NXtomo / DXchange
        locations are probed.
    angles_in_degrees:
        If ``None`` (default) the unit is guessed: a span greater than ``2*pi``
        is assumed to be degrees. Pass explicitly to override.
    """
    h5py = _import_h5py()
    with h5py.File(path, "r") as f:
        dpath = data_path or _first_present(f, _DATA_CANDIDATES)
        if dpath is None:
            raise KeyError(
                f"No projection dataset found in {path!s}. Tried {_DATA_CANDIDATES}. "
                "Pass data_path=... explicitly."
            )
        data = np.asarray(f[dpath][()])

        apath = angle_path or _first_present(f, _ANGLE_CANDIDATES)
        if apath is not None:
            angles = np.asarray(f[apath][()], dtype=np.float64)
        else:
            # No angle dataset: assume a uniform 0..180 deg scan.
            angles = np.linspace(0.0, np.pi, data.shape[0], endpoint=False)

        metadata: dict[str, Any] = {
            "source_path": os.fspath(path),
            "data_path": dpath,
            "angle_path": apath,
        }

    angles = _to_radians(angles, angles_in_degrees)
    return ProjectionData(data=data, angles=angles, metadata=metadata)


def _to_radians(angles: np.ndarray, angles_in_degrees: bool | None) -> np.ndarray:
    if angles_in_degrees is None:
        span = float(np.ptp(angles)) if angles.size else 0.0
        angles_in_degrees = span > (2.0 * np.pi + 1e-6)
    return np.deg2rad(angles) if angles_in_degrees else angles


def save_projections(
    path: str | os.PathLike[str],
    proj: ProjectionData,
    *,
    compression: str | None = "gzip",
) -> None:
    """Write a :class:`ProjectionData` to HDF5 using the DXchange layout.

    Angles are stored in radians at ``/exchange/theta``; data at
    ``/exchange/data``. Round-trips with :func:`load_projections`.
    """
    h5py = _import_h5py()
    with h5py.File(path, "w") as f:
        grp = f.create_group("exchange")
        grp.create_dataset("data", data=proj.data, compression=compression)
        grp.create_dataset("theta", data=proj.angles)
        for key, value in proj.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                grp.attrs[key] = value
