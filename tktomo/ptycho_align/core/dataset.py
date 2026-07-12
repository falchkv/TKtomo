"""Loading projection stacks for the alignment app.

The container is :class:`tktomo.io.ProjectionData` -- ``(n_angles, height, width)``
with angles in radians -- so the alignment app speaks the same type as the rest of
the toolkit. ``pixel_size_nm`` and ``name`` live in its ``metadata``.

HDF5 loading already exists in :mod:`tktomo.io.hdf5_loader`; this module adds the
two formats it lacks (``.npy``/``.npz`` and a TIFF directory + angles file) and the
validation the GUI needs to fail with a dialog rather than a traceback.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tktomo.io import ProjectionData, load_projections

__all__ = [
    "DatasetProblem",
    "angles_to_radians",
    "inspect_dataset",
    "load_dataset",
    "load_npy",
    "load_tiff_directory",
]

# Above this, an angle array cannot plausibly be radians (a full turn is 2*pi).
_DEGREE_THRESHOLD = 6.3

_TIFF_SUFFIXES = (".tif", ".tiff")


def angles_to_radians(angles: np.ndarray, in_degrees: bool | None = None) -> np.ndarray:
    """Convert angles to radians, autodetecting the unit when not told.

    Autodetect rule (from the spec): if ``max|angle| > 6.3`` it cannot be radians.
    """
    angles = np.asarray(angles, dtype=np.float64).ravel()
    if in_degrees is None:
        in_degrees = bool(np.max(np.abs(angles)) > _DEGREE_THRESHOLD) if angles.size else False
    return np.deg2rad(angles) if in_degrees else angles


def _finalise(
    projections: np.ndarray,
    angles: np.ndarray,
    *,
    name: str,
    pixel_size_nm: float | None = None,
    extra: dict | None = None,
) -> ProjectionData:
    projections = np.asarray(projections)
    if projections.ndim != 3:
        raise ValueError(
            f"Expected a 3-D projection stack (n_theta, n_v, n_u); got shape {projections.shape}"
        )
    metadata = {"name": name, "pixel_size_nm": pixel_size_nm}
    metadata.update(extra or {})
    return ProjectionData(
        data=projections.astype(np.float32), angles=angles, metadata=metadata
    )


def load_npy(path: str | Path, *, angles_in_degrees: bool | None = None) -> ProjectionData:
    """Load ``.npy`` (projections only) or ``.npz`` (``projections`` + ``angles``)."""
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as archive:
            keys = set(archive.files)
            data_key = next((k for k in ("projections", "data", "prj") if k in keys), None)
            if data_key is None:
                raise KeyError(
                    f"{path.name} has no projections array; looked for "
                    f"'projections'/'data'/'prj', found {sorted(keys)}"
                )
            projections = archive[data_key]
            angle_key = next((k for k in ("angles", "theta", "ang") if k in keys), None)
            angles = archive[angle_key] if angle_key else _default_angles(len(projections))
    else:
        projections = np.load(path)
        angles = _default_angles(len(projections))

    return _finalise(
        projections, angles_to_radians(angles, angles_in_degrees), name=path.stem
    )


def load_tiff_directory(
    directory: str | Path,
    angles_path: str | Path | None = None,
    *,
    angles_in_degrees: bool | None = None,
) -> ProjectionData:
    """Load a directory of TIFFs (sorted by filename) plus an angles ``.txt``/``.csv``."""
    import tifffile  # noqa: PLC0415

    directory = Path(directory)
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in _TIFF_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"No .tif/.tiff files in {directory}")

    projections = np.stack([tifffile.imread(f) for f in files])

    if angles_path is None:
        angles = _default_angles(len(files))
    else:
        angles = np.loadtxt(angles_path, delimiter=None, ndmin=1)
        if angles.ndim > 1:  # a CSV with more than one column: take the first
            angles = angles[:, 0]
        if len(angles) != len(files):
            raise ValueError(
                f"{len(angles)} angles but {len(files)} TIFFs in {directory}"
            )

    return _finalise(
        projections, angles_to_radians(angles, angles_in_degrees), name=directory.name
    )


def _default_angles(n: int) -> np.ndarray:
    """Assume a uniform 0-180 deg scan when the file carries no angles."""
    return np.linspace(0.0, np.pi, n, endpoint=False)


def load_dataset(path: str | Path, **kwargs) -> ProjectionData:
    """Dispatch on the path: directory -> TIFFs, ``.npy``/``.npz``, else HDF5."""
    path = Path(path)
    if path.is_dir():
        return load_tiff_directory(path, **kwargs)
    if path.suffix in (".npy", ".npz"):
        return load_npy(path, **kwargs)

    data = load_projections(path, **kwargs)
    data.data = np.asarray(data.data, dtype=np.float32)
    data.metadata.setdefault("name", path.stem)
    data.metadata.setdefault("pixel_size_nm", None)
    return data


class DatasetProblem(Exception):
    """A dataset defect the user must resolve. The GUI shows this as a dialog."""


def inspect_dataset(data: ProjectionData) -> list[str]:
    """Return human-readable warnings about a loaded dataset (empty = healthy).

    Catches the failures that would otherwise surface as a traceback deep inside
    tomopy: NaNs, non-monotonic angles, complex input, all-zero mass.
    """
    problems: list[str] = []
    array = data.data

    if np.iscomplexobj(array):
        problems.append(
            "Projections are complex-valued. This tool aligns PHASE projections -- "
            "take np.angle() of the stack first."
        )
        return problems  # the rest of the checks assume a real array

    if not np.all(np.isfinite(array)):
        n_bad = int((~np.isfinite(array)).sum())
        problems.append(f"{n_bad} non-finite value(s) (NaN/inf) in the projections.")

    angles = np.asarray(data.angles)
    if angles.size > 1:
        diffs = np.diff(angles)
        if not (np.all(diffs > 0) or np.all(diffs < 0)):
            problems.append(
                "Angles are not monotonic. Sort the projections by angle before aligning."
            )

    total = float(np.sum(array, dtype=np.float64))
    if total == 0.0:
        problems.append("The projections sum to exactly zero -- there is no mass to align.")
    elif total < 0.0:
        problems.append(
            "The projection integral is negative. Phase is negative for most samples; "
            "enable 'invert' so the reconstruction and the centre-of-mass see positive mass."
        )

    return problems
