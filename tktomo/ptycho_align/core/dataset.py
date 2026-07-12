"""Loading projection stacks for the alignment app.

The container is :class:`tktomo.io.ProjectionData` -- ``(n_angles, height, width)``
with angles in radians -- so the alignment app speaks the same type as the rest of
the toolkit. ``pixel_size_nm`` and ``name`` live in its ``metadata``.

:mod:`tktomo.io.hdf5_loader` loads HDF5 by *probing conventional locations*
(NXtomo, DXchange, the blender-sim layout). Real ptycho reconstructions routinely
sit somewhere else entirely, under whatever name the beamline's pipeline chose, and
sometimes with the angle axis last. So this module adds the escape hatch that the
convention-driven loader cannot provide: :func:`list_hdf5_datasets` enumerates every
array in the file so the GUI can show a tree of it, and :func:`load_dataset` takes an
explicit ``data_path``/``angle_path``/``axis_order`` to load whichever one the user
picked. It also adds the two formats the io layer lacks (``.npy``/``.npz`` and a TIFF
directory + angles file) and the validation the GUI needs to fail with a dialog
rather than a traceback.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tktomo.io import ProjectionData, load_projections

__all__ = [
    "DatasetProblem",
    "Hdf5Entry",
    "angles_to_radians",
    "inspect_dataset",
    "list_hdf5_datasets",
    "load_dataset",
    "load_hdf5",
    "load_npy",
    "load_tiff_directory",
    "suggest_hdf5_paths",
]

# Above this, an angle array cannot plausibly be radians (a full turn is 2*pi).
_DEGREE_THRESHOLD = 6.3

_TIFF_SUFFIXES = (".tif", ".tiff")

_HDF5_SUFFIXES = (".h5", ".hdf5", ".nxs", ".nx5", ".hdf", ".h5py")

# Substrings that make a 1-D dataset look like a rotation-angle array.
_ANGLE_HINTS = ("angle", "theta", "rot", "omega", "tilt")


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


@dataclass(frozen=True)
class Hdf5Entry:
    """One array inside an HDF5 file, as the dataset browser sees it."""

    path: str  # absolute, e.g. "/recon/phase/projections"
    shape: tuple[int, ...]
    dtype: str

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def is_stack(self) -> bool:
        """Could this be a projection stack? (3-D, and not degenerately small.)"""
        return self.ndim == 3 and min(self.shape) > 1

    @property
    def is_angles(self) -> bool:
        """Could this be an angle array? (1-D with more than one entry.)"""
        return self.ndim == 1 and self.shape[0] > 1


def list_hdf5_datasets(path: str | Path) -> list[Hdf5Entry]:
    """Enumerate every dataset in an HDF5 file, at any depth.

    This is what lets the GUI show the file's real structure instead of guessing:
    the user navigates to the array they mean, wherever the beamline put it.
    """
    h5py = _import_h5py()

    entries: list[Hdf5Entry] = []

    def visit(name: str, item) -> None:
        if isinstance(item, h5py.Dataset):
            entries.append(
                Hdf5Entry(
                    path="/" + name,
                    shape=tuple(int(n) for n in item.shape),
                    dtype=str(item.dtype),
                )
            )

    with h5py.File(path, "r") as f:
        f.visititems(visit)

    entries.sort(key=lambda e: e.path)
    return entries


def suggest_hdf5_paths(entries: list[Hdf5Entry]) -> tuple[str | None, str | None]:
    """Pre-select the most likely (projections, angles) pair for the browser.

    A guess, offered as a starting point in a dialog the user can override -- not a
    silent decision. The largest 3-D array is the projections; the angle array is a
    1-D dataset whose length matches its leading axis, preferring one whose name or
    location says "angle"/"theta", and preferring a sibling of the stack.
    """
    stacks = [e for e in entries if e.is_stack]
    if not stacks:
        return None, None
    stack = max(stacks, key=lambda e: int(np.prod(e.shape)))

    group = stack.path.rsplit("/", 1)[0]
    n_angles = stack.shape[0]

    def score(entry: Hdf5Entry) -> tuple[int, int, int]:
        named = any(hint in entry.path.lower() for hint in _ANGLE_HINTS)
        # Length match is the strongest signal, then the name, then proximity.
        return (
            int(entry.shape[0] == n_angles),
            int(named),
            int(entry.path.rsplit("/", 1)[0] == group),
        )

    candidates = [e for e in entries if e.is_angles]
    best = max(candidates, key=score, default=None)
    if best is None or score(best) == (0, 0, 0):
        return stack.path, None
    return stack.path, best.path


def load_hdf5(
    path: str | Path,
    *,
    data_path: str,
    angle_path: str | None = None,
    axis_order: tuple[int, int, int] = (0, 1, 2),
    angles_in_degrees: bool | None = None,
) -> ProjectionData:
    """Load explicitly-named datasets out of an HDF5 file of any layout.

    Parameters
    ----------
    data_path:
        Absolute path of the 3-D projection stack inside the file.
    angle_path:
        Absolute path of the 1-D angle array. If omitted, a uniform 0-180 deg scan
        is assumed.
    axis_order:
        Which stored axis becomes ``(angle, v, u)``. ``(0, 1, 2)`` leaves the array
        as stored; ``(2, 0, 1)`` reads a stack saved as ``(v, u, angle)``.
    """
    h5py = _import_h5py()
    path = Path(path)

    with h5py.File(path, "r") as f:
        if data_path not in f:
            raise KeyError(f"{data_path!r} is not in {path.name}")
        array = np.asarray(f[data_path][()])
        angles = np.asarray(f[angle_path][()]) if angle_path else None
        if angle_path and angles is None:  # pragma: no cover - defensive
            raise KeyError(f"{angle_path!r} is not in {path.name}")

    if array.ndim != 3:
        raise ValueError(
            f"{data_path!r} has shape {array.shape}; a projection stack must be 3-D."
        )
    if sorted(axis_order) != [0, 1, 2]:
        raise ValueError(f"axis_order must be a permutation of (0, 1, 2); got {axis_order}")
    array = np.transpose(array, axis_order)

    if angles is None:
        angles = _default_angles(array.shape[0])
    else:
        angles = angles_to_radians(angles, angles_in_degrees)
        if len(angles) != array.shape[0]:
            raise ValueError(
                f"{angle_path!r} holds {len(angles)} angles but {data_path!r} has "
                f"{array.shape[0]} projections along its leading axis. Either the angle "
                "dataset is the wrong one, or the stack's axis order is."
            )

    return _finalise(
        array,
        angles,
        name=f"{path.stem}{data_path}",
        extra={"source_path": str(path), "data_path": data_path, "angle_path": angle_path},
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


def load_dataset(path: str | Path, *, data_path: str | None = None, **kwargs) -> ProjectionData:
    """Dispatch on the path: directory -> TIFFs, ``.npy``/``.npz``, else HDF5.

    Passing ``data_path`` (with the other keywords of :func:`load_hdf5`) bypasses the
    layout probing entirely and reads exactly the dataset named.
    """
    path = Path(path)
    if path.is_dir():
        return load_tiff_directory(path, **kwargs)
    if path.suffix in (".npy", ".npz"):
        return load_npy(path, **kwargs)

    if data_path is not None:
        return load_hdf5(path, data_path=data_path, **kwargs)

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
