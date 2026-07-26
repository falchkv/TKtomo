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

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tktomo.io import ProjectionData, load_projections

__all__ = [
    "COMPONENTS",
    "Crop",
    "DatasetProblem",
    "Hdf5Entry",
    "angles_to_radians",
    "coerce_load_kwargs",
    "inspect_dataset",
    "jsonable_load_kwargs",
    "list_hdf5_datasets",
    "load_dataset",
    "load_hdf5",
    "load_npy",
    "load_tiff_directory",
    "suggest_hdf5_paths",
    "to_real",
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
    if np.iscomplexobj(projections):
        # Casting complex -> float32 only warns, and silently keeps the real part, which
        # for a ptycho reconstruction is meaningless. Every caller must choose.
        raise ValueError(
            "The projections are complex. Choose a component (phase, amplitude, real, "
            "imaginary) -- see load_hdf5(component=...)."
        )
    metadata = {"name": name, "pixel_size_nm": pixel_size_nm}
    metadata.update(extra or {})
    # asarray, not astype: the HDF5 reader already builds a float32 array, and a
    # gratuitous copy of a multi-gigabyte stack doubles peak memory for nothing.
    return ProjectionData(
        data=np.asarray(projections, dtype=np.float32), angles=angles, metadata=metadata
    )


def load_npy(
    path: str | Path,
    *,
    angles_in_degrees: bool | None = None,
    component: str = "phase",
) -> ProjectionData:
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
        to_real(projections, component),
        angles_to_radians(angles, angles_in_degrees),
        name=path.stem,
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

    @property
    def is_complex(self) -> bool:
        """Complex projections need a component (phase/amplitude) chosen on load."""
        return self.dtype.startswith("complex")

    def stack_shape(self, axis_order: tuple[int, int, int] = (0, 1, 2)) -> tuple[int, int, int]:
        """The ``(angle, v, u)`` shape this dataset has under ``axis_order``."""
        return tuple(self.shape[axis] for axis in axis_order)  # type: ignore[return-value]


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


class Crop:
    """A rectangular detector region, in the ``(v, u)`` frame *after* ``axis_order``.

    Cropping is not a convenience here, it is what makes a large file openable at all:
    a 410 x 733 x 1950 complex64 stack is 4.7 GB on disk and 2.3 GB as float32 in RAM,
    and the alignment holds several copies of it. The crop is applied as an HDF5
    hyperslab *during* the read, so the discarded region is never in memory.
    """

    __slots__ = ("v0", "v1", "u0", "u1")

    def __init__(self, v0: int, v1: int, u0: int, u1: int) -> None:
        if v1 <= v0 or u1 <= u0:
            raise ValueError(f"Empty crop: rows {v0}:{v1}, columns {u0}:{u1}")
        self.v0, self.v1, self.u0, self.u1 = int(v0), int(v1), int(u0), int(u1)

    @classmethod
    def full(cls, shape: tuple[int, int, int]) -> Crop:
        """The whole detector of a stack of ``(angle, v, u)`` shape."""
        return cls(0, shape[1], 0, shape[2])

    def clipped_to(self, shape: tuple[int, int, int]) -> Crop:
        """Clamp to a stack's bounds, so a stale crop cannot read out of range."""
        v1 = min(self.v1, shape[1])
        u1 = min(self.u1, shape[2])
        return Crop(min(self.v0, v1 - 1), v1, min(self.u0, u1 - 1), u1)

    def shifted_by(self, dv: int, du: int) -> Crop:
        """Translate the crop, e.g. to re-express an ROI drawn on already-cropped data."""
        return Crop(self.v0 + dv, self.v1 + dv, self.u0 + du, self.u1 + du)

    @property
    def height(self) -> int:
        return self.v1 - self.v0

    @property
    def width(self) -> int:
        return self.u1 - self.u0

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.v0, self.v1, self.u0, self.u1)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Crop) and self.as_tuple() == other.as_tuple()

    def __repr__(self) -> str:
        return f"Crop(v={self.v0}:{self.v1}, u={self.u0}:{self.u1})"


def jsonable_load_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Render ``load_dataset`` kwargs so they survive JSON or msgpack.

    The browser and crop dialogs build this dict, but the file it describes may live on
    another machine, so the dict has to travel. :class:`Crop` is not serialisable -- it
    goes over as its ``as_tuple()`` -- and tuples arrive as lists.
    """
    out = dict(kwargs)
    crop = out.get("crop")
    if isinstance(crop, Crop):
        out["crop"] = list(crop.as_tuple())
    elif crop is not None:
        out["crop"] = [int(v) for v in crop]
    if out.get("axis_order") is not None:
        out["axis_order"] = [int(a) for a in out["axis_order"]]
    return out


def coerce_load_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Undo :func:`jsonable_load_kwargs` -- the same fix ``AlignConfig.from_dict`` makes.

    Neither JSON nor msgpack has a tuple, so ``axis_order`` and ``crop`` come back as
    lists. ``load_hdf5`` happens to tolerate both today, but that is luck rather than
    design, and the dict is also stored in ``metadata`` and compared against later.
    """
    out = dict(kwargs)
    crop = out.get("crop")
    if crop is not None and not isinstance(crop, Crop):
        out["crop"] = Crop(*(int(v) for v in crop))
    if out.get("axis_order") is not None:
        axis_order = tuple(int(a) for a in out["axis_order"])
        if sorted(axis_order) != [0, 1, 2]:
            raise ValueError(f"axis_order must be a permutation of (0, 1, 2); got {axis_order}")
        out["axis_order"] = axis_order
    return out


# How a complex projection becomes the real image the alignment works on. Phase is the
# default: this is a phase-contrast tool, and ptycho reconstructions are complex.
COMPONENTS: dict[str, Any] = {
    "phase": np.angle,
    "amplitude": np.abs,
    "real": np.real,
    "imaginary": np.imag,
}


def to_real(array: np.ndarray, component: str = "phase") -> np.ndarray:
    """Reduce a complex array to the named real component. Real input passes through."""
    if not np.iscomplexobj(array):
        return np.asarray(array, dtype=np.float32)
    try:
        reducer = COMPONENTS[component]
    except KeyError:
        raise ValueError(
            f"Unknown component {component!r}; choose one of {sorted(COMPONENTS)}"
        ) from None
    return np.asarray(reducer(array), dtype=np.float32)


def load_hdf5(
    path: str | Path,
    *,
    data_path: str,
    angle_path: str | None = None,
    axis_order: tuple[int, int, int] = (0, 1, 2),
    angles_in_degrees: bool | None = None,
    component: str = "phase",
    crop: Crop | tuple[int, int, int, int] | None = None,
    progress: Callable[[int, int], bool] | None = None,
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
    component:
        For complex projections (a ptycho reconstruction is complex), which real
        component to align: ``phase`` (default), ``amplitude``, ``real``, ``imaginary``.
        Ignored for real-valued data.
    crop:
        Detector region to keep, in the post-``axis_order`` ``(v, u)`` frame. Read as a
        hyperslab one projection at a time, so the rest of the file never enters memory.
    progress:
        Called as ``progress(done, total)`` after each projection. Return ``False`` to
        abort the read -- which raises :class:`DatasetProblem`, so a half-read stack can
        never be mistaken for a whole one.
    """
    h5py = _import_h5py()
    path = Path(path)

    if sorted(axis_order) != [0, 1, 2]:
        raise ValueError(f"axis_order must be a permutation of (0, 1, 2); got {axis_order}")

    with h5py.File(path, "r") as f:
        if data_path not in f:
            raise KeyError(f"{data_path!r} is not in {path.name}")
        dataset = f[data_path]

        if dataset.ndim != 3:
            raise ValueError(
                f"{data_path!r} has shape {dataset.shape}; a projection stack must be 3-D."
            )

        stack_shape = tuple(int(dataset.shape[axis]) for axis in axis_order)
        n_angles = stack_shape[0]

        if crop is None:
            crop = Crop.full(stack_shape)  # type: ignore[arg-type]
        elif not isinstance(crop, Crop):
            crop = Crop(*crop)
        crop = crop.clipped_to(stack_shape)  # type: ignore[arg-type]

        angles = _read_angles(
            f, path, data_path, angle_path, n_angles, angles_in_degrees=angles_in_degrees
        )

        array = np.empty((n_angles, crop.height, crop.width), dtype=np.float32)
        read_plane = _plane_reader(dataset, axis_order, crop)
        for i in range(n_angles):
            array[i] = to_real(read_plane(i), component)
            if progress is not None and not progress(i + 1, n_angles):
                raise DatasetProblem(f"Loading {data_path!r} was cancelled.")

    return _finalise(
        array,
        angles,
        name=f"{path.stem}{data_path}",
        extra={
            "source_path": str(path),
            "data_path": data_path,
            "angle_path": angle_path,
            "axis_order": tuple(axis_order),
            "component": component,
            "crop": crop.as_tuple(),
            "full_shape": stack_shape,
        },
    )


def _read_angles(
    f,
    path: Path,
    data_path: str,
    angle_path: str | None,
    n_angles: int,
    *,
    angles_in_degrees: bool | None,
) -> np.ndarray:
    if not angle_path:
        return _default_angles(n_angles)
    if angle_path not in f:
        raise KeyError(f"{angle_path!r} is not in {path.name}")
    angles = angles_to_radians(np.asarray(f[angle_path][()]), angles_in_degrees)
    if len(angles) != n_angles:
        raise ValueError(
            f"{angle_path!r} holds {len(angles)} angles but {data_path!r} has "
            f"{n_angles} projections along its leading axis. Either the angle "
            "dataset is the wrong one, or the stack's axis order is."
        )
    return angles


def _plane_reader(dataset, axis_order: tuple[int, int, int], crop: Crop):
    """Return ``read(i) -> (v, u) plane``, slicing in the file rather than in memory.

    The index is built in the *stored* frame so h5py reads only the cropped hyperslab;
    the returned plane is then transposed into the ``(v, u)`` frame ``axis_order`` asks
    for. For the usual ``(0, 1, 2)`` this is just ``dataset[i, v0:v1, u0:u1]``.
    """
    angle_axis, v_axis, u_axis = axis_order
    rest = [axis for axis in (0, 1, 2) if axis != angle_axis]
    plane_order = (rest.index(v_axis), rest.index(u_axis))

    def read(i: int) -> np.ndarray:
        index: list[Any] = [None, None, None]
        index[angle_axis] = i
        index[v_axis] = slice(crop.v0, crop.v1)
        index[u_axis] = slice(crop.u0, crop.u1)
        plane = dataset[tuple(index)]
        return plane if plane_order == (0, 1) else np.transpose(plane, plane_order)

    return read


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
