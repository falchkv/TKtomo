"""Session save/load and exports.

A session is a single HDF5 file holding everything needed to resume stepping: the
preprocessed projections, the angles, the *full* per-iteration shift history, the
config, the current volume and the metadata. Closing the app and reopening the
session puts you back exactly where you were.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tktomo.io import ProjectionData
from tktomo.ptycho_align.core.engine import AlignConfig, AlignmentEngine
from tktomo.ptycho_align.core.state import IterationResult

__all__ = [
    "export_convergence_csv",
    "export_shifts_csv",
    "export_tiff_stack",
    "export_volume",
    "load_session",
    "save_session",
]

_SESSION_VERSION = 1


def _h5py():
    import h5py  # noqa: PLC0415

    return h5py


def save_session(path: str | Path, engine: AlignmentEngine) -> None:
    """Write the engine's whole state to ``path``."""
    h5py = _h5py()
    state = engine.state

    with h5py.File(path, "w") as handle:
        handle.attrs["session_version"] = _SESSION_VERSION
        handle.attrs["config"] = json.dumps(engine.config.to_dict())
        handle.attrs["center"] = float(state.center)
        # Metadata is free-form, so JSON it rather than fighting HDF5 attr types.
        handle.attrs["metadata"] = json.dumps(_jsonable(state.metadata))

        handle.create_dataset("projections", data=state.original, compression="gzip")
        handle.create_dataset("angles", data=state.angles)
        handle["angles"].attrs["units"] = "radians"
        handle.create_dataset("sx", data=state.sx)
        handle.create_dataset("sy", data=state.sy)
        if state.volume is not None:
            handle.create_dataset("volume", data=state.volume, compression="gzip")

        history = handle.create_group("history")
        for result in state.history:
            group = history.create_group(str(result.iteration))
            group.create_dataset("sx", data=result.sx)
            group.create_dataset("sy", data=result.sy)
            group.create_dataset("dsx", data=result.dsx)
            group.create_dataset("dsy", data=result.dsy)
            group.attrs["error"] = result.error
            group.attrs["residual"] = result.residual
            group.attrs["center"] = result.center
            group.attrs["wallclock_s"] = result.wallclock_s
            group.attrs["config_changed"] = result.config_changed
            # Volumes are only stored for iterations the memory policy kept.
            if result.volume is not None:
                group.create_dataset("volume", data=result.volume, compression="gzip")


def load_session(path: str | Path) -> AlignmentEngine:
    """Rebuild an engine from a session file, history and all."""
    h5py = _h5py()

    with h5py.File(path, "r") as handle:
        version = int(handle.attrs.get("session_version", 0))
        if version != _SESSION_VERSION:
            raise ValueError(
                f"Session format v{version} is not supported (expected v{_SESSION_VERSION})."
            )

        config = AlignConfig(**_coerce_config(json.loads(handle.attrs["config"])))
        metadata = json.loads(handle.attrs.get("metadata", "{}"))

        dataset = ProjectionData(
            data=handle["projections"][()],
            angles=handle["angles"][()],
            metadata=metadata,
        )
        # pad is already baked into the stored projections; re-applying it on
        # construction would pad the data a second time.
        engine = AlignmentEngine(
            dataset=dataset,
            config=AlignConfig(**{**config.to_dict(), "pad": (0, 0)}),
            sx0=handle["sx"][()],
            sy0=handle["sy"][()],
            center=float(handle.attrs["center"]),
        )
        engine.config = config

        if "volume" in handle:
            engine.state.volume = handle["volume"][()]

        history = []
        for key in sorted(handle["history"], key=int):
            group = handle["history"][key]
            history.append(
                IterationResult(
                    iteration=int(key),
                    sx=group["sx"][()],
                    sy=group["sy"][()],
                    dsx=group["dsx"][()],
                    dsy=group["dsy"][()],
                    error=float(group.attrs["error"]),
                    residual=float(group.attrs["residual"]),
                    center=float(group.attrs["center"]),
                    wallclock_s=float(group.attrs["wallclock_s"]),
                    config_changed=bool(group.attrs.get("config_changed", False)),
                    volume=group["volume"][()] if "volume" in group else None,
                )
            )
        engine.state.history = history

    return engine


def _coerce_config(raw: dict[str, Any]) -> dict[str, Any]:
    # JSON has no tuples; AlignConfig.pad must come back as one.
    if "pad" in raw and raw["pad"] is not None:
        raw["pad"] = tuple(raw["pad"])
    return raw


def _jsonable(metadata: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, (str, int, float, bool, type(None), list, dict)):
            out[key] = value
        else:
            out[key] = str(value)
    return out


# -- exports -----------------------------------------------------------------------


def export_shifts_csv(path: str | Path, angles: np.ndarray, sx: np.ndarray, sy: np.ndarray) -> None:
    table = np.column_stack([angles, np.rad2deg(angles), sx, sy])
    np.savetxt(
        path,
        table,
        delimiter=",",
        header="angle_rad,angle_deg,sx,sy",
        comments="",
        fmt="%.8g",
    )


def export_convergence_csv(path: str | Path, history: list[IterationResult]) -> None:
    table = np.array(
        [[r.iteration, r.error, r.residual, r.center, r.wallclock_s] for r in history]
    )
    np.savetxt(
        path,
        table.reshape(-1, 5),
        delimiter=",",
        header="iteration,shift_rms,residual,center,wallclock_s",
        comments="",
        fmt="%.8g",
    )


def export_tiff_stack(path: str | Path, array: np.ndarray) -> None:
    import tifffile  # noqa: PLC0415

    tifffile.imwrite(str(path), np.asarray(array, dtype=np.float32))


def export_volume(path: str | Path, volume: np.ndarray, angles: np.ndarray | None = None) -> None:
    """Write a volume as TIFF or HDF5, chosen by the file suffix."""
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        export_tiff_stack(path, volume)
        return

    from tktomo.io import save_volume  # noqa: PLC0415

    save_volume(path, volume, angles=angles, metadata={"source": "ptycho-align"})


def export_projections(
    path: str | Path, data: ProjectionData
) -> None:
    """Write an aligned projection stack as TIFF or HDF5, chosen by the suffix."""
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        export_tiff_stack(path, data.data)
        return

    from tktomo.io import save_projections  # noqa: PLC0415

    save_projections(path, data)
