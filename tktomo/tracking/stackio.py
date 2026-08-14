"""Reading and writing the tracking apps' HDF5 stack formats.

Two formats are recognized by signature, so "Load stack" needs no format
dropdown:

- slogger preproc: datasets `proj` (n, h, w) + `theta_rad` (n,), attrs
  `binning` (int) and `crop` (v0, v1, u0, u1 in raw px). Written by the
  slogger graphite-ball pipeline's preprocess stage.
- feature crop: the same plus a `crop_origin` (n, 2) [v, u] dataset of
  per-view window origins and the marker attr `tktomo_feature_crop`.
  Written by the feature-isolation app; the origins are what let a label
  clicked in the moving window map back to the raw detector frame.

Everything else (generic HDF5, TIFF directories) goes through
`tktomo.ptycho_align.core.dataset.load_dataset` in the UI layer, with the
provenance (binning/crop) typed by the user.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tktomo.io import ProjectionData
from tktomo.tracking.coords import CoordinateChain

FEATURE_CROP_MARKER = "tktomo_feature_crop"


def detect_format(path: str | Path) -> str | None:
    """"slogger_preproc", "feature_crop", or None (not ours)."""
    import h5py  # noqa: PLC0415

    path = Path(path)
    if path.suffix.lower() not in (".h5", ".hdf5", ".nx", ".nxs"):
        return None
    try:
        with h5py.File(path, "r") as f:
            if "proj" not in f or "theta_rad" not in f:
                return None
            if FEATURE_CROP_MARKER in f.attrs and "crop_origin" in f:
                return "feature_crop"
            return "slogger_preproc"
    except OSError:
        return None


def load_tracking_stack(path: str | Path
                        ) -> tuple[ProjectionData, CoordinateChain]:
    """Load a recognized stack with its coordinate chain from the attrs."""
    import h5py  # noqa: PLC0415

    path = Path(path)
    kind = detect_format(path)
    if kind is None:
        raise ValueError(f"{path.name} is not a recognized tracking stack "
                         f"(needs `proj` + `theta_rad` datasets)")
    with h5py.File(path, "r") as f:
        proj = np.asarray(f["proj"][()], np.float32)
        theta = np.asarray(f["theta_rad"][()], float)
        binning = int(f.attrs.get("binning", 1))
        # slogger writes extra_bin when recon binned further; compose it
        binning *= int(f.attrs.get("extra_bin", 1))
        crop = tuple(int(x) for x in f.attrs.get("crop", (0, 0, 0, 0)))
        origin = None
        if kind == "feature_crop":
            origin = np.asarray(f["crop_origin"][()], float)
        metadata = {
            "source_path": str(path),
            "format": kind,
            "binning": binning,
            "crop": crop,
        }
        for key in ("window", "source", "keyframes", "u_mode", "v_mode",
                    "sign"):
            if key in f.attrs:
                val = f.attrs[key]
                metadata[key] = (val.tolist() if isinstance(val, np.ndarray)
                                 else val)
    chain = CoordinateChain(binning=binning, crop=crop, view_origin=origin)
    return ProjectionData(data=proj, angles=theta, metadata=metadata), chain


def compose_view_origin(chain: CoordinateChain,
                        window_origin: np.ndarray) -> np.ndarray:
    """Express per-view window origins in the source FILE's binned frame.

    `window_origin` is (n, 2) [v, u] in the frame the user was looking at
    (the loaded frame). Whatever offsets the chain already carries
    (extra_crop, an earlier per-view origin) are folded in, so the written
    file needs only `binning`, `crop` and the composed `crop_origin` to map
    back to raw.
    """
    origin = np.asarray(window_origin, float).copy()
    if origin.ndim != 2 or origin.shape[1] != 2:
        raise ValueError("window_origin must have shape (n_views, 2)")
    if chain.extra_crop is not None:
        origin[:, 0] += chain.extra_crop[0]
        origin[:, 1] += chain.extra_crop[2]
    if chain.view_origin is not None:
        if chain.view_origin.shape[0] != origin.shape[0]:
            raise ValueError("view counts of chain and window differ")
        origin += chain.view_origin
    return origin


def write_feature_crop(path: str | Path, proj: np.ndarray, theta: np.ndarray,
                       window_origin: np.ndarray, chain: CoordinateChain, *,
                       window: tuple[int, int], source: dict | None = None,
                       keyframes: list | None = None,
                       u_mode: str = "sinusoid",
                       v_mode: str = "linear") -> None:
    """Write a feature-crop stack the model-fitting app can load."""
    import h5py  # noqa: PLC0415

    origin = compose_view_origin(chain, window_origin)
    if proj.shape[0] != origin.shape[0] or proj.shape[0] != theta.size:
        raise ValueError("proj, theta and window_origin view counts differ")
    with h5py.File(path, "w") as f:
        f.create_dataset("proj", data=np.asarray(proj, np.float32),
                         compression="gzip")
        f.create_dataset("theta_rad", data=np.asarray(theta, float))
        f.create_dataset("crop_origin", data=origin)
        f.attrs[FEATURE_CROP_MARKER] = 1
        f.attrs["binning"] = int(chain.binning)
        f.attrs["crop"] = [int(x) for x in chain.crop]
        f.attrs["window"] = [int(window[0]), int(window[1])]
        f.attrs["u_mode"] = u_mode
        f.attrs["v_mode"] = v_mode
        if source is not None:
            f.attrs["source"] = json.dumps(source)
        if keyframes is not None:
            f.attrs["keyframes"] = json.dumps(keyframes)


def crop_track_windows(proj: np.ndarray, track_u: np.ndarray,
                       track_v: np.ndarray, window: tuple[int, int]
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Cut a fixed (h, w) window centred on the track, per view.

    Returns (cropped (n, h, w), origin (n, 2) [v, u] in the input frame).
    Windows are clamped inside the stack, so the feature drifts inside the
    window near the edges rather than the crop reading out of range.
    """
    n, full_h, full_w = proj.shape
    h, w = int(window[0]), int(window[1])
    if h > full_h or w > full_w:
        raise ValueError(f"window {h}x{w} exceeds the stack {full_h}x{full_w}")
    origin = np.zeros((n, 2), int)
    out = np.empty((n, h, w), proj.dtype)
    for k in range(n):
        v0 = int(np.clip(round(float(track_v[k]) - h / 2.0), 0, full_h - h))
        u0 = int(np.clip(round(float(track_u[k]) - w / 2.0), 0, full_w - w))
        origin[k] = (v0, u0)
        out[k] = proj[k, v0:v0 + h, u0:u0 + w]
    return out, origin
