"""Save/load of a track-model app session (labels + model + masks + UI).

The projections are deliberately NOT in the session file: a session is a
few kilobytes next to a half-gigabyte stack, and the labels are stored in
RAW-grid coordinates, so re-reading the stack from its source (recorded in
the `source` attr) reproduces the exact state even if the stack is later
re-cropped or re-binned from the same parent data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tktomo.tracking.labels import LabelStore
from tktomo.tracking.model import AxisModel, FreeMask

# version 2: label_table gained provenance columns (kind, quality);
# version-1 files are still readable (LabelStore.from_table accepts both).
# version 3: per-view rotations (rot_horiz, rot_beam, rot_axis) and their
# free flags; older files load with the rotations at zero and fixed.
SESSION_VERSION = 3
ROTATIONS = ("rot_horiz", "rot_beam", "rot_axis")


def save_session(path: str | Path, *, labels: LabelStore,
                 model: AxisModel | None, mask: FreeMask | None,
                 source: dict, ui_state: dict) -> None:
    import h5py  # noqa: PLC0415

    with h5py.File(path, "w") as f:
        f.attrs["session_version"] = SESSION_VERSION
        f.attrs["source"] = json.dumps(source)
        f.attrs["ui"] = json.dumps(ui_state)
        f.create_dataset("label_table", data=labels.to_table())
        if model is not None and mask is not None:
            f.attrs["has_model"] = True
            f.attrs["free_dx"] = bool(mask.dx)
            f.attrs["free_dy"] = bool(mask.dy)
            for name in ROTATIONS:
                f.attrs[f"free_{name}"] = bool(getattr(mask, name))
            f.attrs["model"] = json.dumps({
                "theta_ref": model.theta_ref,
                "theta_scale": model.theta_scale,
            })
            f.create_dataset("theta_rad", data=model.theta)
            for name in ("c_coef", "alpha_coef", "beta_coef", "dx", "dy",
                         "a", "b", "y", *ROTATIONS):
                f.create_dataset(name, data=getattr(model, name))
            f.create_dataset("feature_id", data=model.feature_ids)
            f.create_dataset("free_c", data=mask.c)
            f.create_dataset("free_alpha", data=mask.alpha)
            f.create_dataset("free_beta", data=mask.beta)
            f.create_dataset("free_features", data=mask.features)
        else:
            f.attrs["has_model"] = False


def load_session(path: str | Path) -> dict:
    """Returns {labels, model|None, mask|None, source, ui}."""
    import h5py  # noqa: PLC0415

    with h5py.File(path, "r") as f:
        version = int(f.attrs.get("session_version", -1))
        if version not in (1, 2, SESSION_VERSION):
            raise ValueError(
                f"session_version {version} is not 1, 2 or {SESSION_VERSION}")
        out = {
            "labels": LabelStore.from_table(f["label_table"][()]),
            "source": json.loads(f.attrs["source"]),
            "ui": json.loads(f.attrs["ui"]),
            "model": None,
            "mask": None,
        }
        if f.attrs.get("has_model"):
            info = json.loads(f.attrs["model"])
            theta = f["theta_rad"][()]
            out["model"] = AxisModel(
                theta=theta,
                c_coef=f["c_coef"][()], alpha_coef=f["alpha_coef"][()],
                beta_coef=f["beta_coef"][()],
                dx=f["dx"][()], dy=f["dy"][()],
                feature_ids=f["feature_id"][()].astype(int),
                a=f["a"][()], b=f["b"][()], y=f["y"][()],
                theta_ref=float(info["theta_ref"]),
                theta_scale=float(info["theta_scale"]),
                **{name: (f[name][()] if name in f else np.zeros(theta.size))
                   for name in ROTATIONS},
            )
            out["mask"] = FreeMask(
                dx=bool(f.attrs["free_dx"]), dy=bool(f.attrs["free_dy"]),
                c=f["free_c"][()].astype(bool),
                alpha=f["free_alpha"][()].astype(bool),
                beta=f["free_beta"][()].astype(bool),
                features=f["free_features"][()].astype(bool),
                **{name: bool(f.attrs.get(f"free_{name}", False))
                   for name in ROTATIONS},
            )
    return out
