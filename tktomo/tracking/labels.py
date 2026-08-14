"""Label storage and keyframe interpolation for manual feature tracking.

Labels live in RAW-grid coordinates (see `tktomo.tracking.coords`): the UI
converts a click through its `CoordinateChain` at placement time, so a
label survives reloading the same data at a different binning or crop.
"""

from __future__ import annotations

import numpy as np


class LabelStore:
    """{(feature_id, view): (u_raw, v_raw)} with per-(feature, view) uniqueness.

    Placing a feature again in the same view moves the existing point: one
    physical feature cannot be in two places in one projection.
    """

    def __init__(self) -> None:
        self._points: dict[tuple[int, int], tuple[float, float]] = {}

    def __len__(self) -> int:
        return len(self._points)

    def set(self, feature_id: int, view: int, u: float, v: float) -> None:
        self._points[(int(feature_id), int(view))] = (float(u), float(v))

    def get(self, feature_id: int, view: int) -> tuple[float, float] | None:
        return self._points.get((int(feature_id), int(view)))

    def remove(self, feature_id: int, view: int) -> bool:
        return self._points.pop((int(feature_id), int(view)), None) is not None

    def nearest(self, view: int, u: float, v: float
                ) -> tuple[int, float] | None:
        """(feature_id, distance) of the closest label in `view`, or None."""
        best = None
        for (fid, w), (pu, pv) in self._points.items():
            if w != view:
                continue
            d = float(np.hypot(pu - u, pv - v))
            if best is None or d < best[1]:
                best = (fid, d)
        return best

    def feature_ids(self) -> list[int]:
        return sorted({fid for fid, _ in self._points})

    def counts(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for fid, _ in self._points:
            out[fid] = out.get(fid, 0) + 1
        return out

    def views_of(self, feature_id: int) -> list[int]:
        return sorted(w for fid, w in self._points if fid == feature_id)

    def counts_per_view(self, n_views: int) -> np.ndarray:
        """(n_views,) int: how many labels each view carries. The zeros
        are the frames still missing manual data."""
        counts = np.zeros(int(n_views), int)
        for _fid, view in self._points:
            if 0 <= view < counts.size:
                counts[view] += 1
        return counts

    def in_view(self, view: int) -> list[tuple[int, float, float]]:
        """[(feature_id, u, v)] of every label in one view."""
        return [(fid, uv[0], uv[1])
                for (fid, w), uv in sorted(self._points.items()) if w == view]

    def clear_feature(self, feature_id: int) -> int:
        keys = [k for k in self._points if k[0] == feature_id]
        for k in keys:
            del self._points[k]
        return len(keys)

    # -- solver interface -------------------------------------------------

    def to_arrays(self, n_views: int, feature_ids=None
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(u, v, valid, ids): (F, V) label arrays for `solve_model`.

        Rows follow `feature_ids` when given (so pinned-feature indices stay
        stable), otherwise the sorted ids present in the store.
        """
        ids = np.asarray(self.feature_ids() if feature_ids is None
                         else feature_ids, int)
        row = {int(fid): k for k, fid in enumerate(ids)}
        u = np.zeros((ids.size, n_views))
        v = np.zeros((ids.size, n_views))
        valid = np.zeros((ids.size, n_views), bool)
        for (fid, w), (pu, pv) in self._points.items():
            if fid in row and 0 <= w < n_views:
                u[row[fid], w] = pu
                v[row[fid], w] = pv
                valid[row[fid], w] = True
        return u, v, valid, ids

    # -- (de)serialization ------------------------------------------------

    def to_table(self) -> np.ndarray:
        """(M, 4) float array [feature_id, view, u_raw, v_raw], sorted."""
        rows = [(fid, w, uv[0], uv[1])
                for (fid, w), uv in sorted(self._points.items())]
        return (np.asarray(rows, float) if rows
                else np.zeros((0, 4)))

    @classmethod
    def from_table(cls, table: np.ndarray) -> "LabelStore":
        store = cls()
        for fid, w, u, v in np.asarray(table, float).reshape(-1, 4):
            store.set(int(fid), int(w), u, v)
        return store


def interpolate_track(theta: np.ndarray, key_views, key_u, key_v, *,
                      u_mode: str = "sinusoid", v_mode: str = "linear"
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a single feature's (u, v) over all views from keyframes.

    u_mode "sinusoid" fits u = a*cos(theta) + b*sin(theta) + c, which is
    what a rigid point on a rotating object does; it needs >= 3 keyframes
    and extrapolates physically outside their span. "linear" (and every
    fallback) interpolates in theta, holding the edge values outside the
    keyframe span. v_mode "spline" is a PCHIP (no overshoot); "linear" as
    for u. Keyframes need not be sorted.

    Returns (u(V,), v(V,)).
    """
    theta = np.asarray(theta, float)
    key_views = np.asarray(key_views, int)
    key_u = np.asarray(key_u, float)
    key_v = np.asarray(key_v, float)
    if key_views.size == 0:
        raise ValueError("at least one keyframe is required")
    order = np.argsort(theta[key_views])
    t_key = theta[key_views][order]
    u_key = key_u[order]
    v_key = key_v[order]

    if key_views.size == 1:
        return (np.full(theta.size, u_key[0]), np.full(theta.size, v_key[0]))

    def linear(vals):
        return np.interp(theta, t_key, vals)

    def pchip(vals):
        if t_key.size < 3:
            return linear(vals)
        from scipy.interpolate import PchipInterpolator  # noqa: PLC0415
        out = linear(vals)
        inside = (theta >= t_key[0]) & (theta <= t_key[-1])
        out[inside] = PchipInterpolator(t_key, vals)(theta[inside])
        return out

    if u_mode == "sinusoid" and t_key.size >= 3:
        g = np.column_stack([np.cos(t_key), np.sin(t_key),
                             np.ones(t_key.size)])
        coef, *_ = np.linalg.lstsq(g, u_key, rcond=None)
        u_all = (coef[0] * np.cos(theta) + coef[1] * np.sin(theta) + coef[2])
    elif u_mode in ("sinusoid", "linear"):
        u_all = linear(u_key)
    else:
        raise ValueError(f"unknown u_mode {u_mode!r}")

    if v_mode == "spline":
        v_all = pchip(v_key)
    elif v_mode == "linear":
        v_all = linear(v_key)
    else:
        raise ValueError(f"unknown v_mode {v_mode!r}")
    return u_all, v_all


def sinusoid_fit_info(theta, key_views, key_u) -> dict | None:
    """(a, b, c, rms, amplitude) of the keyframes' sinusoid, for the UI readout."""
    key_views = np.asarray(key_views, int)
    if key_views.size < 3:
        return None
    t = np.asarray(theta, float)[key_views]
    u = np.asarray(key_u, float)
    g = np.column_stack([np.cos(t), np.sin(t), np.ones(t.size)])
    coef, *_ = np.linalg.lstsq(g, u, rcond=None)
    res = u - g @ coef
    return {"a": float(coef[0]), "b": float(coef[1]), "c": float(coef[2]),
            "rms": float(np.sqrt(np.mean(res ** 2))),
            "amplitude": float(np.hypot(coef[0], coef[1]))}
