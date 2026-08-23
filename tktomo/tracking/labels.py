"""Label storage and keyframe interpolation for manual feature tracking.

Labels live in RAW-grid coordinates (see `tktomo.tracking.coords`): the UI
converts a click through its `CoordinateChain` at placement time, so a
label survives reloading the same data at a different binning or crop.

Every label carries a provenance kind: MANUAL (a human click) or AUTO
(placed by the template matcher in `tktomo.tracking.autotrack`), plus a
match quality for auto labels. The rules are asymmetric on purpose: a
manual click OVERWRITES an auto label (the human wins), while the
auto-tracker refuses to touch a manual label. Both kinds enter the model
fit at full weight; provenance exists so the machine's work is visible,
reviewable, and bulk-removable, never so it is silently trusted less.
"""

from __future__ import annotations

import numpy as np

KIND_MANUAL = 0
KIND_AUTO = 1


class LabelStore:
    """{(feature_id, view): (u_raw, v_raw, kind, quality)}.

    Per-(feature, view) uniqueness is enforced on set: one physical
    feature cannot be in two places in one projection.
    """

    def __init__(self) -> None:
        self._points: dict[tuple[int, int],
                           tuple[float, float, int, float]] = {}

    def __len__(self) -> int:
        return len(self._points)

    # -- writing ----------------------------------------------------------

    def set(self, feature_id: int, view: int, u: float, v: float) -> None:
        """Place/move a MANUAL label. Overwrites anything, auto included."""
        self._points[(int(feature_id), int(view))] = (
            float(u), float(v), KIND_MANUAL, float("nan"))

    def set_auto(self, feature_id: int, view: int, u: float, v: float,
                 quality: float) -> bool:
        """Place an AUTO label. Refuses to overwrite a manual one."""
        key = (int(feature_id), int(view))
        existing = self._points.get(key)
        if existing is not None and existing[2] == KIND_MANUAL:
            return False
        self._points[key] = (float(u), float(v), KIND_AUTO, float(quality))
        return True

    def remove(self, feature_id: int, view: int) -> bool:
        return self._points.pop((int(feature_id), int(view)), None) is not None

    def clear_feature(self, feature_id: int) -> int:
        keys = [k for k in self._points if k[0] == feature_id]
        for k in keys:
            del self._points[k]
        return len(keys)

    def clear_auto(self, feature_id: int | None = None) -> int:
        """Delete auto labels: one feature's, or every feature's (None)."""
        keys = [k for k, val in self._points.items()
                if val[2] == KIND_AUTO
                and (feature_id is None or k[0] == feature_id)]
        for k in keys:
            del self._points[k]
        return len(keys)

    # -- reading ----------------------------------------------------------

    def get(self, feature_id: int, view: int) -> tuple[float, float] | None:
        val = self._points.get((int(feature_id), int(view)))
        return None if val is None else (val[0], val[1])

    def kind_of(self, feature_id: int, view: int) -> int | None:
        val = self._points.get((int(feature_id), int(view)))
        return None if val is None else val[2]

    def quality_of(self, feature_id: int, view: int) -> float:
        val = self._points.get((int(feature_id), int(view)))
        return float("nan") if val is None else val[3]

    def nearest(self, view: int, u: float, v: float
                ) -> tuple[int, float] | None:
        """(feature_id, distance) of the closest label in `view`, or None."""
        best = None
        for (fid, w), val in self._points.items():
            if w != view:
                continue
            d = float(np.hypot(val[0] - u, val[1] - v))
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

    def manual_counts(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for (fid, _), val in self._points.items():
            if val[2] == KIND_MANUAL:
                out[fid] = out.get(fid, 0) + 1
        return out

    def views_of(self, feature_id: int) -> list[int]:
        return sorted(w for fid, w in self._points if fid == feature_id)

    def manual_views_of(self, feature_id: int) -> list[int]:
        return sorted(w for (fid, w), val in self._points.items()
                      if fid == feature_id and val[2] == KIND_MANUAL)

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
        return [(fid, val[0], val[1])
                for (fid, w), val in sorted(self._points.items())
                if w == view]

    def in_view_full(self, view: int
                     ) -> list[tuple[int, float, float, int, float]]:
        """[(feature_id, u, v, kind, quality)] of every label in one view."""
        return [(fid, val[0], val[1], val[2], val[3])
                for (fid, w), val in sorted(self._points.items())
                if w == view]

    # -- solver interface -------------------------------------------------

    def to_arrays(self, n_views: int, feature_ids=None
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(u, v, valid, ids): (F, V) label arrays for `solve_model`.

        Both manual and auto labels enter, at full weight. Rows follow
        `feature_ids` when given (so pinned-feature indices stay stable),
        otherwise the sorted ids present in the store.
        """
        ids = np.asarray(self.feature_ids() if feature_ids is None
                         else feature_ids, int)
        row = {int(fid): k for k, fid in enumerate(ids)}
        u = np.zeros((ids.size, n_views))
        v = np.zeros((ids.size, n_views))
        valid = np.zeros((ids.size, n_views), bool)
        for (fid, w), val in self._points.items():
            if fid in row and 0 <= w < n_views:
                u[row[fid], w] = val[0]
                v[row[fid], w] = val[1]
                valid[row[fid], w] = True
        return u, v, valid, ids

    # -- (de)serialization ------------------------------------------------

    def to_table(self) -> np.ndarray:
        """(M, 6) float array [feature_id, view, u_raw, v_raw, kind,
        quality], sorted."""
        rows = [(fid, w, val[0], val[1], val[2], val[3])
                for (fid, w), val in sorted(self._points.items())]
        return (np.asarray(rows, float) if rows
                else np.zeros((0, 6)))

    @classmethod
    def from_table(cls, table: np.ndarray) -> "LabelStore":
        """Accepts the current (M, 6) layout and the legacy (M, 4) one
        (pre-provenance files: everything manual, quality NaN)."""
        table = np.asarray(table, float)
        store = cls()
        if table.size == 0:
            return store
        if table.ndim != 2 or table.shape[1] not in (4, 6):
            raise ValueError(
                f"label table must be (M, 4) or (M, 6), got {table.shape}")
        for row in table:
            fid, view, u, v = int(row[0]), int(row[1]), row[2], row[3]
            if table.shape[1] == 6 and int(row[4]) == KIND_AUTO:
                store.set_auto(fid, view, u, v, row[5])
            else:
                store.set(fid, view, u, v)
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


def reject_auto_outliers(store: "LabelStore", fit, feature_ids,
                         limit: float) -> int:
    """Remove AUTO labels whose fit residual exceeds `limit` (raw px).

    `fit` is a `FitResult` from `solve_model`/`residuals` over arrays built
    with `store.to_arrays(n_views, feature_ids)`, so `fit.obs` indexes rows
    of `feature_ids`. Manual labels are never touched: a human click is
    evidence, an auto label is a guess. Returns the number removed.

    Why this exists: the Huber loop down-weights a 15 px lock-on to ~0.3,
    which still drags a thinly covered view's free shift by several px.
    Dropping it and refitting once recovered most of the gap between the
    phase-correlation completer and a learned confidence in the end-to-end
    alignment test (dx rms 2.95 -> 2.43 raw px at 10-view anchors).
    """
    ids = np.asarray(feature_ids, int)
    i, j = fit.obs
    bad = (np.abs(fit.residual_u) > limit) | (np.abs(fit.residual_v) > limit)
    n = 0
    for fi, vj in zip(i[bad], j[bad]):
        fid, view = int(ids[fi]), int(vj)
        if store.kind_of(fid, view) == KIND_AUTO:
            store.remove(fid, view)
            n += 1
    return n
