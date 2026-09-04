"""Coordinate bookkeeping between raw, preprocessed and cropped grids.

A projection stack usually reaches the tracking UIs after a chain of
transformations of the raw detector frame: a crop (in raw pixels), a
binning, possibly a further crop applied at load time, and possibly a
per-view moving crop produced by the feature-isolation app. Labels are
clicked in the loaded frame but every fitted quantity must be reported on
the unbinned parent ("raw") grid, otherwise a center fitted on one stack
means nothing on another.

The whole fit therefore runs in raw coordinates. This module is the single
place where the conversion lives.

Pixel-center convention: loaded pixel index k covers raw pixels
[u0 + k*b, u0 + (k+1)*b) and its center sits at u0 + k*b + (b-1)/2, the
inverse of the slogger pipeline's `center = c*b + (b-1)/2` rule. Shifts
(differences of positions) scale by the binning alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class CoordinateChain:
    """Maps (u, v) in the loaded frame to the raw parent grid and back.

    binning     loaded px -> raw px scale (composed over all binning steps).
    crop        (v0, v1, u0, u1) in RAW pixels, applied before the binning.
                v1/u1 are carried for provenance; only v0/u0 enter the math.
    extra_crop  optional (v0, v1, u0, u1) crop applied at load time, in the
                loaded (binned) frame of the file.
    view_origin optional (n_views, 2) [v, u] per-view crop origins in the
                file's binned frame, written by the feature-isolation app.
    rebin       a further mean-pool applied to the loaded frame at run time
                (the "bin" control of the track-model app), so the grid on
                screen is the file's grid divided by `rebin`. Applied first,
                before everything above, which all describes the file.

    `scale` (binning * rebin) is the loaded px -> raw px factor.
    """

    binning: int = 1
    crop: tuple[int, int, int, int] = (0, 0, 0, 0)
    extra_crop: tuple[int, int, int, int] | None = None
    view_origin: np.ndarray | None = None
    rebin: int = 1

    def __post_init__(self) -> None:
        if self.binning < 1:
            raise ValueError(f"binning must be >= 1, got {self.binning}")
        if self.rebin < 1:
            raise ValueError(f"rebin must be >= 1, got {self.rebin}")
        if self.view_origin is not None:
            origin = np.asarray(self.view_origin, float)
            if origin.ndim != 2 or origin.shape[1] != 2:
                raise ValueError("view_origin must have shape (n_views, 2)")
            object.__setattr__(self, "view_origin", origin)

    # -- loaded frame -> file's binned frame ------------------------------

    @property
    def scale(self) -> int:
        """Loaded px per raw px, everything composed."""
        return int(self.binning) * int(self.rebin)

    def with_rebin(self, rebin: int) -> "CoordinateChain":
        """The same file provenance, shown at another run-time binning."""
        return CoordinateChain(binning=self.binning, crop=self.crop,
                               extra_crop=self.extra_crop,
                               view_origin=self.view_origin, rebin=int(rebin))

    def _file_frame(self, u, v, view):
        u = np.asarray(u, float)
        v = np.asarray(v, float)
        if self.rebin > 1:
            k, half = self.rebin, (self.rebin - 1) / 2.0
            u = u * k + half
            v = v * k + half
        if self.extra_crop is not None:
            v = v + self.extra_crop[0]
            u = u + self.extra_crop[2]
        if self.view_origin is not None:
            if view is None:
                raise ValueError("view index required: chain has per-view origins")
            view = np.asarray(view, int)
            v = v + self.view_origin[view, 0]
            u = u + self.view_origin[view, 1]
        return u, v

    # -- public conversions ----------------------------------------------

    def to_parent(self, u, v, view=None) -> tuple[np.ndarray, np.ndarray]:
        """Loaded-frame (u, v) [+ view when per-view origins exist] -> raw px."""
        u, v = self._file_frame(u, v, view)
        b = self.binning
        half = (b - 1) / 2.0
        return (self.crop[2] + u * b + half, self.crop[0] + v * b + half)

    def from_parent(self, u_raw, v_raw, view=None) -> tuple[np.ndarray, np.ndarray]:
        """Raw px -> loaded-frame (u, v), the exact inverse of `to_parent`."""
        b = self.binning
        half = (b - 1) / 2.0
        u = (np.asarray(u_raw, float) - self.crop[2] - half) / b
        v = (np.asarray(v_raw, float) - self.crop[0] - half) / b
        if self.view_origin is not None:
            if view is None:
                raise ValueError("view index required: chain has per-view origins")
            view = np.asarray(view, int)
            v = v - self.view_origin[view, 0]
            u = u - self.view_origin[view, 1]
        if self.extra_crop is not None:
            v = v - self.extra_crop[0]
            u = u - self.extra_crop[2]
        if self.rebin > 1:
            k, half = self.rebin, (self.rebin - 1) / 2.0
            u = (u - half) / k
            v = (v - half) / k
        return u, v

    def shift_to_parent(self, d):
        """A displacement in loaded px is a displacement in raw px times scale."""
        return np.asarray(d, float) * self.scale

    def shift_from_parent(self, d_raw):
        return np.asarray(d_raw, float) / self.scale

    def parent_to_grid(self, u_raw, grid_binning: int):
        """Raw px -> a grid of the SAME raw crop but binning `grid_binning`.

        This is how a raw-frame center is expressed on e.g. the preproc grid
        (grid_binning = 2 for the slogger stacks): the inverse of
        `center = c*b + (b-1)/2` composed with the raw crop offset.
        """
        g = int(grid_binning)
        if g < 1:
            raise ValueError(f"grid_binning must be >= 1, got {g}")
        return (np.asarray(u_raw, float) - self.crop[2] - (g - 1) / 2.0) / g

    def grid_to_parent(self, u_grid, grid_binning: int):
        g = int(grid_binning)
        return self.crop[2] + np.asarray(u_grid, float) * g + (g - 1) / 2.0

    # -- provenance -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "binning": int(self.binning),
            "crop": [int(x) for x in self.crop],
            "extra_crop": (None if self.extra_crop is None
                           else [int(x) for x in self.extra_crop]),
            "has_view_origin": self.view_origin is not None,
            "rebin": int(self.rebin),
        }


def regrid_uv(u, v, src_bin: int, dst_bin: int):
    """(u, v) on a grid mean-pooled by `src_bin` expressed on the grid
    mean-pooled by `dst_bin`, both pools of the same parent frame.

    Goes through `CoordinateChain` so the pixel-centre rule has one home:
    pixel 0 of a bin 4 grid is centred on pixel 1.5 of the parent.
    """
    src, dst = int(src_bin), int(dst_bin)
    if src == dst:
        return np.asarray(u, float), np.asarray(v, float)
    parent = CoordinateChain(rebin=src).to_parent(u, v)
    return CoordinateChain(rebin=dst).from_parent(*parent)
