"""Shared synthetic stacks for the tracking tests."""

from __future__ import annotations

import numpy as np


def blob_stack(n_views=40, ny=72, nx=112):
    """One crisp blob on a known sinusoid: (stack, theta, truth_u, truth_v)."""
    theta = np.linspace(0, np.pi, n_views)
    truth_u = 25.0 * np.cos(theta) + 55.0
    truth_v = np.full(n_views, 36.0)
    rng = np.random.default_rng(0)
    stack = np.empty((n_views, ny, nx), np.float32)
    yy, xx = np.mgrid[0:ny, 0:nx]
    for j in range(n_views):
        frame = 0.02 * rng.standard_normal((ny, nx))
        frame += np.exp(-((yy - truth_v[j]) ** 2 + (xx - truth_u[j]) ** 2)
                        / (2 * 2.5 ** 2))
        stack[j] = frame
    return stack, theta, truth_u, truth_v


def write_blob_file(path, **kwargs):
    """The blob stack as a slogger-preproc file; returns (path, stack, theta, u, v)."""
    import h5py

    stack, theta, u, v = blob_stack(**kwargs)
    with h5py.File(path, "w") as f:
        f["proj"] = stack
        f["theta_rad"] = theta
        f.attrs["binning"] = 2
        f.attrs["crop"] = (10, 154, 20, 244)
    return str(path), stack, theta, u, v
