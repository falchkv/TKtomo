"""Residual rejection removes auto outliers only, and refit improves."""

from __future__ import annotations

import numpy as np

from tktomo.tracking.labels import LabelStore, reject_auto_outliers
from tktomo.tracking.model import AxisModel, solve_model


def _scene(n_views=60, seed=0):
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, np.pi, n_views)
    feats = [(300.0, -120.0, 40.0), (-200.0, 250.0, 90.0), (50.0, 400.0, 140.0)]
    dx = 3.0 * np.sin(4 * theta)
    store = LabelStore()
    for f, (a, b, y) in enumerate(feats):
        for j in range(n_views):
            u = a * np.cos(theta[j]) + b * np.sin(theta[j]) + 1000.0 + dx[j]
            v = y + 0.3 * rng.standard_normal()
            u += 0.3 * rng.standard_normal()
            if j % 10 == 0:
                store.set(f, j, u, v)                 # manual anchors
            else:
                store.set_auto(f, j, u, v, 0.9)
    return theta, store


def test_reject_auto_outliers_removes_only_bad_auto_labels():
    theta, store = _scene()
    ids = np.array([0, 1, 2])
    # plant lock-ons: one auto, one MANUAL (must survive)
    u, v = store.get(1, 23)
    store.set_auto(1, 23, u + 18.0, v, 0.9)
    u, v = store.get(2, 30)
    store.set(2, 30, u + 18.0, v)

    u, v, valid, ids = store.to_arrays(theta.size, ids)
    fit = solve_model(u, v, valid, AxisModel.blank(theta, ids), huber=3.0)
    n = reject_auto_outliers(store, fit, ids, limit=9.0)
    assert n == 1
    assert store.get(1, 23) is None
    assert store.get(2, 30) is not None and store.kind_of(2, 30) == 0

    u2, v2, valid2, _ = store.to_arrays(theta.size, ids)
    fit2 = solve_model(u2, v2, valid2, fit.model, huber=3.0)
    assert fit2.rms_u < fit.rms_u
