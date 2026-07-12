"""Preprocessing tests. The ramp test is the important one: a residual linear phase
ramp is indistinguishable from a lateral shift, so it must come off cleanly."""

from __future__ import annotations

import numpy as np
import pytest

from tktomo.ptycho_align.core.preprocess import (
    background_mask,
    check_mass_positive,
    crop,
    invert,
    pad,
    remove_phase_offset,
    remove_phase_ramp,
)


def _blob(n_v: int = 40, n_u: int = 48) -> np.ndarray:
    """One projection: a compact blob well inside a flat, zero background.

    sigma is kept small on purpose: the blob's tails must die away to far below the
    test tolerance before they reach the border frame, or the "background" the plane
    is fitted over would contain object and the fit would not be exact.
    """
    v, u = np.mgrid[0:n_v, 0:n_u].astype(np.float32)
    return np.exp(-(((v - n_v / 2) ** 2 + (u - n_u / 2) ** 2) / (2 * 2.0**2))).astype(
        np.float32
    )


def test_remove_phase_ramp_removes_a_known_ramp():
    clean = np.stack([_blob(), _blob() * 0.7])
    n_v, n_u = clean.shape[1:]
    v, u = np.mgrid[0:n_v, 0:n_u].astype(np.float32)

    # A different plane on each projection, exactly what a drifting probe produces.
    ramped = clean.copy()
    ramped[0] += 0.03 * u - 0.02 * v + 1.5
    ramped[1] += -0.05 * u + 0.011 * v - 0.3

    recovered = remove_phase_ramp(ramped, border=6)

    # The background must come back to flat zero; float32 plane fit, so ~1e-5.
    mask = background_mask(clean.shape[1:], border=6)
    assert np.abs(recovered[:, mask]).max() < 1e-4
    # And the object must survive, up to the constant the fit also removes.
    for i in range(2):
        difference = recovered[i] - clean[i]
        np.testing.assert_allclose(difference, difference.mean(), atol=2e-4)


def test_remove_phase_ramp_accepts_an_roi_mask():
    frame = _blob()
    n_v, n_u = frame.shape
    v, u = np.mgrid[0:n_v, 0:n_u].astype(np.float32)
    ramped = (frame + 0.02 * u + 0.01 * v)[np.newaxis]

    # An ROI in a corner, away from the blob -- what the user drags in the GUI.
    roi = np.zeros((n_v, n_u), dtype=bool)
    roi[0:10, 0:10] = True

    recovered = remove_phase_ramp(ramped, mask=roi)
    assert np.abs(recovered[0][roi]).max() < 1e-4


def test_remove_phase_offset_zeroes_the_background():
    prj = _blob()[np.newaxis] + 3.25
    recovered = remove_phase_offset(prj, border=6)
    mask = background_mask(prj.shape[1:], border=6)
    assert abs(float(recovered[0][mask].mean())) < 1e-5


def test_crop_and_pad_roundtrip():
    prj = np.stack([_blob()])
    padded = pad(prj, pad_u=5, pad_v=3)
    assert padded.shape == (1, prj.shape[1] + 6, prj.shape[2] + 10)
    assert padded[0, 0, 0] == 0.0

    back = crop(padded, (3, 3 + prj.shape[1], 5, 5 + prj.shape[2]))
    np.testing.assert_allclose(back, prj)


def test_check_mass_positive_flags_inverted_data():
    prj = _blob()[np.newaxis]
    ok, total = check_mass_positive(prj)
    assert ok and total > 0

    ok, total = check_mass_positive(invert(prj))
    assert not ok and total < 0


def test_background_mask_rejects_an_oversized_border():
    with pytest.raises(ValueError, match="no interior"):
        background_mask((10, 10), border=5)
