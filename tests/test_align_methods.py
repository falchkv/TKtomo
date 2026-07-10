import numpy as np

from tktomo.align import available_aligners, get_aligner
from tktomo.align.transform import Transform, apply_transform
from tktomo.io.phantom import generate_volume


def test_registry_lists_builtin_methods():
    names = available_aligners()
    assert "phase-correlation" in names
    assert any(n.startswith("stackreg") for n in names)


def test_phase_correlation_recovers_shift():
    fixed = generate_volume(size=128, n_slices=1)[0]
    true_shift = Transform(dx=5.0, dy=-3.0)
    moving = apply_transform(fixed, true_shift)

    aligner = get_aligner("phase-correlation")
    est = aligner.estimate(fixed, moving)

    # estimate maps moving -> fixed, i.e. it should recover the inverse shift.
    assert abs(est.dx + true_shift.dx) < 0.5
    assert abs(est.dy + true_shift.dy) < 0.5
