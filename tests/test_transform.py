import numpy as np

from tktomo.align.transform import Transform, TransformHistory, apply_transform


def test_transform_dict_roundtrip():
    t = Transform(dx=1.5, dy=-2.0, rotation=3.0)
    assert Transform.from_dict(t.as_dict()) == t


def test_identity_apply_is_noop():
    img = np.random.default_rng(0).random((32, 32))
    out = apply_transform(img, Transform())
    np.testing.assert_array_equal(img, out)


def test_apply_translation_shifts_image():
    img = np.zeros((21, 21))
    img[10, 10] = 1.0
    out = apply_transform(img, Transform(dx=3, dy=0), order=0)
    # dx shifts along columns (last axis).
    assert out[10, 13] == 1.0


def test_history_undo():
    h = TransformHistory()
    assert not h.can_undo()
    h.push(Transform(dx=1))
    h.push(Transform(dx=2))
    assert h.current.dx == 2
    assert h.undo().dx == 1
    assert h.undo().dx == 0  # back to initial identity
    assert not h.can_undo()
