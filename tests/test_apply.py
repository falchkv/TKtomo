import numpy as np
import pytest

from tktomo.align import Transform, apply_projection_transform
from tktomo.io.phantom import generate_phantom
from tktomo.messaging.bus import Message, TOPIC_ALIGNMENT


def test_apply_only_touches_target_projection():
    data = generate_phantom(n_angles=10, size=64, n_slices=4).data
    out = apply_projection_transform(data, 3, Transform(dx=4.0, dy=0.0))

    # every other projection is untouched
    for i in range(data.shape[0]):
        if i != 3:
            np.testing.assert_array_equal(out[i], data[i])
    # the target projection changed
    assert not np.array_equal(out[3], data[3])


def test_apply_shift_matches_expected_pixels():
    data = np.zeros((3, 21, 21))
    data[1, 10, 10] = 1.0
    out = apply_projection_transform(data, 1, Transform(dx=3, dy=0), order=0)
    assert out[1, 10, 13] == 1.0


def test_apply_out_of_range_raises():
    data = np.zeros((3, 8, 8))
    with pytest.raises(IndexError):
        apply_projection_transform(data, 5, Transform(dx=1))


def test_alignment_message_carries_index():
    params = Transform(dx=2.0, dy=-1.0, rotation=0.5).as_dict()
    params["projection_index"] = 7
    restored = Message.from_frames(
        Message(topic=TOPIC_ALIGNMENT, params=params).to_frames()
    )
    assert restored.params["projection_index"] == 7
    assert Transform.from_dict(restored.params) == Transform(dx=2.0, dy=-1.0, rotation=0.5)
