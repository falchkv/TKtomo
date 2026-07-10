import time

import numpy as np
import pytest

from tktomo.messaging import Message, Publisher, Subscriber
from tktomo.messaging.bus import TOPIC_PROJECTION_REPLY


def test_message_frame_roundtrip():
    msg = Message(
        topic="alignment",
        params={"dx": 1.0, "dy": -2.5, "rotation": 3.0},
        arrays={"sinogram": np.arange(12, dtype=np.float32).reshape(3, 4)},
    )
    frames = msg.to_frames()
    restored = Message.from_frames(frames)

    assert restored.topic == "alignment"
    assert restored.params == msg.params
    np.testing.assert_array_equal(restored.arrays["sinogram"], msg.arrays["sinogram"])
    assert restored.arrays["sinogram"].dtype == np.float32


def test_projection_reply_over_sockets():
    pytest.importorskip("zmq")
    address = "tcp://127.0.0.1:5623"  # test-specific port
    publisher = Publisher(address, bind=True)
    subscriber = Subscriber(address, topics=[TOPIC_PROJECTION_REPLY])
    projection = np.arange(20, dtype=np.float64).reshape(4, 5)
    try:
        # Republish until received to sidestep PUB/SUB slow-joiner in the test.
        received = None
        deadline = time.time() + 2.0
        while received is None and time.time() < deadline:
            publisher.publish(
                Message(
                    topic=TOPIC_PROJECTION_REPLY,
                    params={"n_angles": 90, "index": 3, "target": "fixed"},
                    arrays={"projection": projection},
                )
            )
            received = subscriber.poll(timeout_ms=50)
        assert received is not None
        assert received.params == {"n_angles": 90, "index": 3, "target": "fixed"}
        np.testing.assert_array_equal(received.arrays["projection"], projection)
    finally:
        publisher.close()
        subscriber.close()
