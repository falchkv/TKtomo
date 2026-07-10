"""A thin ZeroMQ PUB/SUB bus for passing arrays + parameters between apps.

A :class:`Message` carries a ``topic``, a JSON-friendly ``params`` dict, and any
number of named numpy ``arrays``. Wire format is multipart:

    frame 0: topic (utf-8)
    frame 1: msgpack header {params, array specs [(name, dtype, shape), ...]}
    frame 2..: raw array buffers, in header order (zero-copy)

The serialization helpers are dependency-light (msgpack); the Publisher /
Subscriber import pyzmq lazily so importing the package never requires it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Default transport. Alignment tool publishes transforms / projection requests;
#: viewers subscribe.
DEFAULT_ADDRESS = "tcp://127.0.0.1:5599"
#: Reverse transport for replies. The sinogram viewer publishes requested
#: projections here; the alignment tool subscribes.
DEFAULT_REPLY_ADDRESS = "tcp://127.0.0.1:5600"

# Well-known topics.
TOPIC_ALIGNMENT = "alignment"
TOPIC_SINOGRAM = "sinogram"
#: Alignment tool -> sinogram viewer: "send me projection N".
TOPIC_PROJECTION_REQUEST = "projection_request"
#: Sinogram viewer -> alignment tool: the requested projection array.
TOPIC_PROJECTION_REPLY = "projection_reply"


@dataclass
class Message:
    topic: str
    params: dict[str, Any] = field(default_factory=dict)
    arrays: dict[str, np.ndarray] = field(default_factory=dict)

    def to_frames(self) -> list[bytes]:
        import msgpack  # noqa: PLC0415

        specs = []
        buffers = []
        for name, arr in self.arrays.items():
            arr = np.ascontiguousarray(arr)
            specs.append([name, str(arr.dtype), list(arr.shape)])
            buffers.append(arr.tobytes())
        header = msgpack.packb({"params": self.params, "specs": specs}, use_bin_type=True)
        return [self.topic.encode("utf-8"), header, *buffers]

    @classmethod
    def from_frames(cls, frames: list[bytes]) -> "Message":
        import msgpack  # noqa: PLC0415

        topic = frames[0].decode("utf-8")
        header = msgpack.unpackb(frames[1], raw=False)
        arrays: dict[str, np.ndarray] = {}
        for (name, dtype, shape), buf in zip(header["specs"], frames[2:]):
            arrays[name] = np.frombuffer(buf, dtype=dtype).reshape(shape).copy()
        return cls(topic=topic, params=header.get("params", {}), arrays=arrays)


class Publisher:
    """Publishes :class:`Message` objects on a PUB socket."""

    def __init__(self, address: str = DEFAULT_ADDRESS, *, bind: bool = True) -> None:
        import zmq  # noqa: PLC0415

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        if bind:
            self._sock.bind(address)
        else:
            self._sock.connect(address)

    def publish(self, message: Message) -> None:
        self._sock.send_multipart(message.to_frames())

    def close(self) -> None:
        self._sock.close(0)


class Subscriber:
    """Receives :class:`Message` objects from a SUB socket.

    Poll :meth:`poll` from a Qt timer to keep the UI responsive (non-blocking).
    """

    def __init__(
        self,
        address: str = DEFAULT_ADDRESS,
        *,
        topics: list[str] | None = None,
        bind: bool = False,
    ) -> None:
        import zmq  # noqa: PLC0415

        self._zmq = zmq
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        if bind:
            self._sock.bind(address)
        else:
            self._sock.connect(address)
        for topic in topics or [""]:
            self._sock.setsockopt_string(zmq.SUBSCRIBE, topic)

    def poll(self, timeout_ms: int = 0) -> Message | None:
        """Return a waiting message, or ``None`` if none within ``timeout_ms``."""
        if self._sock.poll(timeout_ms, self._zmq.POLLIN):
            frames = self._sock.recv_multipart()
            return Message.from_frames(frames)
        return None

    def close(self) -> None:
        self._sock.close(0)
