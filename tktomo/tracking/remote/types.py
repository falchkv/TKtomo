"""What crosses the wire between the track-model window and a stack host.

Registered with the session codec at import, so both ends must import this
module (the client and the host both do). Adding a payload type means adding
it here; the codec refuses to guess.

`FramePacket` is the exception to "plain dataclass of numbers and arrays": it
carries a frame already packed down to display precision, because the link
this exists for is slow. See its docstring.

Fields added to a wire dataclass get a default and go at the end. The codec
rebuilds a payload with `cls(**fields)`, so an old sender is understood by a
new receiver, while a new sender to an old server fails loudly on the unknown
field. Both ends ship from the same checkout (`tktomo-track-maxwell` syncs the
source it installs on the node), which is why that is acceptable.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

from tktomo.ptycho_align.core.dataset import Hdf5Entry
from tktomo.ptycho_align.session.codec import register_errors, register_types
from tktomo.ptycho_align.session.protocol import SessionError
from tktomo.tracking.autotrack import (
    AutoLabel,
    AutoTrackJob,
    AutoTrackParams,
    TrackResult,
)
from tktomo.tracking.recon import SliceRequest
from tktomo.tracking.stacksource import AlignedExportRequest, StackInfo

__all__ = ["FramePacket", "NoStack", "WIRE_TYPES", "pack_frame"]

#: zlib level for packed frames. 1, not 6: the point is to spend a few
#: milliseconds of node CPU to save a second of wire, and the levels above 1
#: buy a few percent on this data for several times the time.
ZLIB_LEVEL = 1

#: Quantisation depth. 65536 levels across a frame's own range is roughly two
#: orders of magnitude finer than a display can show, and it halves the bytes.
Q_LEVELS = 65535


class NoStack(SessionError):
    """A pixel or compute verb was asked before a stack was opened."""


@dataclass(frozen=True)
class FramePacket:
    """One frame, packed for a slow link. Display precision, not analysis precision.

    Measured on a laptop tunnelled to a DESY compute node: the round trip is
    4 ms and the bandwidth 0.4 to 0.6 MB/s, so a 1.3 MB float32 frame takes
    about two seconds and the transfer is the whole cost of changing view.

    **Quantised to 16 bits over the frame's own min/max, then zlib'd**, which
    is a little over 2x end to end. Note what this is not: float16 would be
    the same size and much worse, because its resolution is relative to the
    magnitude, so a frame whose values sit in a narrow band on a large offset
    (0.5 rad of phase around 100) would collapse to a handful of levels and
    band visibly. Quantising over the range spends all 65536 levels on the
    range that is actually there, whatever the offset.

    The error is bounded by half a level, `scale / 2`, i.e. 1/131070 of the
    frame's range. That is display precision by a wide margin, and frames are
    only ever displayed: auto-track, the gridrec slice and the aligned export
    all run on the host, on the real pixels. A frame holding NaN or inf, or a
    client that asked for exactness, travels verbatim as float32 instead.
    """

    data: np.ndarray            # the payload bytes, as uint8
    shape: tuple[int, int]
    dtype: str                  # "uint16" (quantised) or "float32" (verbatim)
    compression: str            # "zlib" or "none"
    offset: float = 0.0
    scale: float = 1.0

    @property
    def quantised(self) -> bool:
        return self.dtype == "uint16"

    def unpack(self) -> np.ndarray:
        """The frame back as writable float32, the way `view()` promises it."""
        raw = self.data.tobytes()
        if self.compression == "zlib":
            raw = zlib.decompress(raw)
        flat = np.frombuffer(raw, self.dtype)
        if self.quantised:
            out = flat.astype(np.float32) * np.float32(self.scale)
            out += np.float32(self.offset)
        else:
            out = np.array(flat, np.float32)
        return out.reshape(self.shape)


def pack_frame(frame: np.ndarray, *, quantise: bool = True,
               compress: bool = True) -> FramePacket:
    """Pack one frame for the wire. The host side of `FramePacket`."""
    frame = np.ascontiguousarray(frame, np.float32)
    shape = tuple(int(x) for x in frame.shape)
    dtype, offset, scale = "float32", 0.0, 1.0
    payload = frame
    if quantise and frame.size and bool(np.isfinite(frame).all()):
        lo = float(frame.min())
        span = float(frame.max()) - lo
        # A constant frame has no range to spread over: every code is 0 and
        # the offset carries the value.
        scale = span / Q_LEVELS if span > 0 else 1.0
        offset = lo
        codes = (frame - np.float32(lo)) / np.float32(scale) if span > 0 \
            else np.zeros_like(frame)
        payload = np.clip(np.rint(codes), 0, Q_LEVELS).astype(np.uint16)
        dtype = "uint16"
    raw = payload.tobytes()
    compression = "none"
    if compress:
        blob = zlib.compress(raw, ZLIB_LEVEL)
        # Incompressible data exists; sending it larger would be silly.
        if len(blob) < len(raw):
            raw, compression = blob, "zlib"
    return FramePacket(data=np.frombuffer(raw, np.uint8), shape=shape,
                       dtype=dtype, compression=compression,
                       offset=offset, scale=scale)


WIRE_TYPES: tuple[type, ...] = (
    AlignedExportRequest,
    FramePacket,
    AutoLabel,
    AutoTrackJob,
    AutoTrackParams,
    Hdf5Entry,
    SliceRequest,
    StackInfo,
    TrackResult,
)

register_types(*WIRE_TYPES)
register_errors(NoStack)
