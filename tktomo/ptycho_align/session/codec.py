"""Turning session payloads into frames and back.

The wire format follows :mod:`tktomo.messaging.bus`: a msgpack header frame describing
the structure, then one raw frame per numpy array, in order. Arrays travel as bare
buffers rather than inside the msgpack blob so a plane is one contiguous copy rather
than a re-encoding of every float.

Everything crossing the boundary is a plain dataclass of numbers, strings and arrays --
that was a deliberate constraint on ``types.py``, and it is what lets this module be
generic rather than sixteen hand-written encoders that fall out of step with the
dataclasses they mirror. Add a field to ``SessionSummary`` and it travels; add a *type*
and it must be registered here, which :func:`encode` enforces by refusing to guess.

Three things are not obvious and matter:

**Tuples survive.** msgpack has no tuple, so a naive round trip turns ``original_shape``
into a list and every ``== (0, 0, 0)`` comparison in the window starts failing. Tuples
are tagged and restored. ``coerce_load_kwargs`` in the dataset module exists for the
same reason on the way in.

**Arrays come back writable.** ``np.frombuffer`` is a read-only view onto the received
frame; a viewer that writes into the plane it was handed would raise where the local
session let it through. The copy is the price of the two implementations behaving
identically.

**Exceptions keep their type.** ``Busy``, ``NoEngine``, ``JobFailed`` and the builtins
the verbs raise (``KeyError`` for an unknown stack, ``ValueError`` for a bad axis) are
reconstructed as themselves. A remote session that turned every refusal into a generic
RuntimeError would pass its signature tests and fail every caller that catches ``Busy``.
"""

from __future__ import annotations

import dataclasses as dc
from typing import Any

import numpy as np

from tktomo.ptycho_align.core.com import ComResult
from tktomo.ptycho_align.core.dataset import Hdf5Entry
from tktomo.ptycho_align.core.engine import AlignConfig
from tktomo.ptycho_align.core.preprocess import PreprocessOptions
from tktomo.ptycho_align.core.telemetry import ResourceSample
from tktomo.ptycho_align.session.protocol import (
    Busy,
    Event,
    EventBatch,
    JobFailed,
    JobHandle,
    JobState,
    NoEngine,
    SessionError,
)
from tktomo.ptycho_align.session.types import (
    DatasetSummary,
    IterationSummary,
    PlaneRef,
    PreprocessReport,
    RunPreflight,
    SessionSummary,
    StackSpec,
)

__all__ = ["CodecError", "decode", "encode", "register_errors", "register_types"]

# Tags. Short because they repeat once per node, and prefixed so they cannot collide
# with a genuine dict key -- every real dict is wrapped, so no untagged mapping ever
# reaches the decoder.
_ARRAY = "~a"
_TUPLE = "~t"
_DICT = "~d"
_OBJECT = "~o"
_ERROR = "~e"

#: Payload types allowed on the wire. A type absent from here raises rather than being
#: silently flattened -- a summary that arrived with its ComResult turned into a dict
#: would fail much later and much more confusingly.
_TYPES: tuple[type, ...] = (
    AlignConfig,
    ComResult,
    DatasetSummary,
    Event,
    EventBatch,
    Hdf5Entry,
    IterationSummary,
    JobHandle,
    JobState,
    PlaneRef,
    PreprocessOptions,
    PreprocessReport,
    ResourceSample,
    RunPreflight,
    SessionSummary,
    StackSpec,
)
_BY_NAME = {cls.__name__: cls for cls in _TYPES}

#: Exception types that survive as themselves. The session ones, plus the builtins the
#: verbs genuinely raise. Anything else arrives as a SessionError naming the original.
_ERRORS: tuple[type[BaseException], ...] = (
    Busy,
    NoEngine,
    JobFailed,
    SessionError,
    FileNotFoundError,
    IndexError,
    KeyError,
    MemoryError,
    NotImplementedError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)
_ERRORS_BY_NAME = {cls.__name__: cls for cls in _ERRORS}


class CodecError(Exception):
    """A payload contained something with no defined representation on the wire."""


def register_types(*types: type) -> None:
    """Let further dataclasses cross the wire.

    For a sibling host (the tracking server) whose payloads this module has no business
    importing. Both ends must register the same types, which is why they live in one
    module the client and the host both import.
    """
    for cls in types:
        if not dc.is_dataclass(cls):
            raise TypeError(f"{cls.__name__} is not a dataclass")
        _BY_NAME[cls.__name__] = cls


def register_errors(*types: type[BaseException]) -> None:
    """Let further exception types arrive as themselves rather than as SessionError."""
    for cls in types:
        _ERRORS_BY_NAME[cls.__name__] = cls


# -- encoding -------------------------------------------------------------------------


def encode(obj: Any) -> list[bytes]:
    """Render ``obj`` as ``[header, *array_buffers]``."""
    import msgpack  # noqa: PLC0415

    buffers: list[bytes] = []
    tree = _pack(obj, buffers)
    return [msgpack.packb(tree, use_bin_type=True), *buffers]


def _pack(obj: Any, buffers: list[bytes]) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str, bytes)):
        return obj

    # Before the ndarray branch: a numpy scalar is not an ndarray but is also not a
    # Python float, and msgpack cannot pack one. They turn up in metadata and in
    # anything derived from an array reduction.
    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, np.ndarray):
        array = np.ascontiguousarray(obj)
        buffers.append(array.tobytes())
        return {_ARRAY: len(buffers) - 1, "d": str(array.dtype), "s": list(array.shape)}

    if isinstance(obj, tuple):
        return {_TUPLE: [_pack(item, buffers) for item in obj]}

    if isinstance(obj, list):
        return [_pack(item, buffers) for item in obj]

    if isinstance(obj, dict):
        # As pairs, not as a mapping: metadata is free-form and its keys are not
        # guaranteed to be strings.
        return {_DICT: [[_pack(k, buffers), _pack(v, buffers)] for k, v in obj.items()]}

    if isinstance(obj, BaseException):
        return {_ERROR: _pack_error(obj, buffers)}

    if dc.is_dataclass(obj) and not isinstance(obj, type):
        name = type(obj).__name__
        if name not in _BY_NAME:
            raise CodecError(
                f"{name} is not registered in codec._TYPES, so it cannot cross the "
                "session boundary. Register it, or keep it out of the payload."
            )
        fields = {f.name: _pack(getattr(obj, f.name), buffers) for f in dc.fields(obj)}
        return {_OBJECT: name, "f": fields}

    raise CodecError(f"No wire representation for {type(obj).__name__}")


def _pack_error(exc: BaseException, buffers: list[bytes]) -> dict:
    # args[0] rather than str(exc): KeyError's str() adds quotes, and re-raising from
    # that would accumulate a fresh pair on every hop.
    message = str(exc.args[0]) if exc.args else str(exc)
    packed = {"t": type(exc).__name__, "m": message}
    if isinstance(exc, JobFailed):
        packed["x"] = exc.exc_type
        packed["b"] = exc.traceback
    return packed


# -- decoding -------------------------------------------------------------------------


def decode(frames: list[bytes]) -> Any:
    """Rebuild what :func:`encode` was given."""
    import msgpack  # noqa: PLC0415

    tree = msgpack.unpackb(frames[0], raw=False, strict_map_key=False)
    return _unpack(tree, list(frames[1:]))


def _unpack(node: Any, buffers: list[bytes]) -> Any:
    if isinstance(node, list):
        return [_unpack(item, buffers) for item in node]

    if not isinstance(node, dict):
        return node

    if _ARRAY in node:
        raw = buffers[node[_ARRAY]]
        # A copy, not the frame's memory: frombuffer is read-only, and the local
        # session hands out arrays a caller may write into.
        return np.frombuffer(raw, dtype=node["d"]).reshape(node["s"]).copy()

    if _TUPLE in node:
        return tuple(_unpack(item, buffers) for item in node[_TUPLE])

    if _DICT in node:
        return {_unpack(k, buffers): _unpack(v, buffers) for k, v in node[_DICT]}

    if _ERROR in node:
        return _unpack_error(node[_ERROR])

    if _OBJECT in node:
        name = node[_OBJECT]
        cls = _BY_NAME.get(name)
        if cls is None:
            raise CodecError(f"Received a {name}, which this build does not know")
        return cls(**{k: _unpack(v, buffers) for k, v in node["f"].items()})

    raise CodecError(f"Unrecognised node {sorted(node)}")


def _unpack_error(packed: dict) -> BaseException:
    name, message = packed["t"], packed["m"]
    cls = _ERRORS_BY_NAME.get(name)
    if cls is JobFailed:
        return JobFailed(message, exc_type=packed.get("x", "Exception"), traceback=packed.get("b", ""))
    if cls is None:
        # Named, not swallowed: the caller can still see what actually went wrong even
        # though it cannot catch the original type.
        return SessionError(f"{name}: {message}")
    return cls(message)
