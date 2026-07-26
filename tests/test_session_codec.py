"""The wire codec.

Unit-level, no sockets: these pin the representation itself, so that when a conformance
test fails against the remote session the cause is the session and not the encoding.

Runs in the light environment -- no Qt, no tomopy.
"""

from __future__ import annotations

import numpy as np
import pytest

from tktomo.ptycho_align.core.com import ComResult
from tktomo.ptycho_align.core.telemetry import ResourceSample
from tktomo.ptycho_align.session import (
    IterationSummary,
    PlaneRef,
    SessionSummary,
    StackSpec,
)
from tktomo.ptycho_align.session.codec import CodecError, decode, encode
from tktomo.ptycho_align.session.protocol import (
    Busy,
    Event,
    EventBatch,
    JobFailed,
    JobHandle,
    NoEngine,
    SessionError,
)


def roundtrip(obj):
    return decode(encode(obj))


# -- scalars and containers -----------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [None, True, False, 0, -17, 3.5, "text", b"bytes", [], [1, "two", None], {}],
)
def test_plain_values_survive(value):
    assert roundtrip(value) == value


def test_a_tuple_does_not_come_back_as_a_list():
    """msgpack has no tuple, and the window compares shapes against tuple literals."""
    assert roundtrip((0, 0, 0)) == (0, 0, 0)
    assert isinstance(roundtrip((1, 2)), tuple)
    assert isinstance(roundtrip([1, 2]), list)


def test_nesting_is_preserved():
    value = {"a": [(1, 2), {"b": (3,)}], "c": ((4, 5),)}
    assert roundtrip(value) == value


def test_a_dict_with_non_string_keys_survives():
    """Metadata is free-form; nothing guarantees its keys are strings."""
    assert roundtrip({1: "one", "two": 2}) == {1: "one", "two": 2}


def test_a_numpy_scalar_becomes_a_python_number():
    """msgpack cannot pack one, and they turn up wherever an array was reduced."""
    assert roundtrip(np.float32(1.5)) == 1.5
    assert roundtrip(np.int64(7)) == 7


# -- arrays ---------------------------------------------------------------------------


def test_an_array_round_trips_exactly():
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    back = roundtrip(array)
    np.testing.assert_array_equal(back, array)
    assert back.dtype == array.dtype
    assert back.shape == array.shape


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64", "uint8", "bool"])
def test_dtypes_are_not_silently_promoted(dtype):
    array = np.ones((2, 3), dtype=dtype)
    assert roundtrip(array).dtype == np.dtype(dtype)


def test_an_array_travels_outside_the_header():
    """The point of the multipart format: pixels are a buffer, not re-encoded floats."""
    array = np.zeros((64, 64), dtype=np.float32)
    frames = encode(array)

    assert len(frames) == 2
    assert len(frames[1]) == array.nbytes
    assert len(frames[0]) < 200, "the header should describe the array, not contain it"


def test_a_received_array_can_be_written_into():
    """frombuffer is read-only; the local session hands out arrays a caller may modify."""
    back = roundtrip(np.arange(4, dtype=np.float32))
    back[0] = 99.0  # must not raise
    assert back[0] == 99.0


def test_a_non_contiguous_array_survives():
    """A plane sliced out of a stack need not be contiguous before it is sent."""
    array = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]
    np.testing.assert_array_equal(roundtrip(array), array)


def test_an_empty_array_survives():
    """A summary before a dataset is open carries zero-length angle and shift arrays."""
    back = roundtrip(np.zeros(0, dtype=np.float64))
    assert back.shape == (0,)


# -- payload types --------------------------------------------------------------------


def test_a_stack_spec_round_trips():
    spec = StackSpec("Aligned", (8, 6, 5), available=False, pixel_epoch=3, stale=True)
    assert roundtrip(spec) == spec


def test_a_plane_ref_round_trips():
    assert roundtrip(PlaneRef("Raw", 1, 4)) == PlaneRef("Raw", 1, 4)


def test_an_iteration_summary_keeps_its_arrays_and_its_flags():
    result = IterationSummary(
        iteration=3,
        sx=np.array([1.0, 2.0]),
        sy=np.array([3.0, 4.0]),
        dsx=np.array([0.1, 0.2]),
        dsy=np.array([0.3, 0.4]),
        error=0.5,
        residual=0.25,
        center=32.5,
        wallclock_s=1.5,
        diverging=True,
        runaway="shifts exploded",
        has_volume=True,
    )
    back = roundtrip(result)

    np.testing.assert_array_equal(back.sx, result.sx)
    assert back.iteration == 3
    assert back.diverging is True
    assert back.runaway == "shifts exploded"
    assert back.has_volume is True


def test_a_session_summary_carries_its_nested_payloads():
    com = ComResult(
        sx=np.zeros(2),
        sy=np.zeros(2),
        center=10.0,
        com_u=np.ones(2),
        com_v=np.ones(2),
        fitted_u=np.ones(2),
        fit_residual=0.1,
        amplitude=2.0,
    )
    resources = ResourceSample(
        cpu_percent=12.0,
        rss_bytes=1,
        ram_available=2,
        ram_total=3,
        cpu_count=4,
        cgroup_limit=None,
        cgroup_current=None,
    )
    summary = SessionSummary(
        epoch=2,
        pixel_epoch=5,
        seq=9,
        has_engine=True,
        running=False,
        iteration=1,
        config={"recon_algorithm": "sirt", "pad": (0, 0)},
        bin_factor=2,
        center=32.0,
        angles=np.linspace(0, np.pi, 4),
        sx=np.zeros(4),
        sy=np.zeros(4),
        original_shape=(4, 6, 8),
        stacks=(StackSpec("Raw", (4, 6, 8)),),
        volume_iterations=(1, 2),
        com=com,
        resources=resources,
        metadata={"source": {"path": "/data/x.h5", "kwargs": {"crop": [0, 4, 0, 8]}}},
    )
    back = roundtrip(summary)

    assert back.original_shape == (4, 6, 8)
    assert isinstance(back.stacks, tuple) and back.stacks[0].key == "Raw"
    assert back.volume_iterations == (1, 2)
    assert isinstance(back.com, ComResult) and back.com.center == 10.0
    assert isinstance(back.resources, ResourceSample) and back.resources.cpu_count == 4
    assert back.config["pad"] == (0, 0)
    assert back.metadata["source"]["kwargs"]["crop"] == [0, 4, 0, 8]
    np.testing.assert_allclose(back.angles, summary.angles)


def test_an_event_batch_round_trips():
    batch = EventBatch(
        events=(Event(seq=1, kind="log", payload={"message": "hi"}, job_id="abc"),),
        last_seq=1,
        oldest_seq=1,
        gap=True,
    )
    back = roundtrip(batch)

    assert back.gap is True
    assert isinstance(back.events, tuple)
    assert back.events[0].payload["message"] == "hi"
    assert back.events[0].job_id == "abc"


def test_a_job_handle_round_trips():
    handle = JobHandle.new("start_run")
    assert roundtrip(handle) == handle


def test_an_unregistered_type_is_refused_rather_than_flattened():
    """Silently arriving as a dict would fail much later and much less clearly."""

    class Sneaky:
        pass

    with pytest.raises(CodecError):
        encode(Sneaky())


def test_an_unregistered_dataclass_names_itself_in_the_refusal():
    import dataclasses

    @dataclasses.dataclass
    class Local:
        x: int = 1

    with pytest.raises(CodecError, match="Local"):
        encode(Local())


# -- errors ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [Busy("a run is in flight"), NoEngine("no dataset"), SessionError("generic")],
)
def test_session_errors_keep_their_type(exc):
    back = roundtrip(exc)
    assert type(back) is type(exc)
    assert str(back) == str(exc)


@pytest.mark.parametrize("exc_type", [KeyError, ValueError, FileNotFoundError, MemoryError])
def test_the_builtins_the_verbs_raise_keep_their_type(exc_type):
    """`pytest.raises(KeyError)` has to mean the same thing against either session."""
    back = roundtrip(exc_type("message"))
    assert type(back) is exc_type


def test_a_key_error_does_not_accumulate_quotes_per_hop():
    """KeyError's str() adds quotes; re-raising from str() would nest them."""
    back = roundtrip(roundtrip(KeyError("Unknown stack 'X'")))
    assert back.args[0] == "Unknown stack 'X'"


def test_job_failed_keeps_the_traceback_the_dialog_shows():
    back = roundtrip(JobFailed("boom", exc_type="ValueError", traceback="Traceback..."))
    assert isinstance(back, JobFailed)
    assert back.exc_type == "ValueError"
    assert back.traceback == "Traceback..."


def test_an_unknown_error_type_is_named_rather_than_swallowed():
    class Exotic(Exception):
        pass

    back = roundtrip(Exotic("something specific"))
    assert isinstance(back, SessionError)
    assert "Exotic" in str(back)
    assert "something specific" in str(back)
