"""What is true of the remote session *because* it is remote.

The conformance suite already proves it behaves like the local one. These are the
things that only make sense over a wire: what the server refuses to serve, what
surviving a disconnect means, and that a reconnecting client is told what it missed.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from make_phantom import make_misaligned_dataset  # noqa: E402

from tktomo.io import save_projections  # noqa: E402
from tktomo.ptycho_align.session import (  # noqa: E402
    STACK_RAW,
    LocalSession,
    RemoteSession,
    SessionServer,
)
from tktomo.ptycho_align.session.codec import encode  # noqa: E402

TIMEOUT = 300.0


@pytest.fixture
def server():
    made = SessionServer(address="tcp://127.0.0.1:*").start()
    yield made
    made.stop()


@pytest.fixture
def session(server):
    made = RemoteSession(server.endpoint)
    yield made
    made.close()


@pytest.fixture
def phantom_file(tmp_path):
    data, *_ = make_misaligned_dataset(size=32, n_angles=16, max_shift=1.5, seed=3)
    path = tmp_path / "phantom.h5"
    save_projections(path, data)
    return path


# -- what is on offer -----------------------------------------------------------------


def test_the_server_will_not_serve_a_whole_stack(session, phantom_file):
    """The allowlist, not the client's manners, is what keeps 104 MiB off the wire.

    A client is just bytes on a socket; a server that dispatched by getattr would hand
    over the stack to anyone who guessed the method name.
    """
    session.wait(session.open_dataset(str(phantom_file)), timeout=TIMEOUT)

    for verb in ("read_stack", "read_volume"):
        with pytest.raises(KeyError, match="No such verb"):
            session._call(verb, STACK_RAW)


def test_an_unknown_verb_is_refused_rather_than_ignored(session):
    with pytest.raises(KeyError, match="No such verb"):
        session._call("rm_rf")


def test_a_malformed_request_does_not_take_the_server_down(session, server):
    """One bad frame must not cost everyone else their engine."""
    session._socket.send_multipart([b"not msgpack at all"])
    time.sleep(0.2)

    assert session.summary().has_engine is False, "the server is still answering"


def test_the_session_says_where_it_is(session, server):
    assert session.is_remote is True
    assert session.describe() == server.endpoint

    local = LocalSession()
    try:
        assert local.is_remote is False
        assert local.describe() == "this machine"
    finally:
        local.close()


# -- disconnecting --------------------------------------------------------------------


def test_closing_a_client_does_not_stop_the_engine(server, phantom_file):
    """A closed laptop lid must not throw away the iterations that already ran."""
    first = RemoteSession(server.endpoint)
    first.wait(first.open_dataset(str(phantom_file)), timeout=TIMEOUT)
    first.wait(first.run_com(), timeout=TIMEOUT)
    center = first.summary().center
    first.close()

    second = RemoteSession(server.endpoint)
    try:
        summary = second.summary()
        assert summary.has_engine is True
        assert summary.center == center
        assert second.read_plane(STACK_RAW, 0, 0) is not None
    finally:
        second.close()


def test_a_run_keeps_going_while_no_client_is_attached(server, phantom_file):
    pytest.importorskip("tomopy")
    first = RemoteSession(server.endpoint)
    first.wait(first.open_dataset(str(phantom_file)), timeout=TIMEOUT)
    handle = first.start_run(2)
    first.close()  # walk away mid-run

    second = RemoteSession(server.endpoint)
    try:
        assert second.wait(handle, timeout=TIMEOUT) == 2
        assert second.summary().iteration == 2
    finally:
        second.close()


def test_a_reconnecting_client_is_told_it_missed_events(server, phantom_file):
    """The ring is bounded, so "you are too far behind" has to be sayable."""
    session = RemoteSession(server.endpoint)
    try:
        session.wait(session.open_dataset(str(phantom_file)), timeout=TIMEOUT)
        batch = session.poll_events(0)
        assert batch.gap is False

        # Ask from before the ring's oldest entry, as a client returning from a long
        # sleep would.
        stale = session.poll_events(-1000)
        assert stale.gap is True
        assert stale.last_seq >= batch.last_seq
    finally:
        session.close()


# -- the wire itself ------------------------------------------------------------------


def test_a_late_reply_is_not_read_as_the_next_answer(session, phantom_file):
    """A verb that timed out leaves an answer in flight; ids are what keep it apart."""
    session.wait(session.open_dataset(str(phantom_file)), timeout=TIMEOUT)

    # Fire a request and abandon its reply, exactly as a timeout would.
    session._socket.send_multipart(encode({"id": -1, "verb": "summary", "args": [0]}))
    time.sleep(0.2)

    plane = session.read_plane(STACK_RAW, 0, 1)
    assert isinstance(plane, np.ndarray), "the orphaned summary was mistaken for a plane"


def test_planes_arrive_as_real_pixels_not_a_shape(session, phantom_file):
    """The obvious failure of a codec change: shapes right, contents zero."""
    session.wait(session.open_dataset(str(phantom_file)), timeout=TIMEOUT)

    local = LocalSession()
    try:
        local.wait(local.open_dataset(str(phantom_file)), timeout=TIMEOUT)
        np.testing.assert_allclose(
            session.read_plane(STACK_RAW, 0, 3), local.read_plane(STACK_RAW, 0, 3)
        )
    finally:
        local.close()
