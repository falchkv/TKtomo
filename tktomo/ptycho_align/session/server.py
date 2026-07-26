"""Serving an engine to a GUI that is somewhere else.

Runs on the machine with the data and the cores; the window runs on a laptop. The
server owns an :class:`~tktomo.ptycho_align.session.engine_host.EngineHost` and does
nothing to it that ``LocalSession`` does not -- all the concurrency, the ``Busy``
refusals and the job queue live in the host, so the two implementations cannot diverge
in the places that are hard to get right. This module is transport and nothing else.

**The loop must never block.** A ROUTER socket serving one request at a time is fine
only because every heavy verb returns a ``JobHandle`` immediately and the work happens
on the host's compute thread. Two things follow:

``wait`` is not served. Blocking the loop for the hour a run takes would stall every
other client, every progress poll and every plane read -- including the ones the user is
scrubbing through *while* the run goes. The client implements it by polling ``job_state``.

``subscribe`` is not served either. Events are pulled with ``poll_events``, which was
built in phase 1 with a sequence number and a bounded ring precisely so a client can
disconnect, come back, and ask what it missed.

**Closing a client does not stop the run.** Disconnecting is not cancelling: a user
whose laptop lid closes mid-run should be able to reconnect and find the iterations
still accumulating. Only ``stop()`` here shuts the engine down.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

from tktomo.ptycho_align.session.codec import decode, encode
from tktomo.ptycho_align.session.engine_host import EngineHost

logger = logging.getLogger("tktomo.ptycho_align")

__all__ = ["DEFAULT_ADDRESS", "SessionServer", "main", "serve"]

#: Distinct from the messaging bus's ports, which the other four apps use.
DEFAULT_ADDRESS = "tcp://127.0.0.1:5610"

# How long the loop waits for a request before checking whether it has been asked to
# stop. Short enough that shutdown feels immediate, long enough to be idle in between.
_POLL_MS = 100


def _verbs(host: EngineHost) -> dict[str, Callable[..., Any]]:
    """The allowlist. Explicit, not ``getattr`` on the host.

    ``getattr`` would expose ``read_stack`` and ``read_volume`` the moment someone
    guessed the name, and shipping a 511 MiB volume down a socket is exactly what this
    architecture exists to prevent. It would also expose the job queue internals.
    """
    return {
        # state
        "summary": host.summary,
        "poll_events": lambda since_seq, max_n=256: host.events.since(since_seq, max_n),
        "telemetry": host.telemetry,
        # cheap mutations
        "set_config": host.set_config,
        "set_center": host.set_center,
        "cancel_run": host.cancel_run,
        "cancel_job": host.cancel_job,
        # queries
        "list_hdf5": host.list_hdf5,
        "run_preflight": host.run_preflight,
        "cost_units": host.cost_units,
        "fetch_table": host.fetch_table,
        # pixels -- planes only; the whole-array readers are deliberately absent
        "read_planes": host.read_planes,
        "read_plane": host.read_plane,
        "read_volume_plane": host.read_volume_plane,
        # heavy: exclusive
        "open_dataset": host.open_dataset,
        "apply_preprocessing": host.apply_preprocessing,
        "reset_preprocessing": host.reset_preprocessing,
        "set_bin_factor": host.set_bin_factor,
        "run_com": host.run_com,
        "estimate_center": host.estimate_center,
        "start_run": host.start_run,
        "revert": host.revert,
        "open_session": host.open_session,
        # heavy: queued
        "materialize": host.materialize,
        "save_session": host.save_session,
        "export": host.export,
        # lifecycle
        "job_state": host.job_state,
        # Settled-ness and the current sequence number together, because the client
        # needs both to reproduce local `wait`'s guarantee and asking twice would leave
        # a window in which more events arrive between the two answers.
        "job_settled": lambda job_id: {
            "settled": host.job_settled(job_id),
            "seq": host.events.last_seq,
        },
    }


class SessionServer:
    """A ZeroMQ ROUTER in front of an :class:`EngineHost`."""

    def __init__(self, host: EngineHost | None = None, address: str = DEFAULT_ADDRESS) -> None:
        import zmq  # noqa: PLC0415

        self.host = host if host is not None else EngineHost()
        self._verbs = _verbs(self.host)
        self._zmq = zmq
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.ROUTER)
        self._socket.bind(address)
        # Resolved, because binding to ":*" picks the port and the caller needs to know
        # which -- that is how the tests get an address without fighting over one.
        self.endpoint = self._socket.getsockopt_string(zmq.LAST_ENDPOINT)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        logger.info("Session server listening on %s", self.endpoint)

    # -- lifecycle ---------------------------------------------------------------------

    def start(self) -> "SessionServer":
        """Serve on a background thread. Returns self so it can be chained."""
        self._thread = threading.Thread(target=self.serve_forever, name="session-server")
        self._thread.daemon = True
        self._thread.start()
        return self

    def serve_forever(self) -> None:
        while not self._stop.is_set():
            if not self._socket.poll(_POLL_MS, self._zmq.POLLIN):
                continue
            try:
                identity, *frames = self._socket.recv_multipart()
            except self._zmq.ZMQError:  # socket closed under us during shutdown
                break
            self._socket.send_multipart([identity, *self._handle(frames)])

    def stop(self) -> None:
        """Stop serving and shut the engine down."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self.host.close()
        self._socket.close(0)
        self._context.term()

    def __enter__(self) -> "SessionServer":
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()

    # -- dispatch ----------------------------------------------------------------------

    def _handle(self, frames: list[bytes]) -> list[bytes]:
        """One request in, one reply out. Never raises -- a failure is a reply too.

        Every reply echoes the request's ``id``. A client that gave up waiting must be
        able to recognise the late answer when it finally arrives and drop it, rather
        than read it as the answer to whatever it asked next.
        """
        request: dict = {}
        try:
            request = decode(frames)
            verb = self._verbs[request["verb"]]
        except KeyError as exc:
            return encode({"id": request.get("id"), "error": KeyError(f"No such verb: {exc}")})
        except Exception as exc:  # a malformed request must not kill the server
            logger.warning("Undecodable request: %s", exc)
            return encode({"id": None, "error": exc})

        request_id = request.get("id")
        try:
            result = verb(*request.get("args", []), **request.get("kwargs", {}))
        except Exception as exc:
            # Including Busy and NoEngine: a refusal is a normal outcome the client has
            # to see as itself, not a transport failure.
            return encode({"id": request_id, "error": exc})

        try:
            return encode({"id": request_id, "ok": result})
        except Exception as exc:
            logger.exception("Could not encode the result of %s", request["verb"])
            return encode({"id": request_id, "error": exc})


def serve(address: str = DEFAULT_ADDRESS) -> None:
    """Run a server in the foreground until interrupted."""
    server = SessionServer(address=address)
    print(f"ptycho-align engine serving on {server.endpoint}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.stop()


def main(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        prog="ptycho-align-server",
        description="Serve a ptycho-align engine to a remote GUI. Datasets, sessions and "
        "exports are all resolved on this machine, so run it where the data is.",
    )
    parser.add_argument(
        "--address",
        default="tcp://0.0.0.0:5610",
        help="what to bind (default: %(default)s; use tcp://127.0.0.1:5610 to stay local, "
        "or an SSH tunnel, since there is no authentication here)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log every iteration")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
    )
    serve(args.address)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
