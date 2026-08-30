"""`tktomo-track-server`: serve one projection stack to a remote track-model window.

The transport is the ptycho-align `SessionServer` with a different verb
table. The allowlist is explicit: nothing that returns a whole stack exists
on it, and the job queue internals are unreachable.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from tktomo.ptycho_align.session.jobs import job_verbs
from tktomo.ptycho_align.session.server import SessionServer
from tktomo.tracking.remote import types as _types  # noqa: F401 - registers the codec
from tktomo.tracking.remote.host import TrackingHost

logger = logging.getLogger("tktomo.tracking")

__all__ = ["DEFAULT_ADDRESS", "main", "serve", "tracking_verbs"]

#: Distinct from the ptycho-align server (5610) and the messaging bus (5599/5600).
DEFAULT_ADDRESS = "tcp://127.0.0.1:5611"


def tracking_verbs(host: TrackingHost) -> dict[str, Callable[..., Any]]:
    """The allowlist. Explicit, not ``getattr`` on the host."""
    return {
        # cheap
        "info": host.info,
        "detect_format": host.detect_format,
        "list_hdf5": host.list_hdf5,
        "autotrack_available": host.autotrack_available,
        "set_angles": host.set_angles,
        # pixels: frames only, never the stack
        "read_view": host.read_view,
        "read_views": host.read_views,
        "read_frames": host.read_frames,
        # jobs
        "open_stack": host.open_stack,
        "set_binning": host.set_binning,
        "gridrec_slice": host.gridrec_slice,
        "autotrack": host.autotrack,
        "export_aligned": host.export_aligned,
        # lifecycle: poll_events, job_state, job_settled, cancel_job
        **job_verbs(host),
    }


def make_server(address: str = DEFAULT_ADDRESS,
                host: TrackingHost | None = None) -> SessionServer:
    host = host if host is not None else TrackingHost()
    return SessionServer(host, address, verbs=tracking_verbs(host),
                         name="track-server")


def serve(address: str = DEFAULT_ADDRESS, path: str | None = None) -> None:
    """Run a server in the foreground until interrupted."""
    server = make_server(address)
    if path:
        print(f"opening {path} ...")
        info = server.host.wait(server.host.open_stack(path))
        print(f"  {info.shape[0]} views of {info.shape[1]}x{info.shape[2]} ({info.kind})")
    print(f"track-model stack server on {server.endpoint}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.stop()


def main(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        prog="tktomo-track-server",
        description="Serve a projection stack to a remote track-model window. Paths "
        "are resolved on this machine, so run it where the data is.",
    )
    parser.add_argument("stack", nargs="?", help="stack to open at startup (optional)")
    parser.add_argument(
        "--address",
        default=DEFAULT_ADDRESS,
        help="what to bind (default: %(default)s; keep it on localhost and reach it "
        "through an SSH tunnel, there is no authentication here)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
    )
    serve(args.address, args.stack)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
