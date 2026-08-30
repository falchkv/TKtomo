"""Serving a projection stack to a track-model window on another machine.

`TrackingHost` + `tktomo-track-server` run where the data is; the window
holds a `RemoteStackSource`. zmq and msgpack are imported lazily by the
session transport, so importing this package needs neither.
"""

from tktomo.tracking.remote.client import RemoteStackSource
from tktomo.tracking.remote.host import TrackingHost
from tktomo.tracking.remote.server import DEFAULT_ADDRESS, make_server, tracking_verbs
from tktomo.tracking.remote.types import NoStack

__all__ = [
    "DEFAULT_ADDRESS",
    "NoStack",
    "RemoteStackSource",
    "TrackingHost",
    "make_server",
    "tracking_verbs",
]
