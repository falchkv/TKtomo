"""The boundary between the ptycho-align GUI and whatever runs the engine.

The GUI talks to an ``AlignmentSession`` and nothing else. Two implementations satisfy
it: one driving an :class:`~tktomo.ptycho_align.core.engine.AlignmentEngine` in this
process, one driving a server on a cluster node. Whether the compute is here or 400 km
away is a connection detail.

Qt-free by design, and it has to stay that way: the core is importable and testable
without the ``ui`` extra installed, and this layer sits underneath the window. Sessions
emit plain event objects; the adapter that turns them into Qt signals lives in ``ui/``.
"""

from __future__ import annotations

from tktomo.ptycho_align.session.engine_host import (
    STACK_ALIGNED,
    STACK_DIFFERENCE,
    STACK_KEYS,
    STACK_RAW,
    STACK_REPROJECTION,
    EngineHost,
)
from tktomo.ptycho_align.session.local import LocalSession
from tktomo.ptycho_align.session.protocol import (
    AlignmentSession,
    Busy,
    Event,
    EventBatch,
    EventLog,
    JobFailed,
    JobHandle,
    NoEngine,
    SessionError,
)
from tktomo.ptycho_align.session.types import (
    DatasetSummary,
    IterationSummary,
    PlaneRef,
    PreprocessReport,
    ResourceSample,
    RunPreflight,
    SessionSummary,
    StackSpec,
)

__all__ = [
    "STACK_ALIGNED",
    "STACK_DIFFERENCE",
    "STACK_KEYS",
    "STACK_RAW",
    "STACK_REPROJECTION",
    "AlignmentSession",
    "Busy",
    "DatasetSummary",
    "EngineHost",
    "Event",
    "EventBatch",
    "EventLog",
    "IterationSummary",
    "JobFailed",
    "JobHandle",
    "LocalSession",
    "NoEngine",
    "PlaneRef",
    "PreprocessReport",
    "ResourceSample",
    "RunPreflight",
    "SessionError",
    "SessionSummary",
    "StackSpec",
]
