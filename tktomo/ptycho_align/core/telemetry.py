"""RAM and CPU readings, and the cgroup limits that actually bind.

A run is long -- iterations of tens of minutes are normal -- and its two failure modes
are invisible from the alignment views: the process creeping towards the memory ceiling
until the kernel kills it, which looks like the window simply vanishing with no
traceback, and the reconstruction quietly running on one core when it was meant to use
eight.

The reason this is not just ``/proc/meminfo``: on a cluster node the job runs inside a
cgroup, and ``MemAvailable`` reports the **whole machine** -- often hundreds of GB --
while the job is confined to what it asked for. Sizing a run off that number
under-reports the pressure by an order of magnitude, and the step gets OOM-killed
anyway. The same goes for ``os.cpu_count()`` versus the CPUs actually allocated. So
every reading here takes whichever of machine and cgroup is tighter.

Deliberately dependency-free: psutil is not a dependency of this package, and reading a
few files under /proc and /sys is not worth making it one. Where the files do not exist
the functions return ``None`` rather than guessing.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ResourceMonitor",
    "ResourceSample",
    "allocated_cpu_count",
    "available_ram_bytes",
    "cgroup_memory",
    "read_cpu_busy_idle",
    "read_meminfo",
    "read_process_rss",
]

# cgroup v1 writes this when a controller is unlimited; it is PAGE_COUNTER_MAX scaled by
# the page size, not a real ceiling, and must not be mistaken for one.
_V1_UNLIMITED = 0x7FFFFFFFFFFFF000

# Module-level so a test can point them at a fixture tree: the layout that matters here
# is a cluster node's, which is exactly the one the developer's machine does not have.
_PROC_CGROUP = Path("/proc/self/cgroup")
_CGROUP_V2_ROOT = Path("/sys/fs/cgroup")
_CGROUP_V1_ROOT = Path("/sys/fs/cgroup/memory")


@dataclass(frozen=True)
class ResourceSample:
    """One reading of the machine the compute is actually running on."""

    cpu_percent: float
    rss_bytes: int
    ram_available: int
    ram_total: int
    cpu_count: int
    cgroup_limit: int | None = None
    cgroup_current: int | None = None

    @property
    def headroom_bytes(self) -> int:
        """What we could still allocate, honouring the cgroup if one binds."""
        if self.cgroup_limit is None or self.cgroup_current is None:
            return self.ram_available
        return min(self.ram_available, max(0, self.cgroup_limit - self.cgroup_current))


def read_meminfo() -> tuple[int, int] | None:
    """``(total, available)`` system RAM in bytes, or None without /proc.

    ``MemAvailable`` rather than MemFree: it accounts for reclaimable page cache, so it
    is the honest answer to "how much could I allocate".
    """
    total = available = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                total = int(line.split()[1]) * 1024
            elif line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
            if total is not None and available is not None:
                return total, available
    except (OSError, ValueError, IndexError):
        return None
    return None


def read_process_rss() -> int | None:
    """This process's resident set size in bytes."""
    try:
        fields = Path("/proc/self/statm").read_text().split()
        return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError, AttributeError):
        return None


def read_cpu_busy_idle() -> tuple[float, float] | None:
    """Cumulative ``(busy, idle)`` jiffies across all CPUs, from /proc/stat.

    System-wide rather than per-process on purpose: tomopy fans work out to worker
    *processes* when ``ncore > 1``, whose CPU time never appears in this process's own
    counters. Watching the machine is what actually answers "is the reconstruction using
    the cores I gave it".
    """
    try:
        first = Path("/proc/stat").read_text().split("\n", 1)[0].split()
        values = [float(v) for v in first[1:]]
    except (OSError, ValueError, IndexError):
        return None
    if len(values) < 4:
        return None
    # user nice system idle iowait irq softirq ...
    idle = values[3] + (values[4] if len(values) > 4 else 0.0)
    return sum(values) - idle, idle


def allocated_cpu_count() -> int:
    """CPUs this process may actually run on.

    ``os.cpu_count()`` reports the whole node, which on a shared cluster machine can be
    an order of magnitude more than the allocation. The affinity mask is what the
    scheduler actually granted.
    """
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _cgroup_dirs() -> list[Path]:
    """Directories holding this process's memory cgroup, leaf first.

    A limit set on an ancestor binds just as hard as one on the leaf -- under SLURM the
    ceiling usually sits on the job scope rather than the task -- so the caller has to
    look at the whole chain, not just where the process happens to sit.
    """
    try:
        lines = _PROC_CGROUP.read_text().splitlines()
    except OSError:
        return []

    dirs: list[Path] = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        controllers, relative = parts[1], parts[2].lstrip("/")
        if controllers == "":  # v2: "0::/path"
            root = _CGROUP_V2_ROOT
        elif "memory" in controllers.split(","):  # v1: "N:memory,foo:/path"
            root = _CGROUP_V1_ROOT
        else:
            continue
        if not root.is_dir():
            continue

        current = root / relative if relative else root
        while True:
            dirs.append(current)
            if current == root or root not in current.parents:
                break
            current = current.parent
    return dirs


def _read_int(path: Path) -> int | None:
    try:
        text = path.read_text().strip()
    except (OSError, ValueError):
        return None
    if text == "max":  # v2's way of saying unlimited
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    return None if value >= _V1_UNLIMITED else value


def cgroup_memory() -> tuple[int, int] | None:
    """``(limit, current)`` bytes for the tightest cgroup binding this process.

    Returns None when there is no limit -- an ordinary workstation, or a cluster that
    does not confine memory. Both v2 (``memory.max``) and v1
    (``memory.limit_in_bytes``) are handled.
    """
    tightest: tuple[int, int] | None = None
    for directory in _cgroup_dirs():
        for limit_name, current_name in (
            ("memory.max", "memory.current"),
            ("memory.limit_in_bytes", "memory.usage_in_bytes"),
        ):
            limit = _read_int(directory / limit_name)
            if limit is None:
                continue
            current = _read_int(directory / current_name) or 0
            if tightest is None or limit < tightest[0]:
                tightest = (limit, current)
            break
    return tightest


def available_ram_bytes() -> int | None:
    """Memory we could actually allocate right now, or None if we cannot tell.

    Takes the tighter of what the machine reports and what the cgroup allows. See the
    module docstring for why the cgroup half is not optional on a cluster.
    """
    memory = read_meminfo()
    available = memory[1] if memory is not None else None

    if available is None:
        try:
            available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (AttributeError, ValueError, OSError):
            available = None  # not POSIX, or the sysconf names are missing

    limits = cgroup_memory()
    if limits is not None:
        headroom = max(0, limits[0] - limits[1])
        return headroom if available is None else min(available, headroom)
    return available


class ResourceMonitor:
    """Turns the raw readings into :class:`ResourceSample`, tracking CPU as a rate.

    CPU load only means anything as a difference between two readings, so the monitor
    has to remember the previous one; the first sample can only establish the baseline.
    """

    def __init__(self) -> None:
        self._previous_cpu: tuple[float, float] | None = None
        self._started = time.monotonic()

    @property
    def supported(self) -> bool:
        return read_meminfo() is not None and read_cpu_busy_idle() is not None

    def sample(self) -> ResourceSample | None:
        """Take one reading, or None if /proc is unavailable."""
        memory = read_meminfo()
        rss = read_process_rss()
        cpu_now = read_cpu_busy_idle()
        if memory is None or rss is None or cpu_now is None:
            return None

        percent = 0.0
        if self._previous_cpu is not None:
            busy = cpu_now[0] - self._previous_cpu[0]
            idle = cpu_now[1] - self._previous_cpu[1]
            if busy + idle > 0:
                percent = 100.0 * busy / (busy + idle)
        self._previous_cpu = cpu_now

        limits = cgroup_memory()
        total, available = memory
        return ResourceSample(
            cpu_percent=percent,
            rss_bytes=rss,
            ram_available=available,
            ram_total=total,
            cpu_count=allocated_cpu_count(),
            cgroup_limit=None if limits is None else limits[0],
            cgroup_current=None if limits is None else limits[1],
        )
