"""Resource readings, and the cgroup limits that override them.

The case that matters cannot be reproduced on a developer workstation: on a cluster node
the job runs inside a cgroup, and /proc/meminfo reports the *whole machine* -- often
hundreds of GB -- while the step is confined to what it asked for. Sizing a run off the
machine number there under-reports by an order of magnitude and the job gets OOM-killed
anyway, so these tests build the /sys layout by hand.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tktomo.ptycho_align.core import telemetry

GB = 1024**3


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def fake_cgroup(tmp_path, monkeypatch):
    """Point the module at a fixture tree instead of the real /proc and /sys."""

    def build(proc_text: str, files: dict[str, str], *, v1: bool = False) -> None:
        _write(tmp_path / "proc_cgroup", proc_text)
        root = tmp_path / ("sys/fs/cgroup/memory" if v1 else "sys/fs/cgroup")
        root.mkdir(parents=True, exist_ok=True)
        for relative, text in files.items():
            _write(root / relative, text)
        monkeypatch.setattr(telemetry, "_PROC_CGROUP", tmp_path / "proc_cgroup")
        monkeypatch.setattr(telemetry, "_CGROUP_V2_ROOT", tmp_path / "sys/fs/cgroup")
        monkeypatch.setattr(telemetry, "_CGROUP_V1_ROOT", tmp_path / "sys/fs/cgroup/memory")

    return build


def test_cgroup_v2_limit_on_an_ancestor_is_found(fake_cgroup):
    """SLURM puts the ceiling on the job scope, not the leaf the process sits in.

    The leaf says "max" -- unlimited -- but the job above it does not, and that is the
    number that actually kills the step.
    """
    fake_cgroup(
        "0::/system.slice/slurmstepd.scope/job_4242/step_0/user/task_0\n",
        {
            "system.slice/slurmstepd.scope/job_4242/memory.max": str(64 * GB),
            "system.slice/slurmstepd.scope/job_4242/memory.current": str(20 * GB),
            "system.slice/slurmstepd.scope/job_4242/step_0/user/task_0/memory.max": "max",
            "system.slice/slurmstepd.scope/job_4242/step_0/user/task_0/memory.current": str(
                20 * GB
            ),
        },
    )

    assert telemetry.cgroup_memory() == (64 * GB, 20 * GB)


def test_cgroup_v2_reports_no_limit_when_everything_is_max(fake_cgroup):
    fake_cgroup(
        "0::/user.slice\n",
        {"user.slice/memory.max": "max", "user.slice/memory.current": str(3 * GB)},
    )
    assert telemetry.cgroup_memory() is None


def test_cgroup_v1_limit_is_read(fake_cgroup):
    fake_cgroup(
        "9:memory:/slurm/uid_1000/job_77/step_0\n",
        {
            "slurm/uid_1000/job_77/memory.limit_in_bytes": str(32 * GB),
            "slurm/uid_1000/job_77/memory.usage_in_bytes": str(8 * GB),
        },
        v1=True,
    )
    assert telemetry.cgroup_memory() == (32 * GB, 8 * GB)


def test_cgroup_v1_sentinel_is_not_mistaken_for_a_limit(fake_cgroup):
    """v1 writes PAGE_COUNTER_MAX for "unlimited"; treating it as a ceiling would
    claim ~8 exabytes of headroom."""
    fake_cgroup(
        "9:memory:/user\n",
        {
            "user/memory.limit_in_bytes": str(telemetry._V1_UNLIMITED),
            "user/memory.usage_in_bytes": str(GB),
        },
        v1=True,
    )
    assert telemetry.cgroup_memory() is None


def test_the_tightest_cgroup_in_the_chain_wins(fake_cgroup):
    fake_cgroup(
        "0::/outer/inner\n",
        {
            "outer/memory.max": str(100 * GB),
            "outer/memory.current": str(10 * GB),
            "outer/inner/memory.max": str(16 * GB),
            "outer/inner/memory.current": str(4 * GB),
        },
    )
    assert telemetry.cgroup_memory() == (16 * GB, 4 * GB)


def test_available_ram_prefers_the_cgroup_headroom_over_the_machine(monkeypatch):
    """The whole point: a 512 GB node does not mean a 512 GB job."""
    monkeypatch.setattr(telemetry, "read_meminfo", lambda: (512 * GB, 400 * GB))
    monkeypatch.setattr(telemetry, "cgroup_memory", lambda: (64 * GB, 60 * GB))

    assert telemetry.available_ram_bytes() == 4 * GB


def test_available_ram_falls_back_to_the_machine_without_a_cgroup(monkeypatch):
    monkeypatch.setattr(telemetry, "read_meminfo", lambda: (16 * GB, 5 * GB))
    monkeypatch.setattr(telemetry, "cgroup_memory", lambda: None)

    assert telemetry.available_ram_bytes() == 5 * GB


def test_available_ram_uses_the_machine_when_it_is_the_tighter_of_the_two(monkeypatch):
    """A generous cgroup limit does not conjure memory the machine has already spent."""
    monkeypatch.setattr(telemetry, "read_meminfo", lambda: (16 * GB, 2 * GB))
    monkeypatch.setattr(telemetry, "cgroup_memory", lambda: (64 * GB, 0))

    assert telemetry.available_ram_bytes() == 2 * GB


def test_headroom_on_a_sample_honours_the_cgroup():
    sample = telemetry.ResourceSample(
        cpu_percent=12.0,
        rss_bytes=GB,
        ram_available=400 * GB,
        ram_total=512 * GB,
        cpu_count=8,
        cgroup_limit=64 * GB,
        cgroup_current=60 * GB,
    )
    assert sample.headroom_bytes == 4 * GB


def test_headroom_without_a_cgroup_is_just_the_machine():
    sample = telemetry.ResourceSample(
        cpu_percent=0.0, rss_bytes=GB, ram_available=5 * GB, ram_total=16 * GB, cpu_count=4
    )
    assert sample.headroom_bytes == 5 * GB


def test_a_monitor_produces_a_sample_on_this_machine():
    monitor = telemetry.ResourceMonitor()
    if not monitor.supported:
        pytest.skip("needs /proc")

    reading = monitor.sample()
    assert reading is not None
    assert reading.ram_total > 0
    assert reading.cpu_count >= 1
    assert 0.0 <= reading.cpu_percent <= 100.0
