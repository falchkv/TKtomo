"""`tktomo-track-maxwell` without a cluster: the pure pieces and the start
sequence with the cluster primitives scripted."""

from __future__ import annotations

import pytest

from tktomo.tracking.remote import launch
from tktomo.tracking.remote.launch import (
    JOB_NAME,
    LaunchError,
    Remote,
    parse_squeue,
    sbatch_script,
    ssh_config,
)


def test_sbatch_script_runs_the_server_from_the_env():
    text = sbatch_script(remote_dir="/home/u/tktomo", stack="/gpfs/my stack.h5",
                         port=5611, partition="maxcpu", time_limit="04:00:00",
                         cpus=8, mem="64G", constraint="EPYC")
    assert text.startswith("#!/bin/bash\n")
    assert f"#SBATCH --job-name={JOB_NAME}" in text
    assert "#SBATCH --partition=maxcpu" in text
    assert "#SBATCH --time=04:00:00" in text
    assert "#SBATCH --cpus-per-task=8" in text
    assert "#SBATCH --mem=64G" in text and "#SBATCH --constraint=EPYC" in text
    assert "#SBATCH --output=/home/u/tktomo/jobs/slurm-%j.out" in text
    last = text.strip().splitlines()[-1]
    assert last.startswith("exec /home/u/tktomo/env/bin/python -m "
                           "tktomo.tracking.remote.server")
    assert "'/gpfs/my stack.h5'" in last            # quoted, not split
    assert "--address tcp://127.0.0.1:5611" in last
    assert "hostname -f" in text                    # the node names itself in the log
    # optional bits stay out when not asked for
    bare = sbatch_script(remote_dir="/t", stack=None, port=5, partition="p",
                         time_limit="1:00:00", cpus=1)
    assert "--mem" not in bare and "--constraint" not in bare
    assert bare.strip().splitlines()[-1].endswith("--address tcp://127.0.0.1:5 -v")


def test_parse_squeue_states():
    s = parse_squeue("RUNNING max-wn154 None\n")
    assert s.running and s.node == "max-wn154" and s.reason == ""
    s = parse_squeue("PENDING (null) Priority")
    assert not s.running and not s.gone and s.node == "" and s.reason == "Priority"
    s = parse_squeue("PENDING  Resources")
    assert s.state == "PENDING" and not s.running
    assert parse_squeue("").gone and parse_squeue("  \n").gone


def test_ssh_config_takes_first_value():
    text = "user jdoe\nhostname max-display.desy.de\n" \
           "identityfile ~/.ssh/id_ed25519_maxwell\nidentityfile ~/.ssh/id_rsa\n"
    cfg = ssh_config(text)
    assert cfg["user"] == "jdoe"
    assert cfg["identityfile"] == "~/.ssh/id_ed25519_maxwell"


class _Scripted(Remote):
    """A Remote whose cluster and process primitives are canned."""

    def __init__(self, squeue_lines, fail_tunnel=False):
        super().__init__(host="maxwell", remote_dir="~/tktomo", port=5611,
                         quiet=True)
        self.squeue_lines = list(squeue_lines)
        self.fail_tunnel = fail_tunnel
        self.calls: list[str] = []
        self._cfg = {"user": "u", "identityfile": "~/.ssh/k"}

    def run(self, cmd, *, input=None, check=True, timeout=None):  # noqa: A002
        joined = " ".join(cmd)
        self.calls.append(joined)
        if cmd[0] == "rsync":
            return ""
        if cmd[0] == "ssh":
            command = cmd[-1]
            if "sbatch --parsable" in command:
                assert input and input.startswith("#!/bin/bash")
                return "424242\n"
            if command.startswith("squeue"):
                return self.squeue_lines.pop(0) if self.squeue_lines else ""
            if command.startswith("test -x"):
                return "yes\n"
            if command.startswith("mkdir"):
                return "/home/u/tktomo\n"
            if command.startswith("scancel"):
                return ""
            if command.startswith("tail"):
                return "log tail\n"
        raise AssertionError(f"unexpected command {joined}")

    def spawn(self, cmd, *, detach=False):
        self.calls.append("spawn " + " ".join(cmd))
        assert detach == (cmd[0] == "ssh")      # tunnels detach, the window does not

        class Proc:
            pid = 999
            returncode = None

            def poll(self_inner):
                return 1 if self.fail_tunnel and cmd[0] == "ssh" else None

            def wait(self_inner):
                return 0

            def terminate(self_inner):
                self.calls.append("terminate " + cmd[0])

        return Proc()


@pytest.fixture
def no_sleep(monkeypatch, tmp_path):
    monkeypatch.setattr(launch.time, "sleep", lambda s: None)
    monkeypatch.setattr(launch, "STATE_FILE", tmp_path / "state.json")
    # the port is free until a tunnel has been spawned
    state = {"spawned": False}
    orig = _Scripted.spawn

    def spawn(self, cmd, **kw):
        state["spawned"] = cmd[0] == "ssh" or state["spawned"]
        return orig(self, cmd, **kw)

    monkeypatch.setattr(launch, "port_free",
                        lambda port, host="127.0.0.1": not state["spawned"])
    monkeypatch.setattr(_Scripted, "spawn", spawn)


def _args(**over):
    ns = launch.build_parser().parse_args(["start", "/gpfs/s.h5"] + [
        a for k, v in over.items() for a in ([f"--{k}"] if v is True else [])])
    return ns


def test_start_runs_submit_wait_tunnel_app_cancel(no_sleep):
    remote = _Scripted(["PENDING (null) Priority", "RUNNING max-wn154 None"])
    rc = launch.cmd_start(remote, _args())
    assert rc == 0
    kinds = [c.split()[0] if not c.startswith("ssh") else c.split("maxwell ")[-1].split()[0]
             for c in remote.calls]
    # rsync, submit, two polls, tunnel, app, cancel: in that order
    assert kinds.index("rsync") < kinds.index("sbatch") if "sbatch" in kinds else True
    assert [c for c in remote.calls if "squeue" in c].__len__() == 2
    spawns = [c for c in remote.calls if c.startswith("spawn")]
    assert spawns[0].startswith("spawn ssh -N") and "-J maxwell" in spawns[0]
    assert "-L 5611:localhost:5611" in spawns[0] and "u@max-wn154" in spawns[0]
    assert "-i" in spawns[0]
    assert "track_model_app --connect tcp://127.0.0.1:5611" in spawns[1]
    assert remote.calls.index(spawns[0]) < remote.calls.index(spawns[1])
    assert any("scancel 424242" in c for c in remote.calls)
    assert remote.calls[-1].endswith("scancel 424242") or "terminate" in remote.calls[-2]


def test_start_cancels_the_job_when_the_tunnel_fails(no_sleep):
    remote = _Scripted(["RUNNING max-wn154 None"], fail_tunnel=True)
    with pytest.raises(LaunchError, match="tunnel"):
        launch.cmd_start(remote, _args())
    assert any("scancel 424242" in c for c in remote.calls)
    assert not any("track_model_app" in c for c in remote.calls)


def test_start_reports_a_job_that_died_in_the_queue(no_sleep):
    remote = _Scripted([""])
    with pytest.raises(LaunchError, match="left the queue"):
        launch.cmd_start(remote, _args())


def test_no_app_leaves_job_and_tunnel(no_sleep):
    remote = _Scripted(["RUNNING max-wn154 None"])
    assert launch.cmd_start(remote, _args(**{"no-app": True})) == 0
    assert not any("scancel" in c for c in remote.calls)
    assert not any("track_model_app" in c for c in remote.calls)
    assert not any(c.startswith("terminate") for c in remote.calls)


def test_keep_leaves_the_job(no_sleep):
    remote = _Scripted(["RUNNING max-wn154 None"])
    assert launch.cmd_start(remote, _args(keep=True)) == 0
    assert not any("scancel" in c for c in remote.calls)
    assert any("track_model_app" in c for c in remote.calls)


def test_main_maps_launch_errors_to_exit_codes(monkeypatch, capsys):
    def boom(remote, args):
        raise LaunchError("no way", code=3)
    monkeypatch.setattr(launch, "cmd_status", boom)
    assert launch.main(["status"]) == 3
    assert "no way" in capsys.readouterr().err
