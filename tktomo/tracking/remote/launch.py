"""`tktomo-track-maxwell`: run the stack server on a Maxwell node from your laptop.

The remote track-model workflow needs four things that are each easy and
together tedious: a Python environment on the cluster, a SLURM job that runs
`tktomo-track-server` on a compute node, an SSH tunnel to that node once it
is known, and the window on the laptop pointed at the tunnel. This script
does them with your own ssh alias and nothing else::

    tktomo-track-maxwell setup                    # once: sync source, build env
    tktomo-track-maxwell start /path/on/gpfs.h5   # submit, wait, tunnel, app, clean up
    tktomo-track-maxwell status
    tktomo-track-maxwell stop

Everything on the cluster lives under `--remote-dir` (default
`/gpfs/petra3/scratch/$USER/tktomo`, Maxwell's per-user scratch; home
quotas are too small for a conda env):
`src/` (an rsync of this repo), `env/` (a conda-forge env with tomopy),
`jobs/` (sbatch scripts and logs). Compute nodes cannot ssh out, so the
laptop tunnels *in* through the login node; the server only ever binds the
node's loopback, so reaching it needs your ssh login and nothing is opened
on the cluster network.

Qt-free and dependency-free (stdlib only): the laptop needs `ssh` and
`rsync`, the app needs the `ui` extra, the cluster needs `mamba` or `conda`.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "JOB_NAME",
    "Remote",
    "main",
    "parse_squeue",
    "sbatch_script",
    "ssh_config",
]

JOB_NAME = "tktomo-track-server"
DEFAULT_HOST = "maxwell"
#: Maxwell scratch: no quota, cleaned after three months, mounted on the
#: compute nodes. Home has a small quota and a conda env does not fit it.
#: Everything here is rebuilt by `setup`, so losing it costs minutes.
DEFAULT_REMOTE_DIR = "/gpfs/petra3/scratch/$USER/tktomo"
DEFAULT_PORT = 5611
ENV_PYTHON = "python=3.12"
#: conda-forge packages for the node: what the server imports at start, what
#: opening a stack needs, tomopy for the gridrec slice, sklearn + joblib for
#: auto-complete. No Qt.
ENV_PACKAGES = (
    "numpy", "scipy", "h5py", "hdf5plugin", "scikit-image", "tifffile",
    "pyzmq", "msgpack-python", "scikit-learn", "joblib", "tomopy",
    "pip", "setuptools",     # so the editable install needs no PyPI access
)
SYNC_EXCLUDES = (".git", ".idea", "tests", "docs", "*.blend", "meshes_*",
                 "__pycache__", "*.egg-info", ".pytest_cache", "*.h5")
STATE_FILE = Path("~/.cache/tktomo/track-maxwell.json").expanduser()

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_UNREACHABLE = 0, 1, 2, 3


class LaunchError(Exception):
    """Something the user has to fix; the message is the whole story."""

    def __init__(self, message: str, code: int = EXIT_ERROR) -> None:
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------
# pure pieces (tested without a cluster)
# ---------------------------------------------------------------------------

def sbatch_script(*, remote_dir: str, stack: str | None, port: int,
                  partition: str, time_limit: str, cpus: int,
                  mem: str | None = None, constraint: str | None = None,
                  verbose: bool = True) -> str:
    """The batch script that runs the server on the node it lands on."""
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={JOB_NAME}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --time={time_limit}",
        "#SBATCH --nodes=1",
        f"#SBATCH --cpus-per-task={int(cpus)}",
        f"#SBATCH --output={remote_dir}/jobs/slurm-%j.out",
    ]
    if mem:
        lines.append(f"#SBATCH --mem={mem}")
    if constraint:
        lines.append(f"#SBATCH --constraint={constraint}")
    python = f"{remote_dir}/env/bin/python"
    args = [python, "-m", "tktomo.tracking.remote.server"]
    if stack:
        args.append(stack)
    args += ["--address", f"tcp://127.0.0.1:{int(port)}"]
    if verbose:
        args.append("-v")
    lines += [
        "set -u",
        'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"',
        'export NUMEXPR_MAX_THREADS="${SLURM_CPUS_PER_TASK:-1}"',
        f'echo "node $(hostname -f) port {int(port)}"',
        f'test -x {shlex.quote(python)} || {{ echo "no env at {remote_dir}/env: '
        'run tktomo-track-maxwell setup" >&2; exit 1; }',
        "exec " + " ".join(shlex.quote(a) for a in args),
        "",
    ]
    return "\n".join(lines)


@dataclass(frozen=True)
class JobState:
    state: str            # RUNNING, PENDING, ... or "" when the job left the queue
    node: str = ""
    reason: str = ""

    @property
    def gone(self) -> bool:
        return self.state == ""

    @property
    def running(self) -> bool:
        return self.state == "RUNNING" and bool(self.node)


def parse_squeue(text: str) -> JobState:
    """One line of `squeue -h -o "%T %N %r"` (empty when the job is gone)."""
    line = text.strip().splitlines()[0].strip() if text.strip() else ""
    if not line:
        return JobState("")
    parts = line.split(None, 2)
    state = parts[0]
    node = parts[1] if len(parts) > 1 and parts[1] not in ("(null)", "") else ""
    reason = parts[2] if len(parts) > 2 else ""
    if reason in ("None", "(null)"):
        reason = ""
    return JobState(state, node, reason)


def ssh_config(text: str) -> dict[str, str]:
    """`ssh -G host` output as a dict: lower-case keyword -> first value.

    What we need from it: `user` and `identityfile`, so the hop to the
    compute node offers the same identity as the hop to the login node.
    """
    out: dict[str, str] = {}
    for line in text.splitlines():
        key, _, value = line.strip().partition(" ")
        if key and key.lower() not in out:
            out[key.lower()] = value.strip()
    return out


def port_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket() as s:
        s.settimeout(0.2)
        return s.connect_ex((host, int(port))) != 0


# ---------------------------------------------------------------------------
# the cluster
# ---------------------------------------------------------------------------

@dataclass
class Remote:
    """One ssh alias and a directory under it. All cluster I/O goes through here."""

    host: str = DEFAULT_HOST
    remote_dir: str = DEFAULT_REMOTE_DIR
    port: int = DEFAULT_PORT
    quiet: bool = False
    _cfg: dict[str, str] | None = field(default=None, repr=False)

    # -- primitives (monkeypatched in tests) ------------------------------------------

    def run(self, cmd: list[str], *, input: str | None = None,   # noqa: A002
            check: bool = True, timeout: float | None = 120.0) -> str:
        """Run a local command, return stdout. Raises LaunchError on failure."""
        try:
            proc = subprocess.run(cmd, input=input, capture_output=True, text=True,
                                  timeout=timeout)
        except FileNotFoundError:
            raise LaunchError(f"{cmd[0]} is not installed on this machine") from None
        except subprocess.TimeoutExpired:
            raise LaunchError(f"timed out: {' '.join(cmd[:3])} ...") from None
        if check and proc.returncode != 0:
            err = (proc.stderr or proc.stdout).strip()
            code = EXIT_ERROR
            if cmd[0] == "ssh" and proc.returncode == 255:
                code = EXIT_UNREACHABLE
                err += f"\ncannot reach {self.host}: is the DESY VPN up?"
            raise LaunchError(err or f"{cmd[0]} failed ({proc.returncode})", code)
        return proc.stdout

    def ssh(self, command: str, *, input: str | None = None,   # noqa: A002
            check: bool = True, timeout: float | None = 120.0) -> str:
        return self.run(["ssh", "-o", "BatchMode=yes", self.host, command],
                        input=input, check=check, timeout=timeout)

    def spawn(self, cmd: list[str], *, detach: bool = False) -> subprocess.Popen:
        """Start a child. `detach` gives it no stdin/stdout and its own session,
        for a tunnel that must outlive this process (and a shell pipe that
        must not wait on it)."""
        if not detach:
            return subprocess.Popen(cmd)
        return subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                start_new_session=True)

    def say(self, message: str) -> None:
        if not self.quiet:
            print(message, flush=True)

    # -- ssh identity ------------------------------------------------------------------

    def config(self) -> dict[str, str]:
        if self._cfg is None:
            self._cfg = ssh_config(self.run(["ssh", "-G", self.host]))
        return self._cfg

    @property
    def user(self) -> str:
        return self.config().get("user") or os.environ.get("USER", "")

    def node_ssh_args(self, node: str) -> list[str]:
        """ssh arguments for the hop login node -> compute node."""
        args = ["-J", self.host, "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=accept-new"]
        identity = self.config().get("identityfile")
        if identity:
            args += ["-i", os.path.expanduser(identity), "-o", "IdentitiesOnly=yes"]
        return args + [f"{self.user}@{node}"]

    # -- source and env ----------------------------------------------------------------

    def resolve_dir(self) -> str:
        """Make `remote_dir` absolute: `~` and `$USER` are the cluster
        shell's to expand, and neither sbatch headers nor quoted paths do."""
        out = self.ssh(f"mkdir -p {self.remote_dir}/src {self.remote_dir}/jobs && "
                       f"cd {self.remote_dir} && pwd").strip()
        if not out.startswith("/"):
            raise LaunchError(f"could not resolve {self.remote_dir} on {self.host}")
        self.remote_dir = out
        return out

    def sync_source(self, repo: Path) -> None:
        if shutil.which("rsync") is None:
            raise LaunchError("rsync is not installed on this machine")
        cmd = ["rsync", "-az", "--delete"]
        for pattern in SYNC_EXCLUDES:
            cmd += ["--exclude", pattern]
        cmd += [f"{repo}/", f"{self.host}:{self.remote_dir}/src/"]
        self.say(f"syncing {repo} -> {self.host}:{self.remote_dir}/src")
        self.run(cmd, timeout=600.0)

    def env_exists(self) -> bool:
        out = self.ssh(f"test -x {self.remote_dir}/env/bin/python && echo yes",
                       check=False)
        return out.strip() == "yes"

    def build_env(self, rebuild: bool = False) -> None:
        if rebuild:
            self.ssh(f"rm -rf {self.remote_dir}/env")
        packages = " ".join((ENV_PYTHON,) + ENV_PACKAGES)
        if not rebuild and self.env_exists():
            self.say(f"env exists at {self.host}:{self.remote_dir}/env: "
                     "checking its packages (--rebuild recreates it)")
            verb = "install"
        else:
            self.say(f"creating {self.host}:{self.remote_dir}/env "
                     "(conda-forge, a few minutes)")
            verb = "create"
        self.ssh(
            "m=$(command -v mamba || command -v conda || "
            "ls /software/mamba/*/condabin/mamba 2>/dev/null | tail -1); "
            'test -n "$m" || { echo "no mamba/conda on PATH" >&2; exit 1; }; '
            f'"$m" {verb} -y -q -p {self.remote_dir}/env -c conda-forge '
            f"{packages}", timeout=3600.0)
        self.say("installing tktomo into the env")
        self.ssh(f"{self.remote_dir}/env/bin/pip install -q --no-deps "
                 f"--no-build-isolation -e {self.remote_dir}/src", timeout=600.0)
        check = self.ssh(
            f"{self.remote_dir}/env/bin/python -c "
            + shlex.quote(
                "import tktomo.tracking.remote.server, tomopy, zmq, msgpack, h5py;"
                "from tktomo.tracking import learned_match;"
                "ok, why = learned_match.available();"
                "print('server imports fine; auto-complete', "
                "'available' if ok else 'unavailable: ' + why)"))
        self.say(check.strip())

    # -- jobs --------------------------------------------------------------------------

    def submit(self, script: str) -> str:
        """Write the script under jobs/ and sbatch it. Returns the job id."""
        name = f"track-{time.strftime('%Y%m%d-%H%M%S')}.sbatch"
        path = f"{self.remote_dir}/jobs/{name}"
        out = self.ssh(f"mkdir -p {self.remote_dir}/jobs && cat > {path} && "
                       f"sbatch --parsable {path}", input=script)
        job_id = out.strip().split(";")[0]
        if not job_id.isdigit():
            raise LaunchError(f"sbatch did not return a job id: {out.strip()!r}")
        return job_id

    def job_state(self, job_id: str) -> JobState:
        return parse_squeue(self.ssh(f'squeue -h -j {job_id} -o "%T %N %r"',
                                     check=False))

    def wait_running(self, job_id: str, *, timeout: float, poll: float = 5.0) -> str:
        """Block until the job runs; return its node. Raise if it dies or times out."""
        deadline = time.monotonic() + timeout
        last = None
        while True:
            state = self.job_state(job_id)
            if state.running:
                return state.node
            if state.gone:
                raise LaunchError(
                    f"job {job_id} left the queue before it ran:\n"
                    + self.log_tail(job_id))
            if (state.state, state.reason) != last:
                why = f" ({state.reason})" if state.reason else ""
                self.say(f"job {job_id}: {state.state}{why}")
                last = (state.state, state.reason)
            if time.monotonic() >= deadline:
                self.cancel(job_id)
                raise LaunchError(f"job {job_id} did not start within "
                                  f"{timeout:.0f} s; cancelled it")
            time.sleep(poll)

    def log_tail(self, job_id: str, lines: int = 20) -> str:
        return self.ssh(f"tail -n {int(lines)} {self.remote_dir}/jobs/"
                        f"slurm-{job_id}.out 2>/dev/null", check=False)

    def cancel(self, job_id: str) -> None:
        self.ssh(f"scancel {job_id}", check=False)

    def list_jobs(self) -> list[tuple[str, str, str, str]]:
        out = self.ssh(f'squeue -h -u {self.user} -n {JOB_NAME} -o "%i %T %N %M"',
                       check=False)
        rows = []
        for line in out.splitlines():
            parts = line.split()
            if parts:
                rows.append(tuple((parts + ["", "", ""])[:4]))
        return rows

    # -- tunnel ------------------------------------------------------------------------

    def open_tunnel(self, node: str, *, timeout: float = 60.0) -> subprocess.Popen:
        if not port_free(self.port):
            raise LaunchError(f"local port {self.port} is in use (an old tunnel? "
                              "`tktomo-track-maxwell stop`, or pick --port)")
        cmd = ["ssh", "-N", "-o", "ExitOnForwardFailure=yes",
               "-L", f"{self.port}:localhost:{self.port}"] + self.node_ssh_args(node)
        self.say(f"tunnel: {' '.join(cmd)}")
        proc = self.spawn(cmd, detach=True)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise LaunchError(f"tunnel to {node} exited with {proc.returncode}")
            if not port_free(self.port):
                return proc
            time.sleep(0.5)
        proc.terminate()
        raise LaunchError(f"tunnel to {node} did not come up within {timeout:.0f} s")


# ---------------------------------------------------------------------------
# state file: the tunnel pid, so `stop` can find it
# ---------------------------------------------------------------------------

def _load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except (OSError, ValueError):
        return {}


def _save_state(state: dict) -> None:
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state))
    except OSError:
        pass


def _kill_pid(pid: int) -> bool:
    try:
        os.kill(int(pid), 15)
    except (OSError, ValueError):
        return False
    return True


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def repo_root() -> Path:
    """The checkout this module was imported from."""
    return Path(__file__).resolve().parents[3]


def cmd_setup(remote: Remote, args) -> int:
    remote.resolve_dir()
    remote.sync_source(repo_root())
    remote.build_env(rebuild=args.rebuild)
    remote.say("done: `tktomo-track-maxwell start <stack.h5>`")
    return EXIT_OK


def cmd_start(remote: Remote, args) -> int:
    remote.resolve_dir()
    if not args.no_sync:
        if not remote.env_exists():
            raise LaunchError(f"no env at {remote.host}:{remote.remote_dir}/env: "
                              "run `tktomo-track-maxwell setup` first")
        remote.sync_source(repo_root())
    script = sbatch_script(remote_dir=remote.remote_dir, stack=args.stack,
                           port=remote.port, partition=args.partition,
                           time_limit=args.time, cpus=args.cpus, mem=args.mem,
                           constraint=args.constraint)
    job_id = remote.submit(script)
    remote.say(f"submitted job {job_id} on {remote.host} ({args.partition}, "
               f"{args.time}, {args.cpus} cpus)")
    tunnel = None
    keep = args.keep
    try:
        node = remote.wait_running(job_id, timeout=args.wait)
        remote.say(f"job {job_id} running on {node}")
        tunnel = remote.open_tunnel(node)
        _save_state({"job": job_id, "node": node, "port": remote.port,
                     "tunnel_pid": tunnel.pid, "host": remote.host})
        address = f"tcp://127.0.0.1:{remote.port}"
        if args.no_app:
            remote.say(f"connect with: python -m tktomo.ui.track_model_app "
                       f"--connect {address}")
            remote.say("stop with:    tktomo-track-maxwell stop")
            keep = True
            tunnel = None      # leave it up for the user's own client
            return EXIT_OK
        app = [sys.executable, "-m", "tktomo.ui.track_model_app",
               "--connect", address]
        if args.exact_frames:
            app.append("--exact-frames")
        remote.say("starting the track-model window")
        rc = remote.spawn(app).wait()
        if rc != 0:
            remote.say(f"the window exited with {rc}")
        return EXIT_OK if rc == 0 else EXIT_ERROR
    except KeyboardInterrupt:
        remote.say("interrupted")
        return EXIT_ERROR
    finally:
        if tunnel is not None:
            tunnel.terminate()
        if keep:
            if not args.no_app:
                remote.say(f"job {job_id} kept running on {remote.host}; "
                           "`tktomo-track-maxwell stop` ends it")
        else:
            remote.cancel(job_id)
            remote.say(f"cancelled job {job_id}")
            _save_state({})


def cmd_status(remote: Remote, args) -> int:
    rows = remote.list_jobs()
    if not rows:
        remote.say(f"no {JOB_NAME} jobs for {remote.user} on {remote.host}")
    for job_id, state, node, elapsed in rows:
        remote.say(f"job {job_id}: {state} {node} {elapsed}".rstrip())
    state = _load_state()
    if state.get("tunnel_pid") and not port_free(int(state.get("port", remote.port))):
        remote.say(f"tunnel up on port {state.get('port')} to {state.get('node')} "
                   f"(pid {state['tunnel_pid']})")
    return EXIT_OK


def cmd_stop(remote: Remote, args) -> int:
    rows = remote.list_jobs()
    for job_id, *_ in rows:
        remote.cancel(job_id)
        remote.say(f"cancelled job {job_id}")
    if not rows:
        remote.say(f"no {JOB_NAME} jobs to cancel")
    state = _load_state()
    if state.get("tunnel_pid") and _kill_pid(state["tunnel_pid"]):
        remote.say(f"closed tunnel (pid {state['tunnel_pid']})")
    _save_state({})
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tktomo-track-maxwell",
        description="Run the track-model stack server on a Maxwell compute node "
                    "and connect the window on this machine to it.")
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help="ssh alias of the login node (default: %(default)s)")
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR,
                        help="directory on the cluster for source, env and "
                             "job logs (default: %(default)s, the Maxwell "
                             "scratch: no quota, wiped after 3 months, "
                             "`setup` rebuilds it)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="port on the node and on this machine "
                             "(default: %(default)s)")
    parser.add_argument("-q", "--quiet", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("setup", help="sync the source and build the env (once)")
    p.add_argument("--rebuild", action="store_true",
                   help="recreate the env even if it exists")

    p = sub.add_parser("start", help="submit the server, tunnel, run the window")
    p.add_argument("stack", nargs="?",
                   help="stack to open, as a path on the cluster (optional: "
                        "File > Open remote stack works too)")
    p.add_argument("--partition", default="maxcpu")
    p.add_argument("--time", default="04:00:00", help="walltime (default: %(default)s)")
    p.add_argument("--cpus", type=int, default=8)
    p.add_argument("--mem", default=None, help="e.g. 64G (default: partition default)")
    p.add_argument("--constraint", default=None)
    p.add_argument("--wait", type=float, default=900.0,
                   help="seconds to wait for the job to start (default: %(default)s)")
    p.add_argument("--no-sync", action="store_true",
                   help="do not rsync the source first")
    p.add_argument("--no-app", action="store_true",
                   help="leave the tunnel up and print the connect line "
                        "instead of starting the window")
    p.add_argument("--keep", action="store_true",
                   help="leave the job running when the window closes")
    p.add_argument("--exact-frames", action="store_true",
                   help="passed to the window: unpacked float32 frames")

    sub.add_parser("status", help="list this user's server jobs")
    sub.add_parser("stop", help="cancel this user's server jobs and the tunnel")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    remote = Remote(host=args.host, remote_dir=args.remote_dir, port=args.port,
                    quiet=args.quiet)
    command = {"setup": cmd_setup, "start": cmd_start,
               "status": cmd_status, "stop": cmd_stop}[args.command]
    try:
        return command(remote, args)
    except LaunchError as exc:
        print(f"tktomo-track-maxwell: {exc}", file=sys.stderr)
        return exc.code


if __name__ == "__main__":
    raise SystemExit(main())
