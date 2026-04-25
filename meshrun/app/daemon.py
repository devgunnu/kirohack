"""Cross-platform worker daemon management.

Spawns `meshrun worker` as a detached background process so the user's
terminal is free to run `meshrun submit`, `meshrun status`, etc. Tracks
the process via a PID file under ~/.meshrun/.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

STATE_DIR = Path.home() / ".meshrun"
PID_FILE = STATE_DIR / "worker.pid"
LOG_FILE = STATE_DIR / "worker.log"

IS_WINDOWS = sys.platform == "win32"


def _pid_alive(pid: int) -> bool:
    """Return True if a process with the given PID is running."""
    if pid <= 0:
        return False
    try:
        import psutil  # type: ignore

        return psutil.pid_exists(pid)
    except ImportError:
        pass

    if IS_WINDOWS:
        # signal 0 raises on Windows; fall back to tasklist.
        try:
            output = subprocess.check_output(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=3.0,
            )
            return str(pid) in output
        except Exception:
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True


def read_pid() -> int | None:
    """Return the PID recorded in the PID file, or None if no worker is tracked."""
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
    except (ValueError, OSError):
        return None
    if not _pid_alive(pid):
        try:
            PID_FILE.unlink()
        except OSError:
            pass
        return None
    return pid


def is_running() -> bool:
    """Return True if a worker daemon is currently tracked and alive."""
    return read_pid() is not None


def spawn_worker(
    coordinator: str,
    gpu_limit_mb: int,
    data_port: int = 9100,
    grpc_port: int = 50052,
    advertise: str | None = None,
    device: int = 0,
) -> int:
    """Spawn the worker as a detached background process.

    Returns the PID of the spawned process. Raises RuntimeError if a
    worker is already running.
    """
    existing = read_pid()
    if existing is not None:
        raise RuntimeError(f"Worker already running (PID {existing})")

    STATE_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "meshrun.app.cli.main",
        "worker",
        "--coordinator", coordinator,
        "--data-port", str(data_port),
        "--grpc-port", str(grpc_port),
        "--gpu-limit", str(gpu_limit_mb),
        "--device", str(device),
    ]
    if advertise:
        cmd.extend(["--advertise", advertise])

    log_fh = open(LOG_FILE, "ab")

    popen_kwargs: dict = {
        "stdout": log_fh,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
        "close_fds": True,
    }

    if IS_WINDOWS:
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        CREATE_NO_WINDOW = 0x08000000
        popen_kwargs["creationflags"] = (
            DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
        )
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, **popen_kwargs)

    PID_FILE.write_text(str(proc.pid))
    return proc.pid


def stop_worker(timeout: float = 10.0) -> bool:
    """Stop the running worker daemon.

    Sends SIGTERM (or the Windows equivalent) and waits up to `timeout`
    seconds for it to exit. Returns True if a worker was stopped, False
    if none was running.
    """
    pid = read_pid()
    if pid is None:
        return False

    try:
        if IS_WINDOWS:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5.0,
            )
        else:
            os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    except Exception:
        pass

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _pid_alive(pid):
            break
        time.sleep(0.2)

    if _pid_alive(pid):
        try:
            if IS_WINDOWS:
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid), "/T"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5.0,
                )
            else:
                os.kill(pid, signal.SIGKILL)
        except Exception:
            pass

    try:
        PID_FILE.unlink()
    except OSError:
        pass
    return True
