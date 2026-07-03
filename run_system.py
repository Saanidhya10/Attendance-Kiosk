"""
run_system.py
==============

Phase 5 -- single-command orchestrator for the enterprise Office
Attendance Management System. Launches the FastAPI backend
(``uvicorn main:app``) and the Streamlit dashboard (``streamlit run
dashboard.py``) as managed subprocesses, streams both logs into one
terminal with colored ``[BACKEND]`` / ``[FRONTEND]`` prefixes, and
guarantees clean shutdown of both child processes on Ctrl+C.

Usage:
    python run_system.py

Dependencies:
    pip install uvicorn streamlit
    pip install psutil     # optional, enables auto-killing stale port holders
    pip install colorama   # optional, improves color support on older Windows terminals

Design notes
------------
* Child processes are started in their own process group/session
  (``CREATE_NEW_PROCESS_GROUP`` on Windows, ``os.setsid`` on POSIX) so a
  Ctrl+C in this console does **not** propagate directly to them. That
  gives this script full control over shutdown: it catches the
  KeyboardInterrupt itself and then signals each child explicitly,
  first gracefully (SIGTERM / CTRL_BREAK_EVENT) and then forcefully
  (SIGKILL / taskkill) if it doesn't exit in time.
* SIGTERM is also translated into the same shutdown path (on POSIX), so
  this script cleans up properly if it's stopped by a process manager,
  not just by an interactive Ctrl+C.
* Output streaming uses one daemon thread per child process reading
  line-by-line from a merged stdout/stderr pipe, so neither process's
  logs block the other's or block the main supervisory loop.
"""

from __future__ import annotations

import logging
import os
import platform
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Optional

# --- Optional dependencies -------------------------------------------------

try:
    import psutil
except ImportError:
    psutil = None  # auto-kill-stale-process feature will be disabled

try:
    import colorama
    from colorama import Fore, Style

    colorama.init(autoreset=False)
    COLOR_BACKEND = Fore.CYAN
    COLOR_FRONTEND = Fore.MAGENTA
    COLOR_ORCHESTRATOR = Fore.YELLOW
    COLOR_RESET = Style.RESET_ALL
except ImportError:
    # Plain ANSI codes -- work fine on macOS/Linux terminals and modern
    # Windows Terminal / PowerShell 7+. Legacy cmd.exe without colorama
    # may just print these as raw text, which is a harmless degradation.
    COLOR_BACKEND = "\033[96m"
    COLOR_FRONTEND = "\033[95m"
    COLOR_ORCHESTRATOR = "\033[93m"
    COLOR_RESET = "\033[0m"


# --- Configuration -----------------------------------------------------

HOST: str = "127.0.0.1"
BACKEND_PORT: int = 8000
FRONTEND_PORT: int = 8501
GRACEFUL_SHUTDOWN_TIMEOUT: int = 10  # seconds to wait before force-killing


logging.basicConfig(
    level=logging.INFO,
    format=f"{COLOR_ORCHESTRATOR}[ORCHESTRATOR]{COLOR_RESET} %(message)s",
)
logger = logging.getLogger("run_system")


# =============================================================================
# Port checking / freeing
# =============================================================================

def is_port_in_use(port: int, host: str = HOST) -> bool:
    """Check whether something is already listening on ``host:port``.

    Args:
        port: TCP port to check.
        host: Host/interface to check against.

    Returns:
        ``True`` if a connection could be established (port is occupied).
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _kill_process_on_port(port: int) -> bool:
    """Attempt to find and terminate whatever process is listening on ``port``.

    Requires ``psutil`` -- if it isn't installed, this is a no-op that
    always returns ``False`` so the caller can fall back to printing
    manual instructions.

    Args:
        port: TCP port whose listening process should be terminated.

    Returns:
        ``True`` if a process was found and successfully terminated.
    """
    if psutil is None:
        return False

    killed_any = False
    try:
        connections = psutil.net_connections(kind="inet")
    except (psutil.AccessDenied, PermissionError) as exc:
        logger.warning("Insufficient permissions to inspect network connections: %s", exc)
        return False

    for conn in connections:
        if not conn.laddr or conn.laddr.port != port:
            continue
        if conn.status != psutil.CONN_LISTEN or conn.pid is None:
            continue
        try:
            proc = psutil.Process(conn.pid)
            logger.warning(
                "Terminating stale process %r (PID %d) holding port %d.",
                proc.name(), conn.pid, port,
            )
            proc.terminate()
            proc.wait(timeout=5)
            killed_any = True
        except psutil.NoSuchProcess:
            killed_any = True  # already gone
        except (psutil.AccessDenied, psutil.TimeoutExpired) as exc:
            logger.error("Could not terminate PID %d on port %d: %s", conn.pid, port, exc)

    return killed_any


def _print_manual_kill_instructions(port: int) -> None:
    """Print OS-specific commands the user can run to free a stuck port."""
    print(f"\nPlease free port {port} manually, then re-run this script:\n")
    if platform.system() == "Windows":
        print(f"    netstat -ano | findstr :{port}")
        print(f"    taskkill /PID <pid_from_above> /F\n")
    else:
        print(f"    lsof -i :{port}")
        print(f"    kill -9 <pid_from_above>\n")


def ensure_port_available(port: int, service_name: str) -> None:
    """Verify a port is free, auto-killing a stale occupant if possible.

    If the port is occupied and it can't be freed automatically (either
    because ``psutil`` isn't installed, or the termination attempt
    fails), this prints manual instructions and exits the whole script
    with a non-zero status -- per spec, we do not attempt to launch on
    top of an already-occupied port.

    Args:
        port: TCP port required by ``service_name``.
        service_name: Human-readable name used in log messages.
    """
    if not is_port_in_use(port):
        logger.info("Port %d is free for %s.", port, service_name)
        return

    logger.warning("Port %d (%s) is already in use.", port, service_name)

    if psutil is None:
        logger.warning(
            "psutil is not installed, so stale processes can't be auto-killed. "
            "Install it with `pip install psutil` to enable this, or free the port manually."
        )
        _print_manual_kill_instructions(port)
        sys.exit(1)

    logger.warning("Attempting to terminate the process using port %d...", port)
    if _kill_process_on_port(port):
        time.sleep(1.0)  # give the OS a moment to release the socket
        if not is_port_in_use(port):
            logger.info("Port %d freed successfully.", port)
            return
        logger.error("Port %d is still in use after termination attempt.", port)
    else:
        logger.error("Could not find or terminate the process holding port %d.", port)

    _print_manual_kill_instructions(port)
    sys.exit(1)


# =============================================================================
# Process launching
# =============================================================================

def _popen_kwargs() -> dict:
    """Build platform-appropriate kwargs for launching a managed child process.

    On Windows, the child is placed in a new process group so it doesn't
    receive the console's Ctrl+C directly (we relay shutdown ourselves via
    CTRL_BREAK_EVENT). On POSIX, the child is made a new session leader via
    ``os.setsid`` for the same reason (SIGINT from the terminal only
    targets the foreground process group, which the child is removed from).
    """
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"  # real-time log streaming from Python children
    env["PYTHONIOENCODING"] = "utf-8"  # prevent UnicodeEncodeError from deepface emojis

    kwargs: dict = dict(
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    if platform.system() == "Windows":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["preexec_fn"] = os.setsid  # noqa: PLW1509 -- intentional, single-threaded fork
    return kwargs


def start_process(command: list[str], label: str) -> subprocess.Popen:
    """Launch a child process with merged stdout/stderr piped back to us.

    Args:
        command: Argv list to execute (no shell interpolation).
        label: Human-readable label used in log messages.

    Returns:
        The running ``subprocess.Popen`` handle.

    Raises:
        SystemExit: If the executable can't be found/started at all.
    """
    logger.info("Starting %s: %s", label, " ".join(command))
    try:
        return subprocess.Popen(command, **_popen_kwargs())
    except (OSError, FileNotFoundError) as exc:
        logger.error("Failed to start %s: %s", label, exc)
        sys.exit(1)


def stream_output(process: subprocess.Popen, prefix: str, color: str) -> None:
    """Continuously relay a child process's output to our stdout, prefixed and colored.

    Runs in its own daemon thread for the lifetime of the process. Exits
    quietly once the pipe closes (either the process exited, or we closed
    it ourselves during shutdown).

    Args:
        process: The child process to read output from.
        prefix: Literal prefix to prepend to every line, e.g. ``"[BACKEND]"``.
        color: ANSI color code to wrap the prefix in.
    """
    if process.stdout is None:
        return
    try:
        for line in iter(process.stdout.readline, ""):
            if line == "":
                break
            sys.stdout.write(f"{color}{prefix}{COLOR_RESET} {line}")
            sys.stdout.flush()
    except (ValueError, OSError):
        # Pipe closed out from under us during shutdown -- not an error.
        pass


# =============================================================================
# Shutdown
# =============================================================================

def shutdown_process(process: subprocess.Popen, label: str, timeout: int = GRACEFUL_SHUTDOWN_TIMEOUT) -> None:
    """Terminate a managed child process gracefully, then forcefully if needed.

    Args:
        process: The child process to stop.
        label: Human-readable label used in log messages.
        timeout: Seconds to wait for graceful exit before force-killing.
    """
    if process.poll() is not None:
        logger.info("%s already exited (code %s).", label, process.poll())
        return

    logger.info("Stopping %s (PID %d)...", label, process.pid)
    try:
        if platform.system() == "Windows":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except (ProcessLookupError, OSError) as exc:
        logger.warning("Could not send graceful stop signal to %s: %s", label, exc)

    try:
        process.wait(timeout=timeout)
        logger.info("%s stopped gracefully.", label)
        return
    except subprocess.TimeoutExpired:
        logger.warning("%s did not stop within %ds -- forcing termination.", label, timeout)

    try:
        if platform.system() == "Windows":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                capture_output=True, check=False,
            )
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait(timeout=5)
        logger.info("%s force-killed.", label)
    except Exception as exc:  # noqa: BLE001 -- best-effort cleanup, never re-raise here
        logger.error("Failed to force-kill %s: %s", label, exc)
    finally:
        if process.stdout is not None:
            process.stdout.close()


def shutdown_all(backend: Optional[subprocess.Popen], frontend: Optional[subprocess.Popen]) -> None:
    """Shut down both child processes, ensuring neither is left as a zombie/orphan.

    Args:
        backend: The FastAPI/uvicorn process handle, if it was started.
        frontend: The Streamlit process handle, if it was started.
    """
    logger.info("Shutting down all managed processes...")
    if frontend is not None:
        shutdown_process(frontend, "Streamlit dashboard")
    if backend is not None:
        shutdown_process(backend, "FastAPI backend")
    logger.info("Shutdown complete. No processes should remain on ports %d / %d.", BACKEND_PORT, FRONTEND_PORT)


# =============================================================================
# Main
# =============================================================================

def _install_sigterm_handler() -> None:
    """Route SIGTERM through the same shutdown path as Ctrl+C (POSIX only).

    Windows doesn't have a directly analogous, reliably-catchable SIGTERM,
    so this is a no-op there; Ctrl+C (KeyboardInterrupt) is already handled
    in ``main()``.
    """
    if platform.system() == "Windows":
        return

    def _handler(signum, frame):  # noqa: ANN001, ARG001
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, _handler)


def main() -> None:
    _install_sigterm_handler()

    print("=" * 70)
    print("  Office Attendance Management System -- Orchestrator")
    print("=" * 70)

    ensure_port_available(BACKEND_PORT, "FastAPI backend")
    ensure_port_available(FRONTEND_PORT, "Streamlit dashboard")

    backend_cmd = [
        sys.executable, "-m", "uvicorn", "main:app",
        "--host", HOST, "--port", str(BACKEND_PORT),
    ]
    frontend_cmd = [
        sys.executable, "-m", "streamlit", "run", "dashboard.py",
        "--server.port", str(FRONTEND_PORT),
    ]

    backend_proc: Optional[subprocess.Popen] = None
    frontend_proc: Optional[subprocess.Popen] = None

    try:
        backend_proc = start_process(backend_cmd, "FastAPI backend")
        frontend_proc = start_process(frontend_cmd, "Streamlit dashboard")

        threading.Thread(
            target=stream_output, args=(backend_proc, "[BACKEND]", COLOR_BACKEND), daemon=True,
        ).start()
        threading.Thread(
            target=stream_output, args=(frontend_proc, "[FRONTEND]", COLOR_FRONTEND), daemon=True,
        ).start()

        logger.info("Backend docs:  http://%s:%d/docs", HOST, BACKEND_PORT)
        logger.info("Dashboard:     http://%s:%d", HOST, FRONTEND_PORT)
        logger.info("Press Ctrl+C to stop both processes.")

        while True:
            backend_exit = backend_proc.poll()
            frontend_exit = frontend_proc.poll()
            if backend_exit is not None:
                logger.error("Backend process exited unexpectedly (code %s).", backend_exit)
                break
            if frontend_exit is not None:
                logger.error("Frontend process exited unexpectedly (code %s).", frontend_exit)
                break
            time.sleep(0.5)

    except KeyboardInterrupt:
        logger.info("Ctrl+C received -- shutting down...")
    finally:
        shutdown_all(backend_proc, frontend_proc)


if __name__ == "__main__":
    main()
