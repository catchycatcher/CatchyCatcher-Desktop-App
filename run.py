#!/usr/bin/env python3
"""One-command launcher for the Catchy Catcher processing server.

Starts the FastAPI server (``server.py``) and -- unless ``--no-tunnel`` -- opens
a Gradio share tunnel (``tunnel.py``) so the static web frontend can reach it
from anywhere. The tunnel prints the base62 share code to paste into the
frontend.

    python run.py               # server + public share tunnel
    python run.py --no-tunnel   # server only, reachable on the LAN
    python run.py --port 9000

The project's ``.venv`` is used automatically, so there is no need to activate
it first. If it does not exist yet, ``run.py`` runs ``setup_env.py`` for you
(installing uv, a managed Python, ffmpeg and the dependencies) and then starts
the server. A fresh checkout therefore only needs::

    python run.py
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONTEND_URL = "https://catchycatcher.github.io"


READY_MARKER = ROOT / ".venv" / ".catchy_ready"


def _venv_python() -> Path | None:
    """Path to the interpreter inside ``.venv``, or ``None`` if it is absent."""
    bin_dir = "Scripts" if os.name == "nt" else "bin"
    exe = "python.exe" if os.name == "nt" else "python"
    candidate = ROOT / ".venv" / bin_dir / exe
    return candidate if candidate.is_file() else None


def refresh_path_from_registry() -> None:
    """Re-read PATH from the Windows registry so freshly installed tools appear.

    ``winget`` (used by ``setup_env.py`` to install ffmpeg) updates the
    persistent PATH but not the environment of already-running processes, so a
    first run that just installed ffmpeg would not otherwise see it.
    """
    if os.name != "nt":
        return
    try:
        import winreg
    except ImportError:  # pragma: no cover - only reachable off Windows
        return
    locations = (
        (
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
        ),
        (winreg.HKEY_CURRENT_USER, r"Environment"),
    )
    parts: list[str] = []
    for root, subkey in locations:
        try:
            with winreg.OpenKey(root, subkey) as key:
                value, _ = winreg.QueryValueEx(key, "Path")
                parts.append(os.path.expandvars(value))
        except OSError:
            continue
    if parts:
        os.environ["PATH"] = ";".join(parts) + ";" + os.environ.get("PATH", "")


def ensure_environment() -> Path:
    """Return a usable ``.venv`` interpreter, setting the project up if needed.

    A fresh checkout has no ``.venv``: rather than failing, run ``setup_env.py``
    (uv + Python + ffmpeg + dependencies) and then carry on.
    """
    python = _venv_python()
    if python is not None and READY_MARKER.is_file():
        return python

    setup = ROOT / "setup_env.py"
    if not setup.is_file():
        if python is not None:
            return python
        print(
            "error: no .venv and no setup_env.py to create one.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print("no ready environment found - running first-time setup (setup_env.py).")
    print("this can take a few minutes and only happens once.\n")
    result = subprocess.run([sys.executable, str(setup)], cwd=str(ROOT))
    if result.returncode != 0:
        print("error: environment setup failed.", file=sys.stderr)
        raise SystemExit(result.returncode)

    # setup_env.py may have just installed ffmpeg; make it visible to children.
    refresh_path_from_registry()

    python = _venv_python()
    if python is None:
        print(
            "error: setup finished but no .venv interpreter was found.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return python


def wait_for_health(port: int, timeout: float = 40.0) -> bool:
    """Poll ``/health`` until the API answers, or we run out of patience."""
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.3)
    return False


def stop(proc: subprocess.Popen | None) -> None:
    """Terminate a child process, escalating to kill if it will not go."""
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Catchy Catcher processing server and share tunnel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("CATCHYCATCHER_PORT", "8000")),
        help="Port the API server listens on.",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("CATCHYCATCHER_HOST", "0.0.0.0"),
        help="Interface the API server binds to.",
    )
    parser.add_argument(
        "--no-tunnel",
        action="store_true",
        help="Skip the Gradio share tunnel (LAN access only).",
    )
    return parser.parse_args(argv)


def banner(port: int) -> None:
    print("=" * 64)
    print("  Catchy Catcher processing server")
    print("=" * 64)
    print(f"  API       http://127.0.0.1:{port}")
    print(f"  Docs      http://127.0.0.1:{port}/docs")
    print(f"  Frontend  {FRONTEND_URL}")
    print("=" * 64)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Line-buffer our status lines so they interleave correctly with the child
    # processes (Python block-buffers pipes; children do not).
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    # Pick up tools (e.g. ffmpeg) that an earlier setup added to the registry
    # PATH, then make sure an environment exists before starting anything.
    refresh_path_from_registry()
    python = ensure_environment()

    env = os.environ.copy()
    env["CATCHYCATCHER_HOST"] = args.host
    env["CATCHYCATCHER_PORT"] = str(args.port)

    banner(args.port)

    server: subprocess.Popen | None = None
    tunnel: subprocess.Popen | None = None
    try:
        print("starting the API server...")
        server = subprocess.Popen([str(python), str(ROOT / "server.py")], env=env)

        if wait_for_health(args.port):
            print("server is ready.")
        elif server.poll() is not None:
            print("server exited during startup.", file=sys.stderr)
            return server.returncode or 1
        else:
            print("server is slow to start; check the output above for errors.", file=sys.stderr)

        if args.no_tunnel:
            print(f"\nno tunnel: reach the API at http://<this-machine>:{args.port}")
            print("press Ctrl+C to stop.\n")
        else:
            print("\nopening a share tunnel...")
            print("paste the share code it prints into the frontend to connect.\n")
            tunnel = subprocess.Popen(
                [str(python), str(ROOT / "tunnel.py"), str(args.port), "--host", "127.0.0.1"],
                env=env,
            )

        while True:
            if server.poll() is not None:
                print(f"\nAPI server exited with code {server.returncode}.", file=sys.stderr)
                return server.returncode or 0
            if tunnel is not None and tunnel.poll() is not None:
                print(
                    "\nshare tunnel stopped; the API server is still running on the LAN.",
                    file=sys.stderr,
                )
                tunnel = None
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nshutting down...")
        return 0
    finally:
        stop(tunnel)
        stop(server)


if __name__ == "__main__":
    # Ctrl+C raises KeyboardInterrupt; SIGTERM (e.g. `kill`) becomes SystemExit so
    # the `finally` block still stops the children.
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    sys.exit(main())
