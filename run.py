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
it first. Create it once with ``python setup_env.py``.
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


def venv_python() -> Path:
    """The interpreter inside ``.venv`` (falling back to the current one)."""
    bin_dir = "Scripts" if os.name == "nt" else "bin"
    exe = "python.exe" if os.name == "nt" else "python"
    candidate = ROOT / ".venv" / bin_dir / exe
    if candidate.is_file():
        return candidate
    print(
        "warning: no .venv found; using the current interpreter instead.\n"
        "         run `python setup_env.py` once to create it.",
        file=sys.stderr,
    )
    return Path(sys.executable)


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
    python = venv_python()

    # Keep our status lines interleaved correctly with the child processes when
    # stdout is piped (Python block-buffers pipes, children do not).
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

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
