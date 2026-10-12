#!/usr/bin/env python3
"""One-command environment setup for the Catchy Catcher processing engine.

Works on Windows, macOS and Linux. Run it with any Python 3.8+ interpreter::

    python setup_env.py

It will:

1. Make sure ``uv`` is available (bootstrapping it if necessary).
2. Make sure a CPython 3.12 interpreter is available (managed by ``uv``).
3. Make sure ``ffmpeg`` is on PATH (installing it with the platform package
   manager where possible).
4. Create ``.venv`` and install the audio stack: PyTorch (CUDA, CPU or Apple
   MPS) plus demucs, whisperx, librosa and FastAPI.
5. Download the NLTK tokenizer data WhisperX's forced alignment needs.

PyTorch is pinned to the 2.8.0 line on purpose: from torchaudio 2.9 onward the
``load``/``save`` helpers hard-require TorchCodec, which in turn needs FFmpeg
*shared* libraries (v4-7). The pinned 2.8.x release still uses torchaudio's
soundfile/ffmpeg fallback, which is what :mod:`pipeline` relies on.

Options::

    python setup_env.py                 # auto-detect GPU/MPS, else CPU
    python setup_env.py --cpu           # force the CPU wheels
    python setup_env.py --python 3.11   # different interpreter version
    python setup_env.py --skip-ffmpeg   # never touch the system ffmpeg
    python setup_env.py --skip-nltk     # skip the NLTK data download
    python setup_env.py --torch-index URL   # override the PyTorch wheel index
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"

#: torch/torchaudio/torchvision versions pinned for compatibility (see above).
TORCH_VERSION = "2.8.0"
TORCHVISION_VERSION = "0.23.0"
TORCH_PINS = (
    f"torch=={TORCH_VERSION}",
    f"torchaudio=={TORCH_VERSION}",
    f"torchvision=={TORCHVISION_VERSION}",
)

DEFAULT_PYTHON = "3.12"
CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"


# ---------------------------------------------------------------------------
# Small console helpers
# ---------------------------------------------------------------------------


def log(message: str) -> None:
    print(f"[setup] {message}", flush=True)


def warn(message: str) -> None:
    print(f"[setup] WARNING: {message}", file=sys.stderr, flush=True)


def fail(message: str) -> "None":
    print(f"[setup] ERROR: {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def run(cmd: Sequence[str], *, check: bool = True, env=None) -> int:
    """Echo and run a command, returning its exit code."""
    pretty = " ".join(str(part) for part in cmd)
    log(f"$ {pretty}")
    try:
        completed = subprocess.run(cmd, check=check, env=env)
    except FileNotFoundError:
        if check:
            raise
        return 127
    return completed.returncode


# ---------------------------------------------------------------------------
# uv bootstrap
# ---------------------------------------------------------------------------


def _find_uv() -> str | None:
    found = shutil.which("uv")
    if found:
        return found
    exe = "uv.exe" if os.name == "nt" else "uv"
    for candidate in (
        Path.home() / ".local" / "bin" / exe,
        Path.home() / ".cargo" / "bin" / exe,
    ):
        if candidate.exists():
            return str(candidate)
    return None


def ensure_uv() -> str:
    uv = _find_uv()
    if uv:
        log(f"uv found: {uv}")
        return uv

    log("uv not found; installing it...")
    system = platform.system()
    installed = False
    try:
        if system == "Windows":
            run(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    "irm https://astral.sh/uv/install.ps1 | iex",
                ]
            )
        else:
            run(["sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"])
        installed = True
    except Exception as exc:  # noqa: BLE001 - fall back to pip below
        warn(f"official uv installer failed ({exc}); trying pip")

    if not installed:
        run([sys.executable, "-m", "pip", "install", "--user", "uv"], check=False)

    uv = _find_uv()
    if not uv:
        fail(
            "could not install uv automatically. Install it manually from "
            "https://docs.astral.sh/uv/ and re-run this script."
        )
    return uv


# ---------------------------------------------------------------------------
# ffmpeg
# ---------------------------------------------------------------------------

_LINUX_FFMPEG_MANAGERS = [
    (["apt-get", "install", "-y", "ffmpeg"], True),
    (["dnf", "install", "-y", "ffmpeg"], True),
    (["yum", "install", "-y", "ffmpeg"], True),
    (["zypper", "install", "-y", "ffmpeg"], True),
    (["pacman", "-S", "--noconfirm", "ffmpeg"], True),
    (["apk", "add", "ffmpeg"], True),
]


def _sudo_prefix() -> list[str]:
    if os.name == "nt":
        return []
    try:
        if os.geteuid() == 0:  # type: ignore[attr-defined]
            return []
    except AttributeError:
        pass
    return ["sudo"] if shutil.which("sudo") else []


def ensure_ffmpeg(skip: bool) -> None:
    existing = shutil.which("ffmpeg")
    if existing:
        log(f"ffmpeg found: {existing}")
        return

    if skip:
        warn("ffmpeg is missing and --skip-ffmpeg was given; install it manually.")
        return

    system = platform.system()
    log(f"ffmpeg not found; attempting to install it on {system}...")

    try:
        if system == "Windows":
            if not shutil.which("winget"):
                warn(
                    "winget is unavailable. Install ffmpeg manually and put it "
                    "on PATH: https://www.gyan.dev/ffmpeg/builds/"
                )
                return
            run(
                [
                    "winget",
                    "install",
                    "--id",
                    "Gyan.FFmpeg",
                    "-e",
                    "--accept-source-agreements",
                    "--accept-package-agreements",
                    "--disable-interactivity",
                ]
            )
            log(
                "ffmpeg installed. Open a NEW terminal (or reboot) so PATH is "
                "refreshed before running the server."
            )
        elif system == "Darwin":
            if not shutil.which("brew"):
                warn(
                    "Homebrew is unavailable. Install it from https://brew.sh "
                    "and run 'brew install ffmpeg'."
                )
                return
            run(["brew", "install", "ffmpeg"])
        else:
            for manager, needs_sudo in _LINUX_FFMPEG_MANAGERS:
                if not shutil.which(manager[0]):
                    continue
                prefix = _sudo_prefix() if needs_sudo else []
                run(prefix + manager, check=False)
                if shutil.which("ffmpeg"):
                    break
            if not shutil.which("ffmpeg"):
                warn(
                    "Automatic ffmpeg install failed. Install it with your "
                    "package manager (e.g. 'sudo apt install ffmpeg')."
                )
    except Exception as exc:  # noqa: BLE001 - never fatal; just warn
        warn(f"could not install ffmpeg automatically ({exc}).")


# ---------------------------------------------------------------------------
# venv + installs
# ---------------------------------------------------------------------------


def venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def create_venv(uv: str, python_version: str) -> None:
    if venv_python().exists():
        log(f"virtual environment already present: {VENV}")
        return
    log(f"creating virtual environment with Python {python_version}...")
    run([uv, "python", "install", python_version], check=False)
    run([uv, "venv", "--python", python_version, str(VENV)])


def _has_nvidia_gpu() -> bool:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return False
    try:
        return (
            subprocess.run(
                [smi],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
            ).returncode
            == 0
        )
    except Exception:  # noqa: BLE001
        return False


def choose_torch_index(args: argparse.Namespace) -> str | None:
    """Return the ``--index-url`` for PyTorch, or ``None`` for plain PyPI."""
    if args.torch_index:
        return args.torch_index
    system = platform.system()
    if system == "Darwin":
        # Apple silicon/macOS wheels on PyPI already include the MPS backend.
        return None
    if args.cpu or not _has_nvidia_gpu():
        return CPU_INDEX
    return CUDA_INDEX


def install_torch(uv: str, py: Path, index: str | None) -> None:
    log(f"installing PyTorch {TORCH_VERSION} ({index or 'PyPI / MPS'})...")
    cmd = [uv, "pip", "install", "--python", str(py), *TORCH_PINS]
    if index:
        cmd += ["--index-url", index]
    run(cmd)


def install_requirements(uv: str, py: Path) -> None:
    if not REQUIREMENTS.is_file():
        fail(f"requirements file not found: {REQUIREMENTS}")

    # Constraints keep the PyTorch pins from being replaced while resolving the
    # rest of the stack (whisperx happily drags in newer torch otherwise).
    fd, constraint_path = tempfile.mkstemp(prefix="catchy_constraints_", suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write("\n".join(TORCH_PINS) + "\n")
        log("installing project requirements...")
        run(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(py),
                "-r",
                str(REQUIREMENTS),
                "-c",
                constraint_path,
            ]
        )
    finally:
        try:
            os.unlink(constraint_path)
        except OSError:
            pass


def _read_torch_versions(py: Path) -> tuple[str, str] | None:
    try:
        out = subprocess.run(
            [
                str(py),
                "-c",
                "import torch,torchaudio;"
                "print(torch.__version__+' '+torchaudio.__version__)",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except Exception:  # noqa: BLE001
        return None
    if out.returncode != 0:
        return None
    parts = out.stdout.strip().split()
    return (parts[0], parts[1]) if len(parts) == 2 else None


def verify_torch(uv: str, py: Path, index: str | None) -> None:
    """Ensure the resolver did not swap in an incompatible torchaudio."""
    versions = _read_torch_versions(py)
    if versions is None:
        warn("could not import torch/torchaudio to verify the install.")
        return
    torch_v, audio_v = versions
    log(f"installed: torch {torch_v}, torchaudio {audio_v}")
    if not torch_v.startswith(TORCH_VERSION) or not audio_v.startswith(TORCH_VERSION):
        warn("PyTorch/torchaudio versions drifted; re-pinning to the verified set.")
        install_torch(uv, py, index)


def download_nltk(py: Path) -> None:
    log("downloading NLTK tokenizer data (punkt, punkt_tab)...")
    env = os.environ.copy()
    # Allows the download to pass through an egress proxy; harmless otherwise.
    env["NLTK_ALLOW_PROXIED_FETCH"] = "1"
    env["NLTK_ALLOW_PROXIED_URLOPEN"] = "1"
    code = (
        "import sys, nltk;"
        "results = [nltk.download(p) for p in ('punkt', 'punkt_tab')];"
        "sys.exit(0 if all(results) else 1)"
    )
    if run([str(py), "-c", code], check=False, env=env) != 0:
        warn("NLTK data download failed; WhisperX alignment may need it later.")


def smoke_test(py: Path) -> None:
    log("running import smoke test...")
    code = (
        "import pipeline, torch;"
        "print('device:', pipeline.detect_device());"
        "print('torch:', torch.__version__)"
    )
    run([str(py), "-c", code], check=False)


def print_next_steps() -> None:
    if os.name == "nt":
        activate_cmd = r".venv\Scripts\activate"
        python_cmd = r".venv\Scripts\python.exe server.py"
    else:
        activate_cmd = "source .venv/bin/activate"
        python_cmd = "python server.py"

    print()
    print("=" * 68)
    print("Setup complete.")
    print()
    print("Activate the environment:")
    print(f"    {activate_cmd}")
    print()
    print("Run the API:")
    print(f"    {python_cmd}")
    print("    # then open http://localhost:8000/docs")
    print()
    print("Try it on the bundled sample:")
    print(
        '    curl -X POST "http://localhost:8000/process-session" '
        '-F "file=@example.m4a" -o result.json'
    )
    print("=" * 68)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install the Catchy Catcher processing environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--python",
        default=DEFAULT_PYTHON,
        help="Python version for the virtual environment.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force the CPU-only PyTorch wheels.",
    )
    parser.add_argument(
        "--torch-index",
        default=None,
        help="Override the PyTorch wheel index URL.",
    )
    parser.add_argument(
        "--skip-ffmpeg",
        action="store_true",
        help="Do not try to install ffmpeg.",
    )
    parser.add_argument(
        "--skip-nltk",
        action="store_true",
        help="Do not download the NLTK tokenizer data.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    log(f"platform: {platform.system()} {platform.release()} ({platform.machine()})")
    log(f"project root: {ROOT}")

    uv = ensure_uv()
    ensure_ffmpeg(args.skip_ffmpeg)
    create_venv(uv, args.python)

    py = venv_python()
    if not py.exists():
        fail(f"virtual environment interpreter not found: {py}")

    index = choose_torch_index(args)
    install_torch(uv, py, index)
    install_requirements(uv, py)
    verify_torch(uv, py, index)

    if not args.skip_nltk:
        download_nltk(py)

    smoke_test(py)

    # Marker so run.py can skip the (idempotent) setup on later launches.
    try:
        (VENV / ".catchy_ready").write_text("ok\n", encoding="utf-8")
    except OSError:
        pass

    print_next_steps()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
