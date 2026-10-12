#!/usr/bin/env python3

import argparse
import secrets
import signal
import sys
import time
import warnings
from pathlib import Path
from urllib.parse import urlparse, urlunparse

# `share_code` (and its base62 alphabet, which must match the frontend's decoder)
# lives in a gradio-free module so it can be unit-tested on its own.
from share_code import share_code

try:
    # Import necessary Gradio components carefully to avoid heavy UI imports
    from gradio import networking, utils
    from gradio.exceptions import Error as GradioError
    from gradio.tunneling import (
        BINARY_FILENAME,
        BINARY_FOLDER,
        BINARY_PATH,
        BINARY_URL,
        CURRENT_TUNNELS,
        ChecksumMismatchError,
    )

    # gradio.strings existed in older Gradio releases but was removed in
    # Gradio 6. Every lookup below already has a default, so fall back to an
    # empty mapping when the module (or an individual key) is unavailable.
    try:
        from gradio import strings
        _STRINGS = strings.en
    except ImportError:
        _STRINGS = {}

    GRADIO_AVAILABLE = True
    # Define strings needed that might be missing in older/different gradio versions
    _TUNNEL_CLOSE_MSG = _STRINGS.get("TUNNEL_CLOSE_MESSAGE", "Closing tunnel...") # Use default if key missing
    _SHARE_LINK_DISPLAY_MSG = _STRINGS.get("SHARE_LINK_DISPLAY", "Running on public URL: {0}")
    _SHARE_CODE_DISPLAY_MSG = _STRINGS.get("SHARE_CODE_DISPLAY", "Share code (base-62): {0}")
    _SHARE_LINK_MSG = _STRINGS.get("SHARE_LINK_MESSAGE", "This share link expires in 72 hours. For free permanent hosting and GPU upgrades, run Gradio in Spaces: https://huggingface.co/spaces")
    _COULD_NOT_GET_SHARE_LINK_CHECKSUM_MSG = _STRINGS.get("COULD_NOT_GET_SHARE_LINK_CHECKSUM", "ERROR: Cannot setup tunnels - Gradio binary checksum mismatch ({0}). Please delete the file and try again.")
    _COULD_NOT_GET_SHARE_LINK_MSG = _STRINGS.get("COULD_NOT_GET_SHARE_LINK", "ERROR: Cannot setup tunnels.")
    _COULD_NOT_GET_SHARE_LINK_MISSING_FILE_MSG = _STRINGS.get("COULD_NOT_GET_SHARE_LINK_MISSING_FILE", "ERROR: Cannot setup tunnels - Gradio binary not found ({0}). Please ensure the file is present or download it manually from {1} and place it in the directory: {3}")


except ImportError:
    print(
        "Error: The 'gradio' package is required. Please install it: pip install gradio",
        file=sys.stderr,
    )
    GRADIO_AVAILABLE = False
    # Define dummy classes/vars/strings for the script structure to work even if import fails
    class GradioError(Exception):
        pass
    class ChecksumMismatchError(GradioError):
        pass
    CURRENT_TUNNELS = []
    BINARY_PATH = "gradio_binary_placeholder"
    BINARY_URL = "gradio_binary_url_placeholder"
    BINARY_FILENAME = "gradio_binary_filename_placeholder"
    BINARY_FOLDER = "gradio_binary_folder_placeholder"

    # Dummy strings if Gradio import failed
    _TUNNEL_CLOSE_MSG = "Closing tunnel..."
    _SHARE_LINK_DISPLAY_MSG = "Running on public URL: {0}"
    _SHARE_CODE_DISPLAY_MSG = "Share code (base-62): {0}"
    _SHARE_LINK_MSG = "This share link expires in 72 hours. For free permanent hosting and GPU upgrades, run Gradio in Spaces: https://huggingface.co/spaces"
    _COULD_NOT_GET_SHARE_LINK_CHECKSUM_MSG = "ERROR: Cannot setup tunnels - Gradio binary checksum mismatch ({0}). Please delete the file and try again."
    _COULD_NOT_GET_SHARE_LINK_MSG = "ERROR: Cannot setup tunnels."
    _COULD_NOT_GET_SHARE_LINK_MISSING_FILE_MSG = "ERROR: Cannot setup tunnels - Gradio binary not found ({0}). Please ensure the file is present or download it manually from {1} and place it in the directory: {3}"


# --- Cleanup Function ---
def cleanup(signum, frame):
    """Gracefully kills tunnel subprocesses on script exit."""
    # Use the safer string definition from above
    print("\n" + _TUNNEL_CLOSE_MSG)
    # Use Gradio's list of active tunnels
    # Make a copy because tunnel.kill() might modify the list
    tunnels_to_kill = list(CURRENT_TUNNELS)
    for tunnel in tunnels_to_kill:
        try:
            if hasattr(tunnel, 'kill') and callable(tunnel.kill):
                tunnel.kill()
                # Allow Gradio's internal cleanup to remove it if necessary
                if tunnel in CURRENT_TUNNELS:
                    CURRENT_TUNNELS.remove(tunnel)
        except Exception as e:
            print(f"Warning: Error closing tunnel process: {e}", file=sys.stderr)
    # Don't exit immediately if called from signal handler during normal operation,
    # let the main loop handle exit. Only exit if called from error handler.
    if signum is None and frame is None: # Heuristic: called from error handler
         sys.exit(1)
    else: # Called from signal handler (Ctrl+C)
         sys.exit(0)


# --- Main Execution ---
def main():
    if not GRADIO_AVAILABLE:
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Create a Gradio-style share tunnel for a locally running web application.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "port",
        type=int,
        help="The local port number your application is running on.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="The local host your application is running on (e.g., 127.0.0.1, 0.0.0.0).",
    )
    parser.add_argument(
        "--share-server-address",
        type=str,
        default=networking.GRADIO_SHARE_SERVER_ADDRESS,
        help="Use a custom FRP server address for sharing.",
    )
    parser.add_argument(
        "--share-server-protocol",
        type=str,
        default="https",
        choices=["http", "https"],
        help="Protocol for the custom FRP server (http or https).",
    )
    parser.add_argument(
        "--share-server-tls-certificate",
        type=str,
        default=None,
        help="Path to TLS certificate for connecting to a custom share server (if protocol is https and server requires it).",
    )
    parser.add_argument(
        "--share-token",
        type=str,
        default=secrets.token_urlsafe(32),
        help="Share token for authentication with the share server (default: auto-generated).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-error messages.",
    )

    args = parser.parse_args()

    if not args.quiet:
        print(
            f"Attempting to create share link for local app running at http://{args.host}:{args.port}..."
        )

    # Register signal handlers for cleanup
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        # The core call to Gradio's tunneling function
        share_url_orig = networking.setup_tunnel(
            local_host=args.host,
            local_port=args.port,
            share_token=args.share_token,
            share_server_address=args.share_server_address,
            share_server_tls_certificate=args.share_server_tls_certificate,
            # label=None # Uncomment if your Gradio version requires this
        )

        # Apply the correct protocol (http/https)
        parsed_url = urlparse(share_url_orig)
        share_url = urlunparse(
            (args.share_server_protocol,) + parsed_url[1:]
        )

        print("-" * 80)
        print(_SHARE_LINK_DISPLAY_MSG.format(share_url))
        print(_SHARE_CODE_DISPLAY_MSG.format(share_code(share_url)))
        if not args.quiet:
            print(_SHARE_LINK_MSG)
        print("-" * 80)
        print("Press Ctrl+C to stop the tunnel.")

        # Keep the script running to maintain the tunnel
        while True:
            time.sleep(1)

    except ChecksumMismatchError:
        print(
            _COULD_NOT_GET_SHARE_LINK_CHECKSUM_MSG.format(
                BINARY_PATH
            ), file=sys.stderr
        )
        sys.exit(1) # Exit directly on specific known errors
    except FileNotFoundError:
         # This might happen if the binary download fails *within* setup_tunnel
         # or if permissions are wrong.
        print(
            _COULD_NOT_GET_SHARE_LINK_MISSING_FILE_MSG.format(
                BINARY_PATH,
                BINARY_URL,
                BINARY_FILENAME,
                BINARY_FOLDER,
            ), file=sys.stderr
        )
        sys.exit(1) # Exit directly on specific known errors
    except GradioError as e:
        # Catch other Gradio-specific errors during tunneling
        # Including the RuntimeError from the original traceback
        print(f"{_COULD_NOT_GET_SHARE_LINK_MSG} Details: {e}", file=sys.stderr)
        # Check specifically for SSL/TLS errors within the details if possible
        if "CERTIFICATE_VERIFY_FAILED" in str(e) or "SSL" in str(e):
             print("\nSSL/TLS Verification failed. This might be due to a proxy, firewall, or outdated certificates.", file=sys.stderr)
             print("Try updating certifi: pip install --upgrade certifi", file=sys.stderr)
        sys.exit(1) # Exit directly on specific known errors
    except Exception as e:
        # Catch any other unexpected error
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        print("Attempting to clean up tunnels...", file=sys.stderr)
        cleanup(None, None) # Attempt cleanup and exit from within cleanup
    # The finally block is usually for cleanup that *always* runs,
    # but cleanup is handled by signals or explicit calls in except blocks.
    # finally:
    #     pass
    return share_code(share_url)


if __name__ == "__main__":
    main()