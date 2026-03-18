#!/usr/bin/env python3
"""Serve the generated docs/ directory via a simple HTTP server."""

from __future__ import annotations

import argparse
import contextlib
import http.server
import os
import socketserver
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DOCS_DIR = REPO_ROOT / "docs"
DEFAULT_PORT = 8000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to bind the server to (default: {DEFAULT_PORT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not DOCS_DIR.exists():
        raise SystemExit(f"docs directory not found at {DOCS_DIR}")

    os.chdir(DOCS_DIR)

    handler = http.server.SimpleHTTPRequestHandler
    with contextlib.ExitStack() as stack:
        httpd: socketserver.TCPServer = stack.enter_context(
            socketserver.ThreadingTCPServer(("", args.port), handler)
        )
        print(f"Serving {DOCS_DIR} at http://localhost:{args.port}/ (Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")


if __name__ == "__main__":
    main()
