from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import socket
from typing import Iterator

from .constants import BUILD_LOCK_PATH


@contextmanager
def acquire_build_lock(lock_path: Path | None = None) -> Iterator[None]:
    resolved_path = Path(lock_path or BUILD_LOCK_PATH)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock_file.seek(0)
            raw_state = lock_file.read().strip()
            detail = _format_lock_holder(raw_state)
            raise RuntimeError(f"another site build is already running{detail}") from exc

        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(_build_lock_state())
        lock_file.flush()
        os.fsync(lock_file.fileno())
        try:
            yield
        finally:
            lock_file.seek(0)
            lock_file.truncate()
            lock_file.flush()
            os.fsync(lock_file.fileno())
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _build_lock_state() -> str:
    return json.dumps(
        {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
    )


def _format_lock_holder(raw_state: str) -> str:
    if not raw_state:
        return ""
    try:
        payload = json.loads(raw_state)
    except json.JSONDecodeError:
        return f" ({raw_state})"

    details: list[str] = []
    pid = payload.get("pid")
    if pid is not None:
        details.append(f"pid {pid}")
    host = payload.get("host")
    if host:
        details.append(f"host {host}")
    started_at = payload.get("started_at")
    if started_at:
        details.append(f"started {started_at}")
    if not details:
        return ""
    return " (" + ", ".join(details) + ")"
