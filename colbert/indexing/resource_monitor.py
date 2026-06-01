"""Lightweight RAM / disk usage reporter for the indexing pipeline.

Logs, at INFO level, the current process resident memory (RSS) and the on-disk size
of a tracked directory (the index / experiments output), together with the change in
each since the previous report.  Dependency-free: RSS is read from ``/proc/self/status``
(with a ``resource`` fallback) and disk size is the sum of file sizes under the dir.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def process_rss_bytes() -> int:
    """Current resident set size of this process, in bytes (0 if unavailable)."""
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024  # kB -> bytes
    except OSError:
        pass
    try:
        import resource

        # ru_maxrss is kB on Linux (peak), bytes on macOS — best-effort fallback.
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except Exception:
        return 0


def dir_disk_bytes(path: Path) -> int:
    """Total size of all files under ``path`` (recursive), in bytes."""
    total = 0
    if not path.exists():
        return 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.stat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def format_bytes(n: float) -> str:
    """Human-readable byte size (signed-safe)."""
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PiB"


def _format_delta(n: int) -> str:
    return ("+" if n >= 0 else "-") + format_bytes(abs(n))


class ResourceMonitor:
    """Reports process RAM (RSS) and tracked-directory disk usage, with deltas.

    Args:
        track_dir: directory whose on-disk size to report (e.g. the index output).
        log: logger to emit INFO records on (defaults to this module's logger).
    """

    def __init__(self, track_dir: str | Path, log: logging.Logger | None = None):
        self.track_dir = Path(track_dir)
        self.log = log or logger
        self._last_rss: int | None = None
        self._last_disk: int | None = None

    def report(self, label: str) -> None:
        rss = process_rss_bytes()
        disk = dir_disk_bytes(self.track_dir)
        rss_delta = "" if self._last_rss is None else f" (Δ {_format_delta(rss - self._last_rss)})"
        disk_delta = "" if self._last_disk is None else f" (Δ {_format_delta(disk - self._last_disk)})"
        self.log.info(
            f"[resources] {label}: "
            f"RAM(RSS) {format_bytes(rss)}{rss_delta} | "
            f"disk[{self.track_dir.name}] {format_bytes(disk)}{disk_delta}"
        )
        self._last_rss = rss
        self._last_disk = disk
