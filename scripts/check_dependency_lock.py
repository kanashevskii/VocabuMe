"""Verify every direct production dependency is pinned identically in the lockfile."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PIN_PATTERN = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s]+)$")


def parse_pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = PIN_PATTERN.fullmatch(line)
        if match is None:
            raise ValueError(f"{path.name}: expected an exact pin, got {line!r}")
        package, version = match.groups()
        pins[package.lower().replace("_", "-")] = version
    return pins


def main() -> int:
    direct = parse_pins(ROOT / "requirements-prod.in")
    locked = parse_pins(ROOT / "requirements-prod.lock")
    mismatches = [
        f"{package}: expected {version}, lock has {locked.get(package, '<missing>')}"
        for package, version in direct.items()
        if locked.get(package) != version
    ]
    if mismatches:
        print("Production dependency lock is stale:", *mismatches, sep="\n- ")
        return 1
    print("Production dependency lock matches all direct dependencies.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
