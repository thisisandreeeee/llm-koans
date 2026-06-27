#!/usr/bin/env python
"""Run all koans or one numbered koan test file.

Usage:
    python tools/check.py
    python tools/check.py 03
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    args = [sys.executable, "-m", "pytest"]
    if len(sys.argv) == 2:
        prefix = sys.argv[1].zfill(2)
        matches = sorted((ROOT / "tests").glob(f"test_{prefix}_*.py"))
        if not matches:
            print(f"No test file found for koan {prefix}")
            return 2
        args.extend(str(p) for p in matches)
    return subprocess.call(args, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
