"""Load project-root `.env` into the process environment (minimal parser).

Does not override variables already set in the environment.
"""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """`exllamav3-tabbyapi-docker-config/` (parent of `scripts/`)."""
    return Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | None = None) -> None:
    """Parse KEY=VALUE lines; supports `#` comments and blank lines."""
    env_path = path or (project_root() / ".env")
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        if key in os.environ:
            continue
        val = value.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        os.environ[key] = val
