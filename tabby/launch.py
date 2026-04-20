#!/usr/bin/env python3
"""Compatibility launcher for TabbyAPI latest image.

The latest image currently ships with dependency skew where `infinity_emb`
expects `huggingface_hub.HfFolder`, but newer huggingface_hub versions removed
that symbol. This launcher provides a small compatibility shim before starting
TabbyAPI.
"""

from __future__ import annotations

import os
from pathlib import Path

import huggingface_hub


def _token_path() -> Path:
    explicit = os.environ.get("HF_TOKEN_PATH")
    if explicit:
        return Path(explicit).expanduser()
    return Path.home() / ".cache" / "huggingface" / "token"


if not hasattr(huggingface_hub, "HfFolder"):
    class HfFolder:
        @staticmethod
        def path_token() -> str:
            return str(_token_path())

        @staticmethod
        def get_token() -> str | None:
            path = _token_path()
            if not path.exists():
                return None
            return path.read_text(encoding="utf-8").strip() or None

        @staticmethod
        def save_token(token: str) -> None:
            path = _token_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(token, encoding="utf-8")

        @staticmethod
        def delete_token() -> None:
            path = _token_path()
            if path.exists():
                path.unlink()

    huggingface_hub.HfFolder = HfFolder


from main import entrypoint  # noqa: E402


if __name__ == "__main__":
    entrypoint()
