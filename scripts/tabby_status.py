#!/usr/bin/env python3
"""Print TabbyAPI currently loaded model and host GPU memory (nvidia-smi).

Uses the same env vars as test_tabby_infer.py:
  OPENAI_BASE_URL (default http://localhost:8000/v1)
  OPENAI_API_KEY  (default local-dev-key)

TabbyAPI exposes the active weights via GET /v1/model; it does not report VRAM.
VRAM lines come from `nvidia-smi` on the machine where you run this script
(typically WSL2 — reflects the same GPU the container uses).

  TABBY_SKIP_NVIDIA_SMI=1  — only query TabbyAPI, skip GPU query.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib import error, request

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from env_util import load_dotenv


def _read_env(name: str, default: str) -> str:
    value = os.environ.get(name, default).strip()
    return value if value else default


load_dotenv()

BASE_URL = _read_env("OPENAI_BASE_URL", "http://localhost:8000/v1").rstrip("/")
API_KEY = _read_env("OPENAI_API_KEY", "local-dev-key")
MODEL_URL = f"{BASE_URL}/model"


def http_get_json(url: str) -> tuple[int, dict | list | None]:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json",
    }
    req = request.Request(url=url, method="GET", headers=headers)
    try:
        with request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            parsed: dict | list | None = json.loads(body) if body else None
            return resp.status, parsed
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body) if body else None
        except json.JSONDecodeError:
            parsed = {"raw": body}
        return exc.code, parsed
    except error.URLError as exc:
        print(f"[NETWORK ERROR] url={url} reason={exc.reason}", file=sys.stderr)
        raise


def _shorten_params(params: object) -> object:
    """Avoid dumping multi-kB Jinja into the terminal."""
    if not isinstance(params, dict):
        return params
    out = dict(params)
    content = out.get("prompt_template_content")
    if isinstance(content, str) and len(content) > 200:
        out["prompt_template_content"] = f"<omitted {len(content)} chars>"
    return out


def print_tabby_model() -> int:
    status, payload = http_get_json(MODEL_URL)
    if status != 200:
        print(f"[TabbyAPI] GET /v1/model -> HTTP {status}")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1

    if not isinstance(payload, dict):
        print(f"[TabbyAPI] Unexpected JSON type: {type(payload).__name__}")
        print(payload)
        return 1

    mid = payload.get("id")
    if not mid:
        print("[TabbyAPI] No model id in response (nothing loaded?):")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1

    print(f"[TabbyAPI] loaded_model_id={mid}")
    params = payload.get("parameters")
    if isinstance(params, dict):
        print("[TabbyAPI] parameters:")
        print(json.dumps(_shorten_params(params), ensure_ascii=False, indent=2))
    else:
        print(f"[TabbyAPI] parameters={params!r}")
    return 0


def print_nvidia_smi() -> int:
    if os.environ.get("TABBY_SKIP_NVIDIA_SMI", "").strip() in ("1", "true", "yes"):
        print("[GPU] TABBY_SKIP_NVIDIA_SMI set — skipping nvidia-smi.")
        return 0

    exe = shutil.which("nvidia-smi")
    if not exe:
        print("[GPU] nvidia-smi not found in PATH — cannot report VRAM.")
        return 0

    cmd = [
        exe,
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[GPU] nvidia-smi failed: {exc}", file=sys.stderr)
        return 1

    if proc.returncode != 0:
        print(f"[GPU] nvidia-smi exit={proc.returncode}", file=sys.stderr)
        if proc.stderr.strip():
            print(proc.stderr.strip(), file=sys.stderr)
        return 1

    lines = [ln.strip() for ln in proc.stdout.strip().splitlines() if ln.strip()]
    if not lines:
        print("[GPU] nvidia-smi returned no rows.")
        return 0

    print("[GPU] index | name | memory.used_MiB | memory.total_MiB | util.gpu_%")
    for ln in lines:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) >= 5:
            idx, name, used, total, util = parts[:5]
            print(f"[GPU] {idx} | {name} | {used} | {total} | {util}")
        else:
            print(f"[GPU] {ln}")
    print(
        "[GPU] memory.used/total are for the whole GPU process mix "
        "(Tabby + others), not Tabby-only."
    )
    return 0


def main() -> int:
    print(f"[INFO] model_url={MODEL_URL}")
    rc = print_tabby_model()
    rc_gpu = print_nvidia_smi()
    return rc if rc != 0 else rc_gpu


if __name__ == "__main__":
    sys.exit(main())
