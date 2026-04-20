#!/usr/bin/env python3
import json
import os
import sys
import time
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

_DEFAULT_MODEL = "default"
BASE_URL = _read_env("OPENAI_BASE_URL", "http://localhost:8000/v1").rstrip("/")
API_KEY = _read_env("OPENAI_API_KEY", "local-dev-key")
# Prefer OPENAI_MODEL if set; otherwise follow TABBY_MODEL_NAME from project `.env`.
MODEL = _read_env("OPENAI_MODEL", _read_env("TABBY_MODEL_NAME", _DEFAULT_MODEL))
RETRIES = int(os.environ.get("TABBY_HEALTH_RETRIES", "12"))
BACKOFF_SECONDS = float(os.environ.get("TABBY_HEALTH_BACKOFF_SEC", "2"))


def http_json(method: str, url: str, payload: dict | None = None) -> tuple[int, dict]:
    data = None
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = request.Request(url=url, data=data, method=method, headers=headers)
    try:
        with request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, json.loads(body) if body else {}
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"[HTTP ERROR] status={exc.code} url={url}")
        print(body)
        raise
    except error.URLError as exc:
        print(f"[NETWORK ERROR] url={url} reason={exc.reason}")
        raise


def post_model_unload() -> None:
    """Ask TabbyAPI to drop the loaded LLM from VRAM (admin endpoint; works with disable_auth)."""
    url = f"{BASE_URL}/model/unload"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    req = request.Request(url=url, data=b"{}", method="POST", headers=headers)
    try:
        with request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
            if resp.status != 200:
                raise RuntimeError(f"Unload HTTP {resp.status}: {raw}")
            print("[INFO] POST /v1/model/unload OK (model tensors released by server).")
            if raw.strip():
                parsed = json.loads(raw)
                if parsed is not None:
                    print(json.dumps(parsed, ensure_ascii=False, indent=2))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"[HTTP ERROR] status={exc.code} url={url}")
        print(body)
        raise
    except error.URLError as exc:
        print(f"[NETWORK ERROR] url={url} reason={exc.reason}")
        raise


def wait_for_models() -> dict:
    models_url = f"{BASE_URL}/models"
    last_error: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            status, payload = http_json("GET", models_url)
            if status == 200 and isinstance(payload.get("data"), list):
                return payload
            print(f"[WARN] Unexpected /models payload: {payload}")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"[WAIT] attempt={attempt}/{RETRIES} failed, retrying...")
        if attempt < RETRIES:
            time.sleep(BACKOFF_SECONDS)
    print("[FAIL] /v1/models did not become ready in time.")
    if last_error is not None:
        raise last_error
    raise RuntimeError("TabbyAPI models endpoint unavailable")


def ensure_model_available(models_payload: dict) -> None:
    names = [item.get("id", "") for item in models_payload.get("data", []) if isinstance(item, dict)]
    print(f"[INFO] Models_in_storage={names}")
    if MODEL not in names:
        raise RuntimeError(
            f"Target model '{MODEL}' not found in /v1/models. "
            f"Check model_name and mounted model directory."
        )


def _print_throughput_metrics(body: dict, elapsed_s: float) -> None:
    """Print token counts and tok/s using OpenAI-style `usage` when present."""
    usage = body.get("usage")
    if not isinstance(usage, dict):
        if usage is None:
            usage_hint = "응답의 usage가 null이라 서버가 토큰 수를 채우지 않음"
        else:
            usage_hint = f"usage가 dict가 아님 (실제 타입: {type(usage).__name__})"
        print(
            f"[METRICS] wall_time_sec={elapsed_s:.3f} "
            f"({usage_hint} — tok/s 미계산)"
        )
        return

    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")

    parts = [
        f"wall_time_sec={elapsed_s:.3f}",
        f"prompt_tokens={prompt_tokens}",
        f"completion_tokens={completion_tokens}",
        f"total_tokens={total_tokens}",
    ]
    if isinstance(completion_tokens, int) and completion_tokens > 0 and elapsed_s > 0:
        parts.append(f"completion_tok_per_s={completion_tokens / elapsed_s:.2f}")
    if isinstance(total_tokens, int) and total_tokens > 0 and elapsed_s > 0:
        parts.append(f"total_tok_per_s={total_tokens / elapsed_s:.2f}")

    print("[METRICS] " + " ".join(parts))
    print(
        "[METRICS] completion_tok_per_s = completion_tokens / wall_time "
        "(원격 호출이면 네트워크 포함; 로컬이면 생성 시간에 가깝습니다)"
    )


def run_chat() -> None:
    chat_url = f"{BASE_URL}/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "너 자신이 누구인지 자세하게 설명해 줘."},
        ],
        "temperature": 0.2,
        "max_tokens": 2048,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    status, body = http_json("POST", chat_url, payload=payload)
    elapsed = time.perf_counter() - t0
    if status != 200:
        raise RuntimeError(f"Chat request failed with status={status}: {body}")

    choices = body.get("choices", [])
    if not choices:
        raise RuntimeError(f"Chat response has no choices: {body}")
    content = choices[0].get("message", {}).get("content", "").strip()
    print("[SUCCESS] inference completed.")
    _print_throughput_metrics(body, elapsed)
    print(content if content else json.dumps(body, ensure_ascii=False, indent=2))
    post_model_unload()


def main() -> int:
    try:
        print(f"[INFO] base_url={BASE_URL}, model={MODEL}")
        models_payload = wait_for_models()
        ensure_model_available(models_payload)
        run_chat()
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
