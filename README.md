# ExLlamaV3 + TabbyAPI Docker Config

Minimal Docker Compose stack for running [TabbyAPI](https://github.com/theroyallab/tabbyAPI) with ExLlamaV3 (EXL3) quantized models on NVIDIA GPUs.

## Features

- Single `.env` file to switch models
- OpenAI-compatible API endpoint
- Health checks and auto-restart
- Compatibility launcher for TabbyAPI dependency issues

## Directory Layout

```text
exllamav3-tabbyapi-docker-config/
├── docker-compose.yml
├── .env.example
├── .gitignore
├── scripts/
│   ├── test_tabby_infer.py
│   ├── tabby_status.py
│   └── env_util.py
└── tabby/
    ├── launch.py
    └── config.yml
```

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/YOUR_USERNAME/exllamav3-tabbyapi-docker-config.git
cd exllamav3-tabbyapi-docker-config
cp .env.example .env
```

### 2. Edit `.env`

Set the path to your EXL3 models directory and choose which model to load:

```bash
# Path to your EXL3 models directory (contains model folders)
MODELS_PATH=/path/to/your/exl3/models

# Model folder name to load (must exist under MODELS_PATH)
TABBY_MODEL_NAME=Your-Model-Folder-Name
```

### Tested model (example)

- [MaxedSet/gemma-4-26B-A4B-it-exl3-4.0bpw](https://huggingface.co/MaxedSet/gemma-4-26B-A4B-it-exl3-4.0bpw)
- Set `TABBY_MODEL_NAME=gemma-4-26B-A4B-it-exl3-4.0bpw`

> VRAM note: In the model card's benchmark snapshot, this model used ~22GB VRAM on an RTX 3090 24GB environment. Actual usage depends on prompt length, max tokens, cache/chunk settings, and concurrent load.



### 3. Validate GPU access in Docker

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

### 4. Start the service

```bash
docker compose up -d
```

### 5. Verify

```bash
# Check logs
docker compose logs -f llm

# Check loaded model
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

## Configuration

### Switching Models

To switch to a different model, edit `TABBY_MODEL_NAME` in `.env` and recreate the container:

```bash
# Edit .env, then:
docker compose up -d --force-recreate
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODELS_PATH` | Host path to EXL3 models directory | (required) |
| `TABBY_MODEL_NAME` | Model folder name to load | (required) |
| `OPENAI_BASE_URL` | API base URL for scripts | `http://localhost:8000/v1` |
| `OPENAI_API_KEY` | API key (disabled by default) | `local-dev-key` |

### Memory Tuning

Edit these values in `docker-compose.yml` based on your GPU VRAM:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--max-batch-size` | Concurrent requests | `1` |
| `--chunk-size` | Prompt processing chunk | `1024` |
| `--cache-size` | KV cache tokens | `32768` |

Start conservative, increase only after verifying stable VRAM behavior.

## Testing

### Python inference test

The test script automatically reads from `.env`:

```bash
python3 scripts/test_tabby_infer.py
```

If model loading is slow, increase retry budget:

```bash
TABBY_HEALTH_RETRIES=30 TABBY_HEALTH_BACKOFF_SEC=3 python3 scripts/test_tabby_infer.py
```

### Check status

```bash
python3 scripts/tabby_status.py
```

### Example curl request

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "your-loaded-model-name",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

## Notes

- This stack runs with `--disable-auth true` for local development convenience.
- `tabby/launch.py` is a compatibility shim for `huggingface_hub.HfFolder` removal in newer versions.
- `tabby/config.yml` values are fallbacks; CLI args from `docker-compose.yml` take precedence.

## License

MIT
