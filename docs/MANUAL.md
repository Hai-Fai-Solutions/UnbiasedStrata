# UnbiasedStrata (PR feat/custom-model) — Operations Manual

## Prerequisites
- This repo checked out in the branch that contains this manual (mise tasks and configs are provided there)
- CUDA toolkit + CMake + make
- Python 3, `hf` CLI, Hugging Face access
- (Optional) Docker + docker compose, mise

## Prerequisites (no Docker required for running Strata)

- [mise](https://mise.jdx.dev/installing-mise.html) (optional, but the mise tasks assume it)
- Git, CMake, CUDA toolkit, Python 3, `hf` CLI

## Why the MTP model

Strata uses a **multi-token / MTP draft head** (cut from the `-MTP` GGUF) in addition to the trunk model. This enables speculative decoding: extra draft tokens are proposed and verified in parallel, improving generation throughput without changing output quality.

Compared to a **single-model llama.cpp-style approach**, the MTP setup needs:
- the main GGUF + a derived trunk pack
- the MTP GGUF + a draft pack
- slightly more RAM/VRAM planning (but gives materially faster decoding)

If you do not want speculation, you can still run with `--spec 0`, but startup refuses `0` because the native pack path requires `spec >= 2`; tune with `--spec 2` for a memory-friendly balance.

## One-time setup

1. Fetch models:
   ```bash
   mise run mtp:download
   ```
2. Generate packs:
   ```bash
   mise run packs:generate
   ```
   (or run `tools/iq_pack.py` / `tools/mtp_qwen36_pack.py` manually)
3. Build:
   ```bash
   mise run build
   ```

## Start the Strata API

Foreground:
```bash
mise run mtp:serve:start
```

Background (logs to `strata.log`):
```bash
mise run mtp:serve:start-bg
tail -f strata.log
```

Manual equivalent:
```bash
./build/unbiased-strata --model models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --pack models/qwen36-q4 --mtp models/qwen36-mtp-draft \
  --ctx 262144 --host 0.0.0.0 --port 8080 \
  -- --kv q4_0 --spec 2 --mtp-max-t 2
```

Access:
- Health: `http://<host>:8080/health`
- OpenAI base URL: `http://<host>:8080/v1`
- Model ID: `qwen3.6-35b-a3b-ud-q4_k_xl`

## OpenCode configuration (confirmed)

```json
{
  "$schema": "https://opencode.ai/config.json",
  "plugins": [
    "superpowers@git+https://github.com/obra/superpowers.git"
  ],
  "providers": {
    "openai": {
      "package": "@opencode/ai/providers/openai-compatible",
      "settings": {
        "baseURL": "http://<host-ip>:8080/v1",
        "apiKey": "local"
      },
      "models": {
        "qwen3.6-35b-a3b-ud-q4_k_xl": {
          "name": "Unbiased Strata Qwen3.6",
          "settings": {
            "temperature": 0.6,
            "top_p": 0.95
          },
          "limit": {
            "context": 262144,
            "output": 8192
          }
        }
      }
    }
  },
  "model": "openai/qwen3.6-35b-a3b-ud-q4_k_xl"
}
```

This configuration was verified against Strata using the OpenAI-compatible provider. Set `reasoning_effort` in request/model settings if you want thinking enabled (`"high"`), or omit/`"none"` for plain `content`.

## Firewall

UFW (if enabled):
```bash
sudo ufw allow 8080/tcp
sudo ufw allow 4000/tcp
sudo ufw reload
```

## Notes
- Keep prompts reasonably sized; very large prompts can OOM during verification.
- Do not commit real `.env` files; only `.env.example` is tracked.
- `.claude` and `.opencodeignore` prevent AI tools from reading `.env`.