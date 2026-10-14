# UnbiasedStrata (PR feat/custom-model) — Operations Manual

## Prerequisites
- Git checkout: `feat/custom-model`
- CUDA toolkit + CMake + make
- Python 3, `hf` CLI, Hugging Face access
- (Optional) Docker + docker compose, mise

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

## OpenCode configuration

```json
{
  "provider": {
    "openai": {
      "options": {
        "baseURL": "http://192.168.1.42:8080/v1",
        "apiKey": "local"
      },
      "models": {
        "qwen3.6-35b-a3b-ud-q4_k_xl": {
          "name": "Unbiased Strata Qwen3.6 UD-Q4_K_XL",
          "options": { "temperature": 0.6, "top_p": 0.95, "max_tokens": 8192 }
        }
      }
    }
  },
  "model": "openai/qwen3.6-35b-a3b-ud-q4_k_xl"
}
```

## Observability stack (moved out)

The Langfuse + LiteLLM stack now lives in a separate repo:
`../Strata-Observability` (see its `README.md`).

Quickstart there:
```bash
cd ../Strata-Observability
cp .env.example .env   # fill secrets + Langfuse keys
docker compose up -d
```

Point clients to LiteLLM (`http://<host>:4000/v1`) when enabled; otherwise use Strata directly at `:8080`.

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