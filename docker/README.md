# Observability stack (Docker)

Provides **Langfuse** (tracing UI) + **LiteLLM** proxy in front of Strata.

## Prerequisites
- Strata running with `--host 0.0.0.0` on port 8080.
- Docker / docker compose.

## Configure

Run from repo root:
```bash
mise run obs:secrets
```

This creates `docker/.env` with random `NEXTAUTH_SECRET`, `SALT`, and `LITELLM_MASTER_KEY` if it doesn't exist. Then edit `docker/.env` to add Langfuse keys after creating a project.

## Start

From repo root:
```bash
mise run obs:up
```

Or directly:
```bash
cd docker
docker compose up -d
```

## Activate Langfuse tracing

1. Open `http://localhost:3000`
2. Create an account and project
3. Copy public/secret keys into `docker/.env`
4. Restart LiteLLM:

```bash
mise run obs:restart-litellm
```

## Client endpoints
- LiteLLM/OpenAI base URL: `http://127.0.0.1:4000/v1`
- API key: value of `LITELLM_MASTER_KEY`
- Langfuse UI: `http://localhost:3000`

## Stop
```bash
mise run obs:down
```

## Notes
- Secrets are only in `docker/.env` (gitignored). Never commit real values.
- If Strata is on a different host, update `api_base` in `litellm-config.yaml`.