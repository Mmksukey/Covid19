# Deploy to Render

This repo contains a FastAPI app under `web/` plus model files in `web/models`.

## One‑click deploy
1. Push this folder to a Git provider (GitHub, GitLab, or Bitbucket).
2. In Render, create a **New Web Service** from that repo. Render will detect `render.yaml`.
3. Confirm plan = Free, then create. Build will install from `web/requirements.txt` and start:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port $PORT
   ```
4. When it’s live, open the service URL. The form is at `/`.

## Notes
- Models are loaded from `web/models`. The default config is in `web/risk_config.yaml`.
- If you rename the service, update `name:` in `render.yaml` (optional).
