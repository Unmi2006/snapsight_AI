# SnapSight AI — Vercel Deployment

## Important architecture note

Vercel should host the **React/Vite frontend** only.

The SnapSight FastAPI backend uses local AI models, filesystem access, hardware/runtime detection, OCR/STT/TTS, and potentially Snapdragon NPU execution. Those workloads should run on the Snapdragon PC or another suitable backend host, not as a normal Vercel frontend deployment.

## Deploy the frontend

1. Push this repository to GitHub.
2. In Vercel, import the repository.
3. Set **Root Directory** to:
   `frontend`
4. Vercel should detect Vite automatically.
5. Build command:
   `npm run build`
6. Output directory:
   `dist`
7. Add an environment variable:
   `VITE_API_URL=https://YOUR-BACKEND-DOMAIN`
8. Deploy.

## Local development

Inside `frontend/`:

```bash
npm install
npm run dev
```

If `VITE_API_URL` is not set, the app uses:

`http://127.0.0.1:8000`

## Backend CORS

The FastAPI backend must allow the deployed Vercel origin in its CORS configuration. For example, add your Vercel domain to `CORS_ORIGINS`.

Do not put private API keys or secrets in `VITE_*` variables. Vite exposes `VITE_*` values to the browser.

## Backend

Run the backend separately on the machine that has the local models:

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

For a public deployment, use HTTPS and a secure backend host/tunnel. A browser cannot call `127.0.0.1` on your PC when the frontend is running from Vercel.

## What stays local

Keep the following out of GitHub/Vercel:

- `backend/model_weights/llm/`
- other large model files
- `.env` files containing secrets
- `node_modules/`
