# Development guide

[Back to README](../README.md)

## Local development

Use Python 3.12 and Node 22 (or a supported newer LTS release). Run backend commands
from the repository root so `backend` and `training` imports resolve.

Windows PowerShell, without activating the virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend/requirements-dev.txt
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Linux/macOS:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --no-deps -c backend/constraints.txt torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r backend/requirements-dev.txt
.venv/bin/python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```sh
cd frontend
npm ci
npm run dev
```

Open http://localhost:5173. On PowerShell systems that block `npm.ps1`, use
`npm.cmd` in place of `npm`. Vite proxies API requests to `127.0.0.1:8000`.
The local `.venv` and `node_modules` are development dependencies, not Docker inputs.

## Configuration

Copy [`.env.example`](../.env.example) to `.env` for local overrides, or set environment variables.
Defaults resolve assets relative to the repository, not the current directory.
Compose sets asset paths, CPU device, and match threshold explicitly in [docker-compose.yml](../docker-compose.yml). Change those settings there for Docker deployments.

| Variable | Default |
| --- | --- |
| `IDCARD_DETECTOR_PATH` | `training/model/yolov8s-detect.pt` |
| `IDCARD_FACE_MODEL_PATH` | `training/model/yolov8n-face.pt` |
| `IDCARD_TEMPLATE_DIR` | `training/template_samples` |
| `IDCARD_DEVICE` | `cpu` |
| `IDCARD_MATCH_THRESHOLD` | `0.8` |
| `IDCARD_MAX_UPLOAD_BYTES` | `10485760` (10 MiB) |
| `IDCARD_MAX_IMAGE_PIXELS` | `20000000` |

If changing the upload limit, also align the UI's file limit and Nginx's body
limit (which includes multipart overhead). The current Docker image targets CPU;
GPU serving would require a different wheel/container setup. More Uvicorn workers
would each load another model copy.

## Docker deployment

Compose mounts weights and templates read-only; they are not baked into the API image.
The backend uses Linux amd64, Python 3.12, CPU PyTorch wheels, and one Uvicorn worker.
ARM hosts use Docker's amd64 emulation. Nginx serves the frontend and proxies `/api`
to the backend.

From the repository root, inspect startup failures or stop the application with:

```sh
docker compose logs backend
docker compose down
```

Health endpoint: http://localhost:8000/api/v1/health

## Validation

Windows commands from the root (substitute `.venv/bin/python` on Linux/macOS):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s training -p 'test*extraction.py' -v
.\.venv\Scripts\python.exe -m unittest discover -s training -p test_detection.py -v
.\.venv\Scripts\python.exe -m pytest backend/tests -q
.\.venv\Scripts\python.exe -m backend.smoke
```

Frontend:

```sh
cd frontend
npm test
npm run build
```

With Compose running, `npm run test:e2e` in `frontend/` runs a real-upload browser
check and saves desktop/mobile screenshots in `artifacts/browser/`. The default
Playwright configuration uses installed Microsoft Edge in headless mode. If Edge
is unavailable, install Playwright Chromium and change `channel` in
`frontend/playwright.config.ts` to `chromium`.

Unit tests use mocked models for deterministic outcomes. The separate real-weight
smoke check compares training and serving inference labels and crop pixels, then
exercises response encoding. It writes local crops and JSON to ignored
`artifacts/smoke/`. Those artifacts may contain document information; they are
not bundled into the frontend.
