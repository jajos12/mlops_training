# MLOps Training Project

## Overview
This project demonstrates a minimal MLOps pipeline for a breast cancer classifier using TensorFlow, FastAPI, and Prometheus monitoring.

## Architecture
- **Model Training:** `train.py` retrains and exports models to the `exports/` directory.
- **Serving:** TensorFlow Serving loads models from `exports/`.
- **API Proxy:** `proxy_api.py` (FastAPI) provides `/predict`, `/canary`, and `/metrics` endpoints, and serves the static UI.
- **UI:**
  - `/static/index.html`: User-friendly prediction form.
  - `/static/metrics.html`: Metrics dashboard with charts.
- **Automation:** GitHub Actions workflow automates retraining, serving, and canary testing.

## Usage
1. **Train a model:**
   ```sh
   python train.py
   ```
2. **Start TensorFlow Serving:**
   ```sh
   docker run -d --rm -p 8501:8501 -v $PWD/exports:/models/my_model -e MODEL_NAME=my_model tensorflow/serving
   ```
3. **Start FastAPI Proxy:**
   ```sh
   uvicorn proxy_api:app --host 0.0.0.0 --port 8000
   ```
4. **Open the UI:**
   - Prediction: [http://localhost:8000/static/index.html](http://localhost:8000/static/index.html)
   - Metrics: [http://localhost:8000/static/metrics.html](http://localhost:8000/static/metrics.html)

## Endpoints
- `POST /predict`: Predict with the latest model (expects JSON, returns predictions).
- `POST /canary?version_a=1&version_b=2`: Compare predictions between two model versions.
- `GET /metrics`: Prometheus metrics for monitoring.

## API Key
- API keys are loaded from `.env` (`API_KEYS`).
- For browser UI, the `/predict` endpoint should be public or use a secure authentication method.

## Automation
- See `.github/workflows/retrain.yml` for CI/CD pipeline.

---
For details, see code comments and the UI pages. This README is intentionally minimal and up-to-date with the current codebase.
