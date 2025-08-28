from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
import requests
import os
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi import Query
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

TFSERVING_URL = os.environ.get("TFSERVING_URL", "http://localhost:8501/v1/models/my_model:predict")
api_keys_env = os.environ.get("API_KEYS", "my-secret-key")
VALID_API_KEYS = set(k.strip() for k in api_keys_env.split(",") if k.strip())

Instrumentator().instrument(app).expose(app)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")

@app.post("/predict")
async def predict(request: Request):
    api_key = request.headers.get("x-api-key")
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    body = await request.body()
    headers = {"content-type": "application/json"}
    tf_response = requests.post(TFSERVING_URL, data=body, headers=headers)
    try:
        return JSONResponse(status_code=tf_response.status_code, content=tf_response.json())
    except Exception:
        return JSONResponse(status_code=500, content={"error": "Invalid response from TensorFlow Serving"})

@app.post("/canary")
async def canary_test(request: Request, version_a: int = Query(...), version_b: int = Query(...)):
    api_key = request.headers.get("x-api-key")
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    body = await request.body()
    headers = {"content-type": "application/json"}
    # Query both versions
    url_a = f"http://localhost:8501/v1/models/my_model/versions/{version_a}:predict"
    url_b = f"http://localhost:8501/v1/models/my_model/versions/{version_b}:predict"
    resp_a = requests.post(url_a, data=body, headers=headers)
    resp_b = requests.post(url_b, data=body, headers=headers)
    try:
        preds_a = resp_a.json().get("predictions", [])
        preds_b = resp_b.json().get("predictions", [])
    except Exception:
        return JSONResponse(status_code=500, content={"error": "Invalid response from TensorFlow Serving"})
    return {"version_a": preds_a, "version_b": preds_b}
