from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import requests
import os

app = FastAPI()

TFSERVING_URL = os.environ.get("TFSERVING_URL", "http://localhost:8501/v1/models/my_model:predict")
VALID_API_KEYS = {"my-secret-key"}  # Replace with your actual key(s) or load securely

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
