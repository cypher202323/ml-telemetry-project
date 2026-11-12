"""FastAPI service that serves house price predictions and exposes metrics for Prometheus."""
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "model.joblib"
KNOWN_ENDPOINTS = {"/predict", "/health"}

# ---- Metrics (Table 3.5) ----
REQUEST_COUNT = Counter(
    "api_requests_total", "Total number of requests", ["method", "endpoint", "status_code"]
)
REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds", "Time taken to process a request", ["endpoint"]
)
REQUEST_ERRORS = Counter(
    "api_request_errors_total", "Number of requests that failed or were rejected", ["endpoint", "status_code"]
)


# ---- Input format: the 8 features, with validation rules (FR2) ----
class HouseFeatures(BaseModel):
    MedInc: float = Field(..., gt=0, description="Median income in the district (x $10,000)")
    HouseAge: float = Field(..., ge=0, description="Median house age in years")
    AveRooms: float = Field(..., gt=0, description="Average rooms per household")
    AveBedrms: float = Field(..., gt=0, description="Average bedrooms per household")
    Population: float = Field(..., ge=0, description="District population")
    AveOccup: float = Field(..., gt=0, description="Average household members")
    Latitude: float = Field(..., ge=32, le=42, description="Latitude (California)")
    Longitude: float = Field(..., ge=-125, le=-114, description="Longitude (California)")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "MedInc": 8.3252, "HouseAge": 41, "AveRooms": 6.984, "AveBedrms": 1.024,
                "Population": 322, "AveOccup": 2.556, "Latitude": 37.88, "Longitude": -122.23,
            }]
        }
    }


FEATURE_ORDER = list(HouseFeatures.model_fields)  # same order the model was trained on
model = None


# ---- Load the model ONCE when the service starts ----
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(title="House Price Prediction API", lifespan=lifespan)


# ---- Record metrics for every request (except /metrics itself) ----
@app.middleware("http")
async def record_metrics(request: Request, call_next):
    if request.url.path == "/metrics":
        return await call_next(request)

    endpoint = request.url.path if request.url.path in KNOWN_ENDPOINTS else "other"
    start = time.perf_counter()
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        status = 500
        raise
    finally:
        REQUEST_LATENCY.labels(endpoint).observe(time.perf_counter() - start)
        REQUEST_COUNT.labels(request.method, endpoint, str(status)).inc()
        if status >= 400:
            REQUEST_ERRORS.labels(endpoint, str(status)).inc()
    return response


# ---- Endpoints ----
@app.post("/predict")
def predict(features: HouseFeatures):
    row = [[getattr(features, name) for name in FEATURE_ORDER]]
    value = float(model.predict(row)[0])
    return {
        "predicted_median_house_value": round(value, 4),
        "unit": "x $100,000",
        "estimated_value_usd": round(value * 100_000, 2),
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)