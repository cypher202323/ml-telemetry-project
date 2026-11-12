from fastapi import FastAPI, Response
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI()

# A metric: counts how many times the main endpoint is called
REQUESTS = Counter("prototype_requests_total", "Number of requests to the root endpoint")

@app.get("/")
def root():
    REQUESTS.inc()                      # add 1 to the counter
    return {"message": "Prototype is running"}

@app.get("/metrics")
def metrics():
    # Prometheus reads this page to collect the metrics
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)