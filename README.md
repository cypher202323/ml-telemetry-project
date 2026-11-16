# Containerized Machine Learning Deployment Architecture with Real-Time Telemetry

Final year project, B.Sc. Information Technology, Bayero University Kano.
Author: Imam Malik Sani

This project deploys a machine learning model as a containerized web service and
monitors it in real time. A house-price prediction model is served through a
FastAPI application. Prometheus collects its request metrics and the containers'
CPU and memory usage (through cAdvisor), Grafana displays them on a live
dashboard, and Locust is used to load-test the system.

## System components

| Container  | Purpose                                         | Address                 |
|------------|-------------------------------------------------|-------------------------|
| api        | FastAPI service that serves the ML model        | http://localhost:8000   |
| prometheus | Collects and stores metrics every 5 seconds     | http://localhost:9090   |
| cadvisor   | Reports CPU and memory usage of each container  | http://localhost:8080   |
| grafana    | Real-time monitoring dashboard                  | http://localhost:3000   |
| locust     | Load testing tool                               | http://localhost:8089   |

## Folder structure

```
ml-telemetry-project/
├── app/                  # The ML API
│   ├── main.py           # FastAPI app: /predict, /health, /metrics
│   ├── train_model.py    # Trains the model and saves model.joblib
│   ├── model.joblib      # The trained model
│   ├── requirements.txt  # Python libraries for the API
│   └── Dockerfile        # Builds the API container image
├── prometheus/
│   └── prometheus.yml    # Scrape targets and interval
├── grafana/
│   ├── provisioning/     # Auto-configures the data source and dashboard
│   └── dashboards/
│       └── ml-telemetry.json   # The dashboard (5 panels)
├── locust/
│   └── locustfile.py     # Simulated users for load testing
├── tests/
│   └── test_main.py      # Unit tests (pytest)
├── prototype/            # Early prototype used during requirements gathering
└── docker-compose.yml    # Starts all five containers together
```

## Requirements

- Docker Desktop (with Docker Compose)
- Python 3.13 (only needed to run the unit tests or retrain the model)

## How to run

From the project folder:

```
docker compose up -d
```

Wait about 30 seconds, then check that all five containers are running:

```
docker compose ps
```

Then open:

- Dashboard: http://localhost:3000 → Dashboards → ML Telemetry Dashboard
  (login admin / admin; viewing is also allowed without logging in)
- Interactive API documentation: http://localhost:8000/docs
- Prometheus targets: http://localhost:9090/targets

To stop everything:

```
docker compose stop
```

## Using the API

| Method | Endpoint  | Description                                   |
|--------|-----------|-----------------------------------------------|
| POST   | /predict  | Returns a predicted median house value        |
| GET    | /health   | Shows whether the service and model are ready |
| GET    | /metrics  | Metrics in Prometheus format                  |

Example request:

```
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"MedInc":8.3252,"HouseAge":41,"AveRooms":6.984,"AveBedrms":1.024,"Population":322,"AveOccup":2.556,"Latitude":37.88,"Longitude":-122.23}'
```

Example response:

```
{"predicted_median_house_value":4.1521,"unit":"x $100,000","estimated_value_usd":415210.59}
```

Invalid input (for example, a latitude outside California) is rejected with HTTP 422.

## The model

Linear regression (scikit-learn) trained on the California Housing dataset,
with an 80/20 train/test split. Test results: R² = 0.576, MAE = 0.533
(about $53,300). The model is deliberately simple because the project focuses
on the deployment and monitoring architecture, not on model accuracy.

## Running the unit tests

```
python -m venv .venv
source .venv/Scripts/activate      # Git Bash on Windows
pip install -r app/requirements.txt pytest httpx
pytest -v
```

## Load testing

With the system running, open http://localhost:8089, enter the number of users
and the ramp-up rate, and click Start. Watch the effect on the Grafana dashboard.