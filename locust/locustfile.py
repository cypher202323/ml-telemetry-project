from locust import HttpUser, task, between

# Sample house data (the same example used in testing)
HOUSE = {
    "MedInc": 8.3252,
    "HouseAge": 41,
    "AveRooms": 6.984,
    "AveBedrms": 1.024,
    "Population": 322,
    "AveOccup": 2.556,
    "Latitude": 37.88,
    "Longitude": -122.23,
}


class PredictionUser(HttpUser):
    # Each simulated user waits 1 to 2 seconds between requests
    wait_time = between(1, 2)

    @task
    def predict(self):
        self.client.post("/predict", json=HOUSE)