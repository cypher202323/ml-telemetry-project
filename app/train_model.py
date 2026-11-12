"""Train a simple regression model on the California Housing dataset and save it."""
from pathlib import Path

import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# The model file is saved in the same folder as this script (app/)
MODEL_PATH = Path(__file__).parent / "model.joblib"


def main():
    # 1. Load the data: 20,640 houses, 8 features each
    data = fetch_california_housing()
    X, y = data.data, data.target   # y = median house value, in units of $100,000
    print("Features:", data.feature_names)

    # 2. Keep 20% of the houses aside to test the model fairly
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 4. Check how well it predicts houses it has never seen
    predictions = model.predict(X_test)
    print(f"R2 score: {r2_score(y_test, predictions):.3f}")
    print(f"Mean absolute error: {mean_absolute_error(y_test, predictions):.3f} (x $100,000)")

    # 5. Save the trained model to a file
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()