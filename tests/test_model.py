import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def test_model_training():
    # Generate random data
    X = np.random.rand(100, 1)
    y = 3 * X.squeeze() + 2 + np.random.randn(100)

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Evaluate model
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)

    # Assert that MSE is within an acceptable range
    assert mse < 1.0, f"Model training failed with high MSE: {mse}"
