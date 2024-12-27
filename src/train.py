import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pickle
import os

# Cấu hình
MODEL_PATH = "models/linear_model.pkl"

# Tạo dữ liệu ngẫu nhiên
X = np.random.rand(100, 1)  # 100 mẫu, mỗi mẫu có 1 đặc trưng
y = 3 * X.squeeze() + 2 + np.random.randn(100)  # y = 3x + 2 + noise

# Chia dữ liệu thành tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán và đánh giá mô hình
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# In kết quả
print(f"Mean Squared Error: {mse:.4f}")

# Lưu mô hình
os.makedirs("models", exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)
print(f"Model saved to {MODEL_PATH}")
