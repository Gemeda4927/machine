import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 1. Generate Higher-Order Data
np.random.seed(43)
n_samples = 100
X = 10 * np.random.rand(n_samples, 1) - 5
noise = 10 * np.random.randn(n_samples, 1)
Y = 0.5 * (X ** 2) + 2 * X + 10 + noise

# 2. Split Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=43)

# 3. Transform Features
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# 4. Fit the Model
model = LinearRegression()
model.fit(X_train_poly, Y_train)

# 5. Evaluate
Y_pred = model.predict(X_test_poly)
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)

print(f"R-squared: {r2:.4f}")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")

# --- Visualizations ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: The Curve Fit
X_sorted = np.sort(X, axis=0)
X_sorted_poly = poly.transform(X_sorted)
Y_sorted_pred = model.predict(X_sorted_poly)

axes[0].scatter(X, Y, color='blue', alpha=0.6, label='Actual Data')
axes[0].plot(X_sorted, Y_sorted_pred, color='red', linewidth=2, label='Model Prediction')
axes[0].set_title('Polynomial Regression Fit')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].legend()

# Plot 2: Residuals (Errors)
residuals = Y_test - Y_pred
axes[1].scatter(Y_pred, residuals, color='green', alpha=0.6)
axes[1].axhline(y=0, color='black', linestyle='--')
axes[1].set_title('Residual Plot (Error Distribution)')
axes[1].set_xlabel('Predicted Values')
axes[1].set_ylabel('Residuals (Actual - Predicted)')

plt.tight_layout()
plt.show()