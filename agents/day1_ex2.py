import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

np.random.seed(43)
n_samples = 100
X = 10 * np.random.rand(n_samples, 1)
noise = np.random.randn(n_samples, 1)
Y = 3 * X + 5 + noise

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=43)

model = LinearRegression()
model.fit(X_train, Y_train)

Y_pred_train = model.predict(X_train)
Y_pred_test = model.predict(X_test)

print(f"Slope: {model.coef_[0][0]:.4f} | Intercept: {model.intercept_[0]:.4f}")
print(f"Train R²: {r2_score(Y_train, Y_pred_train):.4f} | Test R²: {r2_score(Y_test, Y_pred_test):.4f}")
print(f"Test MSE: {mean_squared_error(Y_test, Y_pred_test):.4f}")
print(f"Test MAE: {mean_absolute_error(Y_test, Y_pred_test):.4f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(X_train, Y_train, color='blue', alpha=0.5, label='Train')
axes[0].scatter(X_test, Y_test, color='green', alpha=0.5, label='Test')
axes[0].plot(X_test, Y_pred_test, color='red', linewidth=2, label='Fit')
axes[0].set_title('Model Fit')
axes[0].legend()

residuals = Y_test - Y_pred_test
axes[1].scatter(Y_pred_test, residuals, color='purple', alpha=0.6)
axes[1].axhline(y=0, color='black', linestyle='--')
axes[1].set_xlabel('Predicted Values')
axes[1].set_ylabel('Residuals')
axes[1].set_title('Residual Plot')

axes[2].hist(residuals, bins=15, color='orange', edgecolor='black', alpha=0.7)
axes[2].set_title('Residual Distribution')
axes[2].set_xlabel('Error Value')

plt.tight_layout()
plt.show()