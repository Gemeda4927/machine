import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

np.random.seed(43)
n_samples = 30
X = 6 * np.random.rand(n_samples, 1) - 3
Y = 0.5 * X**3 + X**2 - 2*X + 5 + 10 * np.random.randn(n_samples, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=43)

degree = 10

models = {
    "Linear Regression": make_pipeline(PolynomialFeatures(degree), LinearRegression()),
    "Ridge (L2)": make_pipeline(PolynomialFeatures(degree), StandardScaler(), Ridge(alpha=1.0)),
    "Lasso (L1)": make_pipeline(PolynomialFeatures(degree), StandardScaler(), Lasso(alpha=0.1))
}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
X_sorted = np.linspace(-3, 3, 200).reshape(-1, 1)

for ax, (name, model) in zip(axes, models.items()):
    model.fit(X_train, Y_train)
    
    Y_pred_plot = model.predict(X_sorted)
    Y_pred_test = model.predict(X_test)
    
    r2 = r2_score(Y_test, Y_pred_test)
    
    ax.scatter(X_train, Y_train, color='blue', label='Train Data')
    ax.scatter(X_test, Y_test, color='green', label='Test Data')
    ax.plot(X_sorted, Y_pred_plot, color='red', linewidth=2, label=f'{name} Fit')
    ax.set_title(f'{name}\nTest R²: {r2:.3f}')
    ax.set_ylim(-20, 20)
    ax.legend()

plt.tight_layout()
plt.show()

print("\n--- Model Complexity (Sum of Absolute Coefficients) ---")
for name, model in models.items():
    lr_step = model.steps[-1][1] 
    coef_sum = np.sum(np.abs(lr_step.coef_))
    print(f"{name}: {coef_sum:.2f}")