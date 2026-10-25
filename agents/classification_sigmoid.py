import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Generate Binary Classification Data
np.random.seed(43)
n_samples = 200

# Class 0: Centered around (-2, -2)
X_class0 = np.random.randn(n_samples // 2, 2) - 2
# Class 1: Centered around (2, 2)
X_class1 = np.random.randn(n_samples // 2, 2) + 2

X = np.vstack((X_class0, X_class1))
Y = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2)))

# 2. Split and Scale
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=43)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Train Logistic Regression (Uses Sigmoid internally)
model = LogisticRegression()
model.fit(X_train_scaled, Y_train)

# 4. Predictions
Y_pred = model.predict(X_test_scaled)
Y_prob = model.predict_proba(X_test_scaled)[:, 1] # Probability of Class 1

# 5. Evaluation
print(f"Accuracy: {accuracy_score(Y_test, Y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(Y_test, Y_pred))

# 6. Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Decision Boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

axes[0].contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
axes[0].scatter(X_test[:, 0], X_test[:, 1], c=Y_test, edgecolors='k', cmap='RdBu')
axes[0].set_title('Decision Boundary')

# Plot 2: Sigmoid Probabilities
axes[1].scatter(Y_prob, Y_test, alpha=0.6, color='blue')
axes[1].axvline(x=0.5, color='red', linestyle='--', label='Threshold (0.5)')
axes[1].set_xlabel('Predicted Probability (Sigmoid Output)')
axes[1].set_ylabel('Actual Class')
axes[1].set_title('Sigmoid Output vs Actuals')
axes[1].legend()

plt.tight_layout()
plt.show()