import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 1. The Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 2. Generate Data: Age vs Salary -> Purchase
np.random.seed(43)
n_samples = 200
ages = np.random.randint(20, 60, n_samples)
salaries = np.random.randint(20000, 150000, n_samples)

# Create labels based on a logical rule + noise
z_true = 0.0001 * salaries + 0.1 * ages - 5
probs_true = sigmoid(z_true)
noise = np.random.rand(n_samples)
purchases = ((probs_true + (noise - 0.5) * 0.2) > 0.5).astype(int)

df = pd.DataFrame({'Age': ages, 'Salary': salaries, 'Purchased': purchases})

# 3. Prepare Data
X = df[['Age', 'Salary']].values
Y = df['Purchased'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=43)

# Scale features (Crucial for gradient descent)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train Logistic Regression (Manual Gradient Descent)
learning_rate = 0.1
epochs = 1000
m, n = X_train_scaled.shape
weights = np.zeros(n)
bias = 0

for i in range(epochs):
    # Forward pass
    z = np.dot(X_train_scaled, weights) + bias
    predictions = sigmoid(z)
    
    # Calculate gradients
    error = predictions - Y_train
    dw = (1/m) * np.dot(X_train_scaled.T, error)
    db = (1/m) * np.sum(error)
    
    # Update weights
    weights -= learning_rate * dw
    bias -= learning_rate * db

# 5. Evaluate
def predict(X, weights, bias):
    z = np.dot(X, weights) + bias
    probs = sigmoid(z)
    return (probs >= 0.5).astype(int)

Y_pred = predict(X_test_scaled, weights, bias)

print(f"Accuracy: {accuracy_score(Y_test, Y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(Y_test, Y_pred))

# 6. Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Actual Data
scatter = axes[0].scatter(df['Age'], df['Salary'], c=df['Purchased'], cmap='RdBu', alpha=0.7)
axes[0].set_title('Actual Purchases')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Salary')
axes[0].legend(*scatter.legend_elements(), title="Purchased")

# Plot 2: Decision Boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5), np.arange(y_min, y_max, 500))

# Transform grid points to scaled values to use our manual weights
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_scaled = scaler.transform(grid_points)
Z = predict(grid_scaled, weights, bias)
Z = Z.reshape(xx.shape)

axes[1].contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
axes[1].scatter(X_test[:, 0], X_test[:, 1], c=Y_test, edgecolors='k', cmap='RdBu')
axes[1].set_title('Model Decision Boundary (From Scratch)')
axes[1].set_xlabel('Age')
axes[1].set_ylabel('Salary')

plt.tight_layout()
plt.show()