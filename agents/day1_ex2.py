import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


# ─────────────────────────────────────────────────────────────
# 1. Generate synthetic data
# ─────────────────────────────────────────────────────────────
np.random.seed(42)
n_samples = 500

X = pd.DataFrame({
    'size_sqm':    np.random.normal(120, 30, n_samples).round(1),
    'bedrooms':    np.random.randint(1, 6, n_samples),
    'age_years':   np.random.randint(0, 40, n_samples),
    'distance_km': np.random.uniform(1, 20, n_samples).round(2),
})

true_coefs = {
    'size_sqm':    2500,
    'bedrooms':    8000,
    'age_years':  -1500,
    'distance_km': -3000,
}
base_price = 50000

y = (
    base_price
    + X['size_sqm']    * true_coefs['size_sqm']
    + X['bedrooms']    * true_coefs['bedrooms']
    + X['age_years']   * true_coefs['age_years']
    + X['distance_km'] * true_coefs['distance_km']
    + np.random.normal(0, 15000, n_samples)
).round(2)


# ─────────────────────────────────────────────────────────────
# 2. Train / test split
# ─────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ─────────────────────────────────────────────────────────────
# 3. Fit the model
# ─────────────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)


# ─────────────────────────────────────────────────────────────
# 4. Predictions and metrics
# ─────────────────────────────────────────────────────────────
y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

def metrics(y_true, y_pred, label):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f'{label}:')
    print(f'  MSE  : {mse:,.2f}')
    print(f'  RMSE : {rmse:,.2f}')
    print(f'  MAE  : {mae:,.2f}')
    print(f'  R²   : {r2:.4f}\n')
    return dict(mse=mse, rmse=rmse, mae=mae, r2=r2)

m_train = metrics(y_train, y_pred_train, 'Train')
m_test  = metrics(y_test,  y_pred_test,  'Test')


# ─────────────────────────────────────────────────────────────
# 5. Cross-validation (5-fold, R²)
# ─────────────────────────────────────────────────────────────
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2   = cross_val_score(model, X, y, cv=cv, scoring='r2')
cv_rmse = -cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')

print('5-fold cross-validation:')
print(f'  R²   mean={cv_r2.mean():.4f}  std={cv_r2.std():.4f}')
print(f'  RMSE mean={cv_rmse.mean():,.2f}  std={cv_rmse.std():,.2f}\n')


# ─────────────────────────────────────────────────────────────
# 6. Visualizations
# ─────────────────────────────────────────────────────────────

# ── 6.1 Actual vs Predicted ─────────────────────────────────
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred_test, alpha=0.6, edgecolor='k', linewidth=0.3)
lims = [min(y_test.min(), y_pred_test.min()),
        max(y_test.max(), y_pred_test.max())]
ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
ax.set_xlabel('Actual price')
ax.set_ylabel('Predicted price')
ax.set_title(f'Actual vs Predicted (Test)\nR² = {m_test["r2"]:.4f}')
ax.legend()
plt.tight_layout()
plt.show()

# ── 6.2 Residuals vs Predicted ──────────────────────────────
residuals = y_test - y_pred_test
fig, ax = plt.subplots()
ax.scatter(y_pred_test, residuals, alpha=0.6, edgecolor='k', linewidth=0.3)
ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
ax.set_xlabel('Predicted price')
ax.set_ylabel('Residual (actual − predicted)')
ax.set_title('Residuals vs Predicted (Test)')
plt.tight_layout()
plt.show()

# ── 6.3 Residual distribution ───────────────────────────────
fig, ax = plt.subplots()
sns.histplot(residuals, kde=True, ax=ax, color='steelblue')
ax.axvline(0, color='red', linestyle='--')
ax.set_xlabel('Residual')
ax.set_title(f'Residual Distribution (Test)\nmean={residuals.mean():,.0f}  std={residuals.std():,.0f}')
plt.tight_layout()
plt.show()

# ── 6.4 Coefficients: learned vs true ───────────────────────
coef_df = pd.DataFrame({
    'feature':   X.columns,
    'true':      list(true_coefs.values()),
    'learned':   model.coef_,
})
coef_df = coef_df.melt(id_vars='feature', var_name='kind', value_name='value')

fig, ax = plt.subplots()
sns.barplot(data=coef_df, x='feature', y='value', hue='kind', ax=ax)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_title('Coefficients: Learned vs True')
ax.set_ylabel('Coefficient value')
plt.tight_layout()
plt.show()

# ── 6.5 Feature correlation heatmap ─────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(X.assign(price=y).corr(), annot=True, cmap='coolwarm',
            center=0, fmt='.2f', ax=ax)
ax.set_title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# ── 6.6 Cross-validation scores ─────────────────────────────
fig, ax = plt.subplots()
ax.bar(range(1, len(cv_r2) + 1), cv_r2, color='seagreen', alpha=0.8)
ax.axhline(cv_r2.mean(), color='red', linestyle='--',
           label=f'mean = {cv_r2.mean():.4f}')
ax.set_xticks(range(1, len(cv_r2) + 1))
ax.set_xlabel('Fold')
ax.set_ylabel('R²')
ax.set_title('5-Fold Cross-Validation R²')
ax.set_ylim(0, 1.05)
ax.legend()
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────
# 7. Interpretation summary
# ─────────────────────────────────────────────────────────────
print('─' * 60)
print('INTERPRETATION')
print('─' * 60)

print(f'Train R²  = {m_train["r2"]:.4f}')
print(f'Test  R²  = {m_test["r2"]:.4f}')
gap = m_train['r2'] - m_test['r2']
print(f'Gap       = {gap:+.4f}  '
      f'({"underfitting" if gap < -0.02 else "overfitting" if gap > 0.05 else "healthy"})')
print()

print('Coefficients (learned vs true):')
for f, t, l in zip(X.columns, true_coefs.values(), model.coef_):
    err = abs(l - t) / abs(t) * 100 if t else 0
    print(f'  {f:<12}  true={t:>8,}  learned={l:>10,.2f}  error={err:5.2f}%')

print(f'\nIntercept: true={base_price:,}  learned={model.intercept_:,.2f}')

print('\nResidual diagnostics:')
print(f'  mean      = {residuals.mean():>10,.2f}  (should be ≈ 0)')
print(f'  std       = {residuals.std():>10,.2f}')
print(f'  skewness  = {residuals.skew():>10.3f}  (should be ≈ 0)')
print(f'  max |res| = {residuals.abs().max():>10,.2f}')