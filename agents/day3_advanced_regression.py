import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


# ═════════════════════════════════════════════════════════════
# 1. Generate nonlinear synthetic data
# ═════════════════════════════════════════════════════════════
np.random.seed(42)
n = 200

x = np.linspace(-3, 3, n)
# True relationship: cubic + noise
y_true = 0.5 * x**3 - 1.2 * x**2 + 0.8 * x + 2
y = y_true + np.random.normal(0, 1.5, n)

X = x.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


# ═════════════════════════════════════════════════════════════
# 2. Baseline: plain linear regression on nonlinear data
# ═════════════════════════════════════════════════════════════
lin = LinearRegression().fit(X_train, y_train)

print('─' * 60)
print('1) Linear regression on nonlinear data')
print('─' * 60)
print(f'  Train R² = {lin.score(X_train, y_train):.4f}')
print(f'  Test  R² = {lin.score(X_test,  y_test):.4f}')
print('  → Underfits. The model is too simple for the data.\n')


# ═════════════════════════════════════════════════════════════
# 3. Degree sweep — see overfitting appear
# ═════════════════════════════════════════════════════════════
print('─' * 60)
print('2) Polynomial degree sweep (no regularization)')
print('─' * 60)

degrees = range(1, 16)
train_r2, test_r2 = [], []

for d in degrees:
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=d, include_bias=False)),
        ('lin',  LinearRegression()),
    ])
    pipe.fit(X_train, y_train)
    train_r2.append(pipe.score(X_train, y_train))
    test_r2.append(pipe.score(X_test,  y_test))

sweep = pd.DataFrame({'degree': list(degrees),
                      'train_r2': train_r2,
                      'test_r2':  test_r2})
print(sweep.round(4).to_string(index=False))
print('\n  → Low degree: underfit. High degree: test R² drops = overfit.\n')


# ═════════════════════════════════════════════════════════════
# 4. Ridge / Lasso / ElasticNet with cross-validation
# ═════════════════════════════════════════════════════════════
print('─' * 60)
print('3) Regularized models (degree 10 polynomial)')
print('─' * 60)

DEGREE = 10
cv = KFold(n_splits=5, shuffle=True, random_state=42)

def make_pipe(model):
    return Pipeline([
        ('poly',   PolynomialFeatures(degree=DEGREE, include_bias=False)),
        ('scaler', StandardScaler()),
        ('model',  model),
    ])

models = {
    'Linear (no reg)': make_pipe(LinearRegression()),
    'Ridge  α=1':      make_pipe(Ridge(alpha=1.0)),
    'Ridge  α=10':     make_pipe(Ridge(alpha=10.0)),
    'Lasso  α=0.01':   make_pipe(Lasso(alpha=0.01, max_iter=10000)),
    'Lasso  α=0.1':    make_pipe(Lasso(alpha=0.1,  max_iter=10000)),
    'ElasticNet':      make_pipe(ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)),
}

results = []
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    yp = pipe.predict(X_test)

    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='r2')

    results.append({
        'model':      name,
        'train_r2':   pipe.score(X_train, y_train),
        'test_r2':    pipe.score(X_test,  y_test),
        'test_rmse':  np.sqrt(mean_squared_error(y_test, yp)),
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std':  cv_scores.std(),
    })

results_df = pd.DataFrame(results)
print(results_df.round(4).to_string(index=False))
print()


# ═════════════════════════════════════════════════════════════
# 5. Ridge / Lasso paths (how coefficients shrink with α)
# ═════════════════════════════════════════════════════════════
alphas = np.logspace(-4, 3, 100)

poly = PolynomialFeatures(degree=DEGREE, include_bias=False)
Xtr_poly = StandardScaler().fit_transform(poly.fit_transform(X_train))

ridge_path, lasso_path = [], []
for a in alphas:
    ridge_path.append(Ridge(alpha=a).fit(Xtr_poly, y_train).coef_)
    lasso_path.append(Lasso(alpha=a, max_iter=10000).fit(Xtr_poly, y_train).coef_)
ridge_path = np.array(ridge_path)
lasso_path = np.array(lasso_path)


# ═════════════════════════════════════════════════════════════
# 6. Visualizations
# ═════════════════════════════════════════════════════════════

# ── 6.1 Underfit vs good fit vs overfit ─────────────────────
x_plot = np.linspace(x.min(), x.max(), 500).reshape(-1, 1)

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for ax, deg, title in zip(
    axes, [1, 3, 14],
    ['Underfit (degree 1)', 'Good fit (degree 3)', 'Overfit (degree 14)']
):
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
        ('lin',  LinearRegression()),
    ]).fit(X_train, y_train)

    ax.scatter(X_train, y_train, alpha=0.5, label='train')
    ax.scatter(X_test,  y_test,  alpha=0.5, label='test')
    ax.plot(x_plot, pipe.predict(x_plot), 'r-', linewidth=2, label='model')
    ax.plot(x, y_true, 'k--', linewidth=1, alpha=0.5, label='true')
    ax.set_title(f'{title}\ntrain R²={pipe.score(X_train, y_train):.3f}  '
                 f'test R²={pipe.score(X_test, y_test):.3f}')
    ax.set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].legend()
plt.tight_layout()
plt.show()

# ── 6.2 Degree sweep: train vs test R² ──────────────────────
fig, ax = plt.subplots()
ax.plot(sweep['degree'], sweep['train_r2'], 'o-', label='Train R²')
ax.plot(sweep['degree'], sweep['test_r2'],  's-', label='Test R²')
ax.axhline(1, color='gray', linestyle=':', linewidth=1)
best_deg = sweep.loc[sweep['test_r2'].idxmax(), 'degree']
ax.axvline(best_deg, color='red', linestyle='--', alpha=0.5,
           label=f'best degree = {best_deg}')
ax.set_xlabel('Polynomial degree')
ax.set_ylabel('R²')
ax.set_title('Underfitting → Overfitting as Degree Grows')
ax.legend()
plt.tight_layout()
plt.show()

# ── 6.3 Ridge path ──────────────────────────────────────────
fig, ax = plt.subplots()
for j in range(ridge_path.shape[1]):
    ax.plot(alphas, ridge_path[:, j], linewidth=1)
ax.set_xscale('log')
ax.axhline(0, color='k', linewidth=0.5)
ax.set_xlabel('alpha (log scale)')
ax.set_ylabel('coefficient value')
ax.set_title('Ridge Path — coefficients shrink smoothly, never reach 0')
plt.tight_layout()
plt.show()

# ── 6.4 Lasso path ──────────────────────────────────────────
fig, ax = plt.subplots()
for j in range(lasso_path.shape[1]):
    ax.plot(alphas, lasso_path[:, j], linewidth=1)
ax.set_xscale('log')
ax.axhline(0, color='k', linewidth=0.5)
ax.set_xlabel('alpha (log scale)')
ax.set_ylabel('coefficient value')
ax.set_title('Lasso Path — coefficients hit zero (feature selection)')
plt.tight_layout()
plt.show()

# ── 6.5 Test R² vs alpha for Ridge and Lasso ────────────────
ridge_test, lasso_test = [], []
for a in alphas:
    r = Pipeline([('poly', PolynomialFeatures(DEGREE, include_bias=False)),
                  ('sc',   StandardScaler()),
                  ('m',    Ridge(alpha=a))]).fit(X_train, y_train)
    l = Pipeline([('poly', PolynomialFeatures(DEGREE, include_bias=False)),
                  ('sc',   StandardScaler()),
                  ('m',    Lasso(alpha=a, max_iter=10000))]).fit(X_train, y_train)
    ridge_test.append(r.score(X_test, y_test))
    lasso_test.append(l.score(X_test, y_test))

fig, ax = plt.subplots()
ax.plot(alphas, ridge_test, label='Ridge')
ax.plot(alphas, lasso_test, label='Lasso')
ax.set_xscale('log')
ax.set_xlabel('alpha (log scale)')
ax.set_ylabel('Test R²')
ax.set_title('Test R² vs Regularization Strength')
ax.legend()
plt.tight_layout()
plt.show()

# ── 6.6 Coefficient shrinkage summary ───────────────────────
strong_ridge = Ridge(alpha=100).fit(Xtr_poly, y_train).coef_
strong_lasso = Lasso(alpha=0.1, max_iter=10000).fit(Xtr_poly, y_train).coef_
plain_coefs  = LinearRegression().fit(Xtr_poly, y_train).coef_

n_zero_lasso = int(np.sum(np.isclose(strong_lasso, 0)))
print('─' * 60)
print('4) Feature-selection check (degree 10 polynomial)')
print('─' * 60)
print(f'  Polynomial features total : {Xtr_poly.shape[1]}')
print(f'  Plain LR non-zero coefs   : {np.sum(~np.isclose(plain_coefs, 0))}')
print(f'  Ridge α=100 non-zero      : {np.sum(~np.isclose(strong_ridge, 0))}')
print(f'  Lasso α=0.1 non-zero      : {np.sum(~np.isclose(strong_lasso, 0))}  '
      f'(zeroed: {n_zero_lasso})')
print('  → Lasso eliminated unhelpful features automatically.\n')


# ═════════════════════════════════════════════════════════════
# 7. Interpretation summary
# ═════════════════════════════════════════════════════════════
print('─' * 60)
print('SUMMARY')
print('─' * 60)
print('• Linear regression underfits nonlinear data.')
print('• Polynomial features let a linear model capture curves.')
print('• Too-high degree → overfit: train R² ↑, test R² ↓.')
print('• Ridge (L2): shrinks coefs smoothly, keeps all features.')
print('• Lasso (L1): drives some coefs to exactly 0 → feature selection.')
print('• ElasticNet: combines L1 and L2 (l1_ratio controls the mix).')
print('• Always StandardScaler before regularization — otherwise the')
print('  penalty is applied unequally across features.')
print('• Use cross-validation to pick alpha, not the test set.')