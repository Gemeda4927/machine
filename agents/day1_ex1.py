"""
day1_ex1.py

Load the Seaborn "tips" dataset, split into features and target,
one-hot encode the categorical column, split into train/test sets,
visualize the data, and train + evaluate regression models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42

sns.set_theme(style="whitegrid")

df = pd.read_csv(URL)

# ==================================================================
# SECTION 1 — Raw data overview
# ==================================================================
print("=" * 60)
print("SECTION 1 — Raw data overview")
print("=" * 60)

print("\nFirst 5 rows:\n", df.head())
print("\nShape:", df.shape)
print("\nColumn types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nDescriptive stats:\n", df.describe(include="all"))
print("\nUnique values per categorical column:")
for col in df.select_dtypes(include="object").columns:
    print(f"  {col}: {df[col].unique()}")

# ==================================================================
# SECTION 2 — Feature engineering
# ==================================================================
print("\n" + "=" * 60)
print("SECTION 2 — Feature engineering")
print("=" * 60)

df["tip_pct"] = df["tip"] / df["total_bill"] * 100          # tip as % of bill
df["bill_per_person"] = df["total_bill"] / df["size"]        # per-head cost
df["is_weekend"] = df["day"].isin(["Sat", "Sun"]).astype(int)
df["is_dinner"] = (df["time"] == "Dinner").astype(int)
df["is_smoker"] = (df["smoker"] == "Yes").astype(int)

print("\nNew columns added:")
print(df[["total_bill", "tip", "tip_pct", "bill_per_person", "is_weekend", "is_dinner"]].head())

# ==================================================================
# SECTION 3 — Full-feature feature set (richer than sex only)
# ==================================================================
print("\n" + "=" * 60)
print("SECTION 3 — Build full feature set")
print("=" * 60)

full_features = pd.get_dummies(
    df[[
        "total_bill", "size",
        "sex", "smoker", "day", "time",
        "bill_per_person", "is_weekend", "is_dinner",
    ]],
    columns=["sex", "smoker", "day", "time"],
    drop_first=True,   # avoid dummy-variable trap
)

target = df["tip"]

print("Feature matrix shape:", full_features.shape)
print("Feature columns:", list(full_features.columns))

# ==================================================================
# SECTION 4 — Train/test split
# ==================================================================
print("\n" + "=" * 60)
print("SECTION 4 — Train/test split")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    full_features,
    target,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

print(f"X_train: {X_train.shape}   X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}   y_test: {y_test.shape}")

# ==================================================================
# SECTION 5 — Scaling
# ==================================================================
print("\n" + "=" * 60)
print("SECTION 5 — Standard scaling")
print("=" * 60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaled train mean (should be ~0):", np.round(X_train_scaled.mean(axis=0), 3))
print("Scaled train std  (should be ~1):", np.round(X_train_scaled.std(axis=0), 3))

# ==================================================================
# SECTION 6 — Train multiple regression models
# ==================================================================
print("\n" + "=" * 60)
print("SECTION 6 — Train multiple regression models")
print("=" * 60)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge (alpha=1)": Ridge(alpha=1.0),
    "Lasso (alpha=0.1)": Lasso(alpha=0.1, max_iter=10000),
    "Decision Tree": DecisionTreeRegressor(max_depth=4, random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=RANDOM_STATE),
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
    print(f"\n{name}")
    print(f"  MAE:  {mae:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  R²:   {r2:.3f}")

# ==================================================================
# SECTION 7 — Compare model performance
# ==================================================================
print("\n" + "=" * 60)
print("SECTION 7 — Model comparison")
print("=" * 60)

results_df = pd.DataFrame(results).T.sort_values("R2", ascending=False)
print("\n", results_df)

plt.figure(figsize=(9, 5))
sns.barplot(x=results_df.index, y=results_df["R2"], palette="viridis")
plt.title("Model Comparison — R² score (higher is better)")
plt.ylabel("R²")
plt.xticks(rotation=30, ha="right")
plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
plt.tight_layout()
plt.show()

# ==================================================================
# SECTION 8 — Actual vs Predicted (best model)
# ==================================================================
best_name = results_df.index[0]
best_model = models[best_name]
best_model.fit(X_train_scaled, y_train)
y_pred_best = best_model.predict(X_test_scaled)

print("\n" + "=" * 60)
print(f"SECTION 8 — Best model: {best_name}")
print("=" * 60)

plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred_best, alpha=0.6, color="teal", edgecolor="black")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color="red", linestyle="--", linewidth=1.5, label="Perfect prediction")
plt.xlabel("Actual Tip")
plt.ylabel("Predicted Tip")
plt.title(f"Actual vs Predicted — {best_name}")
plt.legend()
plt.tight_layout()
plt.show()

# ==================================================================
# SECTION 9 — Residual analysis
# ==================================================================
print("\n" + "=" * 60)
print("SECTION 9 — Residual analysis")
print("=" * 60)

residuals = y_test - y_pred_best

plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=25, kde=True, color="orange")
plt.axvline(0, color="black", linestyle="--", linewidth=1)
plt.title(f"Residuals — {best_name}")
plt.xlabel("Residual (actual − predicted)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(y_pred_best, residuals, alpha=0.6, color="steelblue", edgecolor="black")
plt.axhline(0, color="red", linestyle="--", linewidth=1.5)
plt.xlabel("Predicted Tip")
plt.ylabel("Residual")
plt.title("Residuals vs Predicted")
plt.tight_layout()
plt.show()

# ==================================================================
# SECTION 10 — Feature importance
# ==================================================================
print("\n" + "=" * 60)
print("SECTION 10 — Feature importance (tree-based model)")
print("=" * 60)

tree_model = models["Random Forest"]
tree_model.fit(X_train_scaled, y_train)

importances = pd.Series(
    tree_model.feature_importances_,
    index=full_features.columns,
).sort_values(ascending=False)

print("\n", importances)

plt.figure(figsize=(9, 5))
sns.barplot(x=importances.values, y=importances.index, palette="mako")
plt.title("Feature Importances — Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# ==================================================================
# SECTION 11 — Cross-validation
# ==================================================================
print("\n" + "=" * 60)
print("SECTION 11 — 5-fold cross-validation")
print("=" * 60)

for name in ["Linear Regression", "Random Forest", "Gradient Boosting"]:
    scores = cross_val_score(
        models[name], X_train_scaled, y_train,
        cv=5, scoring="r2",
    )
    print(f"{name}: mean R² = {scores.mean():.3f}  ± {scores.std():.3f}")
    print(f"  fold scores: {np.round(scores, 3)}")

# ==================================================================
# SECTION 12 — Extra visualizations
# ==================================================================
print("\n" + "=" * 60)
print("SECTION 12 — Extra visualizations")
print("=" * 60)

plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x="day", y="tip_pct", palette="Set3")
plt.title("Tip Percentage by Day")
plt.xlabel("Day")
plt.ylabel("Tip % of bill")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
sns.violinplot(data=df, x="time", y="tip_pct", hue="sex", split=True, palette="pastel")
plt.title("Tip Percentage by Time & Sex")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
sns.scatterplot(data=df, x="bill_per_person", y="tip", hue="size", palette="viridis", s=70)
plt.title("Tip vs Bill per Person (colored by party size)")
plt.xlabel("Bill per person")
plt.ylabel("Tip")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
sns.heatmap(
    df.select_dtypes(include="number").corr(),
    annot=True, fmt=".2f", cmap="coolwarm", center=0,
    square=True, cbar_kws={"shrink": 0.8},
)
plt.title("Correlation Heatmap (with engineered features)")
plt.tight_layout()
plt.show()

# ==================================================================
# SECTION 13 — Summary
# ==================================================================
print("\n" + "=" * 60)
print("SECTION 13 — Summary")
print("=" * 60)

print(f"\nBest model:     {best_name}")
print(f"R²  on test:    {results_df.loc[best_name, 'R2']:.3f}")
print(f"MAE on test:    {results_df.loc[best_name, 'MAE']:.3f}")
print(f"RMSE on test:   {results_df.loc[best_name, 'RMSE']:.3f}")
print(f"\nTop 3 features:\n", importances.head(3))