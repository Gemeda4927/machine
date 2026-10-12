"""
day1_ex1.py

Load the Seaborn "tips" dataset, split into features and target,
one-hot encode the categorical column, split into train/test sets,
and visualize the data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42

sns.set_theme(style="whitegrid")     # nicer default look

# ------------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------------
df = pd.read_csv(URL)

# ------------------------------------------------------------------
# Define features and target
# ------------------------------------------------------------------
features = pd.get_dummies(df[["total_bill", "sex"]], columns=["sex"])
target = df["tip"]

# ------------------------------------------------------------------
# Split into train and test sets
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

# ------------------------------------------------------------------
# Preview
# ------------------------------------------------------------------
print("Features (first 5):\n", features.head())
print("\nTarget (first 5):\n", target.head())

print("\n--- Train/Test split ---")
print(f"X_train shape: {X_train.shape}")
print(f"X_test  shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test  shape: {y_test.shape}")

print("\nX_train (first 5):\n", X_train.head())
print("\ny_train (first 5):\n", y_train.head())

# ==================================================================
# VISUALIZATIONS
# ==================================================================

# ------------------------------------------------------------------
# 1. Distribution of the target (tip)
# ------------------------------------------------------------------
plt.figure(figsize=(8, 4))
sns.histplot(target, bins=25, kde=True, color="steelblue")
plt.title("Distribution of Tips")
plt.xlabel("Tip (USD)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 2. Distribution of the main feature (total_bill)
# ------------------------------------------------------------------
plt.figure(figsize=(8, 4))
sns.histplot(df["total_bill"], bins=25, kde=True, color="coral")
plt.title("Distribution of Total Bill")
plt.xlabel("Total Bill (USD)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 3. Box plot — tips by sex
# ------------------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="sex", y="tip", palette="Set2")
plt.title("Tips by Sex")
plt.xlabel("Sex")
plt.ylabel("Tip (USD)")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 4. Scatter: total_bill vs tip, colored by sex
# ------------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="total_bill",
    y="tip",
    hue="sex",
    style="sex",
    s=70,
    alpha=0.8,
)
plt.title("Total Bill vs Tip (colored by Sex)")
plt.xlabel("Total Bill (USD)")
plt.ylabel("Tip (USD)")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 5. Correlation heatmap — numeric columns
# ------------------------------------------------------------------
numeric_df = df.select_dtypes(include="number")
plt.figure(figsize=(7, 5))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    cbar_kws={"shrink": 0.8},
)
plt.title("Correlation Heatmap (numeric columns)")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 6. Train vs test — target distribution
# ------------------------------------------------------------------
plt.figure(figsize=(8, 4))
sns.kdeplot(y_train, label="Train", fill=True, color="teal")
sns.kdeplot(y_test, label="Test", fill=True, color="orange")
plt.title("Target Distribution — Train vs Test")
plt.xlabel("Tip (USD)")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 7. Pair plot — all numeric columns (bonus)
# ------------------------------------------------------------------
sns.pairplot(
    df,
    vars=["total_bill", "tip", "size"],
    hue="sex",
    corner=True,
    diag_kind="kde",
)
plt.suptitle("Pair Plot — Numeric Features by Sex", y=1.02)
plt.show()

# ------------------------------------------------------------------
# 8. Bar chart — average tip by day
# ------------------------------------------------------------------
plt.figure(figsize=(7, 4))
sns.barplot(data=df, x="day", y="tip", estimator="mean", palette="pastel")
plt.title("Average Tip by Day")
plt.xlabel("Day")
plt.ylabel("Average Tip (USD)")
plt.tight_layout()
plt.show()