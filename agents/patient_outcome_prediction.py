import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Fictional patient data
data = pd.DataFrame({
    "age": [25, 68, 32, 72, 61, 45, 55, 70, 38, 66],
    "glucose": [110, 190, 120, 210, 175, 130, 160, 200, 115, 185],
    "readmitted": [0, 1, 0, 1, 1, 0, 1, 1, 0, 1]
})

# Input features
X = data[["age", "glucose"]]

# Known outcome labels: 0 = No, 1 = Yes
y = data["readmitted"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Create and train the model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression()
)

model.fit(X_train, y_train)

# Predict probabilities for the test patients
probabilities = model.predict_proba(X_test)[:, 1]

# Predict outcome categories
predictions = model.predict(X_test)

# Evaluate the predictions
print(classification_report(y_test, predictions))

# Display estimated probabilities
print(probabilities)