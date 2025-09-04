import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load training data
data = pd.read_csv("train.csv")

# Assumes last column is the target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split into train/test for quick evaluation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))

# Save the trained model
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")
