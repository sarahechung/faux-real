import pandas as pd
import joblib

# Load test data
test_data = pd.read_csv("test.csv")

# Load saved model
model = joblib.load("model.pkl")

# Predict (no target column in test data assumed)
predictions = model.predict(test_data)

# Save predictions
output = pd.DataFrame(predictions, columns=["Prediction"])
output.to_csv("predictions.csv", index=False)

print("Predictions saved to predictions.csv")
