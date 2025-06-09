import joblib
import numpy as np

# Load model and scaler
model = joblib.load('model/model.pkl')
scaler = joblib.load('model/scaler.pkl')

# Sample input (must match training feature count)
sample_input = np.random.rand(1, 20) * 100  # 20 features like in training

# Scale and predict
scaled_input = scaler.transform(sample_input)
prediction = model.predict(scaled_input)
probability = model.predict_proba(scaled_input)

print("Sample Input:", sample_input[0])
print(f"Prediction: {'Malicious' if prediction[0] else 'Benign'}")
print(f"Confidence: {probability[0][1]:.2%}")