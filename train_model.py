import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import joblib
import os

# Create sample dataset (replace with your actual data)
def create_sample_data():
    np.random.seed(42)
    num_samples = 1000
    num_features = 20
    
    # Features (network traffic characteristics)
    X = np.random.rand(num_samples, num_features) * 100
    
    # Labels (0 = benign, 1 = malicious)
    y = np.random.randint(0, 2, num_samples)
    
    # Make some features more important
    X[:, [0, 5, 12]] *= (y * 3 + 1).reshape(-1, 1)
    
    return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(num_features)]), y

# Create model directory
os.makedirs('model', exist_ok=True)

# Generate and prepare data
X, y = create_sample_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

def train_adaptive_model(X, y):
    """Train model that can handle missing features"""
    # Define preprocessing
    numeric_features = X.select_dtypes(include=['number']).columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_features = X.select_dtypes(include=['object']).columns
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Create adaptive pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            class_weight='balanced',
            random_state=42
        ))
    ])
    
    model.fit(X, y)
    return model

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train_scaled, y_train)

# Evaluate
print(f"Training Accuracy: {model.score(X_train_scaled, y_train):.2f}")
print(f"Testing Accuracy: {model.score(X_test_scaled, y_test):.2f}")

# Save model and scaler
joblib.dump(model, 'model/model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

print("\nModel and scaler saved to model/ directory")
print("Files created:")
print(f"- model/model.pkl ({os.path.getsize('model/model.pkl')/1024:.1f} KB)")
print(f"- model/scaler.pkl ({os.path.getsize('model/scaler.pkl')/1024:.1f} KB)")