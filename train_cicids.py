import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from sklearn.preprocessing import StandardScaler

# Enhanced data loading with detailed validation
try:
    print("Loading and validating processed data...")
    df = pd.read_csv('processed_cicids2017.csv')
    
    # Check for required columns
    required_cols = ['Label']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
        
    # Check and convert data types
    for col in df.columns:
        if col != 'Label':
            if not np.issubdtype(df[col].dtype, np.number):
                try:
                    # Try converting to datetime first, then numeric
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    if df[col].isna().any():
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    else:
                        df[col] = df[col].astype(np.int64) // 10**9  # Unix timestamp
                except:
                    raise ValueError(f"Could not convert column '{col}' to numeric")
    
    print(f"Successfully loaded {len(df)} rows with {len(df.columns)} features")
    print("\nSample of the processed data:")
    print(df.head(3))
    print("\nData types:")
    print(df.dtypes)

except Exception as e:
    print(f"\nERROR during data loading/validation: {str(e)}")
    print("\nPossible solutions:")
    print("1. Ensure preprocess.py has been run successfully")
    print("2. Verify processed_cicids2017.csv exists and is valid")
    print("3. Check for non-numeric values in feature columns")
    raise

# Split features and labels
X = df.drop(['Label'], axis=1)
y = df['Label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    min_samples_split=5,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Evaluate
print("Training Results:")
print(classification_report(y_train, model.predict(X_train_scaled)))
print("\nTest Results:")
print(classification_report(y_test, model.predict(X_test_scaled)))

# Save artifacts
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/cicids_model.pkl')
joblib.dump(scaler, 'model/cicids_scaler.pkl')
print("\nModel saved to model/cicids_model.pkl")
print("Scaler saved to model/cicids_scaler.pkl")