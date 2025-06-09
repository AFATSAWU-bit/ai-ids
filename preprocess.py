import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

def load_and_preprocess(filepath):
    # Load CSV
    df = pd.read_csv(filepath)
    
    # Clean infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Feature engineering - handle numeric conversions safely
    numeric_cols = ['Flow Bytes/s', 'Flow Packets/s']
    for col in numeric_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                print(f"Error converting {col}: {str(e)}")
                problem_rows = df[~df[col].apply(lambda x: str(x).replace('.','',1).isdigit())]
                print(f"Problematic values in {col}: {problem_rows[col].unique()}")
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert datetime columns to numeric (if applicable)
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce').astype(np.int64) // 10**9  # Convert to Unix timestamp
        logger.info("Converted 'Timestamp' to Unix timestamp")
    
    # Label encoding for categorical features
    cat_cols = ['Protocol', 'Fwd PSH Flags', 'Bwd PSH Flags', 
               'Fwd URG Flags', 'Bwd URG Flags', 'FIN Flag Count',
               'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
               'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count',
               'ECE Flag Count']
    
    for col in cat_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col])
    
    # Binary labels (1 for attack, 0 for benign)
    df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    
    return df

import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='preprocessing.log',
    filemode='w'
)
logger = logging.getLogger()

# Process all files with detailed logging
logger.info("Starting data preprocessing...")
data_dir = 'CICIDS2017/MachineLearningCSV'
logger.info(f"Looking for CSV files in: {os.path.abspath(data_dir)}")

# Enhanced directory verification
logger.info("Verifying directory structure...")
if not os.path.exists('CICIDS2017'):
    print("ERROR: CICIDS2017 directory not found at:", os.path.abspath('CICIDS2017'))
    print("Current working directory contents:")
    print(os.listdir('.'))
    raise FileNotFoundError("CICIDS2017 directory missing")

if not os.path.exists(data_dir):
    print("ERROR: MachineLearningCSV directory not found at:", os.path.abspath(data_dir))
    print("Contents of CICIDS2017 directory:")
    print(os.listdir('CICIDS2017'))
    raise FileNotFoundError("MachineLearningCSV directory missing")

print("Directory structure verified successfully")

all_dfs = []
try:
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not files:
        print("\nERROR: No CSV files found in directory")
        print(f"Please ensure the {data_dir} folder contains the dataset CSV files")
        raise FileNotFoundError(f"No CSV files found in {os.path.abspath(data_dir)}")
    
    print(f"Found {len(files)} CSV files in directory")
except Exception as e:
    print(f"\nERROR accessing directory: {str(e)}")
    raise

for file in files:
        if file.endswith('.csv'):
            print(f"\nProcessing {file}...")
            try:
                df = load_and_preprocess(f'CICIDS2017/MachineLearningCSV/{file}')
                print(f"Processed {len(df)} rows from {file}")
                all_dfs.append(df)
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue

# Combine and shuffle
full_df = pd.concat(all_dfs).sample(frac=1).reset_index(drop=True)

# Save processed data
full_df.to_csv('processed_cicids2017.csv', index=False)
print("Saved processed data to processed_cicids2017.csv")