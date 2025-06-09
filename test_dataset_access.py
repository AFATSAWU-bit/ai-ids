import os

print("Testing dataset access...")
data_dir = 'CICIDS2017/MachineLearningCSV'

print(f"\nCurrent directory: {os.getcwd()}")
print(f"Contents: {os.listdir('.')}")

print(f"\nChecking for CICIDS2017 directory...")
if os.path.exists('CICIDS2017'):
    print("Found CICIDS2017 directory")
    print(f"Contents: {os.listdir('CICIDS2017')}")
    
    if os.path.exists(data_dir):
        print(f"\nFound MachineLearningCSV directory")
        files = os.listdir(data_dir)
        print(f"Found {len(files)} files")
        csv_files = [f for f in files if f.endswith('.csv')]
        print(f"Found {len(csv_files)} CSV files")
    else:
        print(f"\nERROR: {data_dir} not found")
else:
    print("ERROR: CICIDS2017 directory not found")
