import pandas as pd
from sklearn.datasets import fetch_openml
import os

print("Downloading the EEG Eye State dataset from OpenML...")
try:
    # Fetch dataset from OpenML (ID 1471: EEG-eye-state)
    # The dataset contains 14 integer features (sensors) and 1 target class
    eeg = fetch_openml('eeg-eye-state', version=1, as_frame=True, parser='auto')
    
    df = eeg.frame
    
    # In OpenML, the target column is usually 'Class'. We'll find it dynamically.
    target_col_original = eeg.target_names[0]
    
    # OpenML mapped: '1' is eye open, '2' is eye closed.
    # We will map it to standard binary: 0 for open, 1 for closed.
    mapping = {'1': 0, '2': 1}
    df['eye_state'] = df[target_col_original].map(mapping)
    
    # Drop the original string class
    df = df.drop(columns=[target_col_original])
    
    # Save to CSV
    csv_filename = 'eeg_data.csv'
    df.to_csv(csv_filename, index=False)
    print(f"Dataset successfully downloaded and saved as '{csv_filename}'.\nTotal rows: {len(df)}")
except Exception as e:
    print(f"Error downloading dataset: {e}")
