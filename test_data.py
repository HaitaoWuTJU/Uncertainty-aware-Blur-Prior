import torch
import numpy as np
import sys
import os


def check_ubp_data(filepath):
    print(f"==================================================")
    print(f"Inspecting File: {filepath}")
    print(f"==================================================\n")
    
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        sys.exit(1)
        
    # Load the dictionary
    # weights_only=False is required because the file contains NumPy arrays and strings
    data = torch.load(filepath, map_location='cpu', weights_only=False)
    
    # 1. Print overall keys
    print(f"Keys found in dataset: {list(data.keys())}\n")
    
    # 2. Inspect each key's shape and type
    for key, val in data.items():
        if isinstance(val, (np.ndarray, torch.Tensor)):
            print(f"-> '{key}'")
            print(f"   Type : {type(val).__name__}")
            print(f"   Shape: {val.shape}")
            print(f"   Dtype: {val.dtype}")
        elif isinstance(val, list):
            print(f"-> '{key}'")
            print(f"   Type : list")
            print(f"   Length: {len(val)}")
            if len(val) > 0:
                print(f"   Sample: {val[:4]} ...")
        else:
            print(f"-> '{key}': {type(val).__name__}")
        print("-" * 40)

    # 3. Print a human-readable summary
    print("\n================== SUMMARY =======================")
    if 'eeg' in data:
        eeg_shape = data['eeg'].shape
        print(f"Total Conditions (Images) : {eeg_shape[0]}")
        print(f"Repetitions per Condition : {eeg_shape[1]}")
        print(f"EEG Channels              : {eeg_shape[2]}")
        print(f"Timepoints                : {eeg_shape[3]}")
        
    if 'times' in data:
        times = data['times']
        print(f"Time Window               : [{times[0]:.3f}s to {times[-1]:.3f}s]")
        
    if 'img' in data:
        print(f"Sample Image Path         : {data['img'][0][0]}")
        
    if 'text' in data:
        print(f"Sample Text Label         : {data['text'][0][0]}")

    print("==================================================\n")

if __name__ == "__main__":
    # You can pass the path as a command line argument or use the default fallback
    default_path = "/work3/s193209/data/eeg_preprocessed_250hz/ubp_format/Preprocessed_data_250Hz_whiten_minus200til800/sub-01/train.pt"
    
    filepath = sys.argv[1] if len(sys.argv) > 1 else default_path
    check_ubp_data(filepath)