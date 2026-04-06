import pandas as pd
import glob
import os

def merge_data():
    """
    Reads all CSV files from data/raw/member_1 and member_2,
    merges them, and saves to data/staging/merged_raw.csv
    """
    raw_dirs = ['data/raw/member_1', 'data/raw/member_2']
    all_files = []
    for d in raw_dirs:
        all_files.extend(glob.glob(os.path.join(d, "*.csv")))
    
    if not all_files:
        print("No CSV files found to merge.")
        # Create an empty dataframe with schema columns if no files found for skeleton testing
        import json
        with open('config/schema.json', 'r') as f:
            schema = json.load(f)
        df = pd.DataFrame(columns=schema['fields'])
    else:
        df_list = [pd.read_csv(f) for f in all_files]
        df = pd.concat(df_list, ignore_index=True)
    
    os.makedirs('data/staging', exist_ok=True)
    df.to_csv('data/staging/merged_raw.csv', index=False)
    print(f"Merged {len(all_files)} files into data/staging/merged_raw.csv")

if __name__ == "__main__":
    merge_data()
