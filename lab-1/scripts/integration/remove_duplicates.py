import pandas as pd
import os

def remove_duplicates():
    """
    Loads merged_raw.csv, drops duplicates by product_id, and saves.
    """
    input_path = 'data/staging/merged_raw.csv'
    if not os.path.exists(input_path):
        print(f"File {input_path} not found.")
        return
    
    df = pd.read_csv(input_path)
    initial_len = len(df)
    
    if 'product_id' in df.columns:
        df = df.drop_duplicates(subset=['product_id'])
    else:
        df = df.drop_duplicates()
        
    print(f"Removed {initial_len - len(df)} duplicates.")
    df.to_csv(input_path, index=False)

if __name__ == "__main__":
    remove_duplicates()
