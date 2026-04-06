import pandas as pd
import os

def clean_missing():
    """
    Fills missing values with mean, mode, or default values.
    """
    input_path = 'data/staging/merged_raw.csv'
    if not os.path.exists(input_path):
        print(f"File {input_path} not found.")
        return

    df = pd.read_csv(input_path)
    
    # Fill price with mean
    if 'price' in df.columns:
        df['price'] = df['price'].fillna(df['price'].mean())
    
    # Fill sold_count with 0
    if 'sold_count' in df.columns:
        df['sold_count'] = df['sold_count'].fillna(0)
    
    # Fill category with 'Unknown'
    if 'category' in df.columns:
        df['category'] = df['category'].fillna('Unknown')
    
    print("Filled missing values.")
    df.to_csv(input_path, index=False)

if __name__ == "__main__":
    clean_missing()
