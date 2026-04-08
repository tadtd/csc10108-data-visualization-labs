import pandas as pd
import numpy as np
import os

def transform_features():
    """
    Adds discount_percent, log_price, and rating_bucket.
    """
    input_path = 'data/staging/merged_raw.csv'
    if not os.path.exists(input_path):
        print(f"File {input_path} not found.")
        return

    df = pd.read_csv(input_path)
    
    # Create discount_percent if not already correct
    if 'price' in df.columns and 'original_price' in df.columns:
        df['discount_percent'] = (1 - df['price'] / df['original_price']) * 100
        df['discount_percent'] = df['discount_percent'].fillna(0)
    
    # Create log_price
    if 'price' in df.columns:
        df['log_price'] = np.log1p(df['price'])
    
    # Create rating_bucket
    if 'rating' in df.columns:
        df['rating_bucket'] = pd.cut(df['rating'], bins=[0, 3, 4, 5], labels=['Low', 'Medium', 'High'])
    
    print("Transformed features.")
    df.to_csv(input_path, index=False)

if __name__ == "__main__":
    transform_features()
