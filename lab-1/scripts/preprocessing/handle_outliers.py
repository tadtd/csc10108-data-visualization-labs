import pandas as pd
import numpy as np
import os

def handle_outliers():
    """
    Removes outliers using the IQR method for price.
    """
    input_path = 'data/staging/merged_raw.csv'
    if not os.path.exists(input_path):
        print(f"File {input_path} not found.")
        return

    df = pd.read_csv(input_path)
    
    if 'price' in df.columns:
        initial_len = len(df)
        Q1 = df['price'].quantile(0.25)
        Q3 = df['price'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
        print(f"Removed {initial_len - len(df)} price outliers.")
    
    df.to_csv(input_path, index=False)

if __name__ == "__main__":
    handle_outliers()
