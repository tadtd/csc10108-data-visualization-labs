import pandas as pd
import os

def build_final_dataset():
    """
    Builds products.csv, reviews.csv, and authors.csv
    """
    input_path = 'data/staging/merged_raw.csv'
    if not os.path.exists(input_path):
        print(f"File {input_path} not found.")
        return

    df = pd.read_csv(input_path)
    os.makedirs('data/processed', exist_ok=True)
    
    # Save products.csv
    df.to_csv('data/processed/products.csv', index=False)
    
    # Mock data for reviews.csv (normally would be crawled differently)
    reviews_df = pd.DataFrame(columns=['product_id', 'user_id', 'rating', 'comment'])
    reviews_df.to_csv('data/processed/reviews.csv', index=False)
    
    # Mock data for authors.csv
    if 'author' in df.columns:
        authors = df['author'].unique()
        authors_df = pd.DataFrame(authors, columns=['author_name'])
        authors_df.to_csv('data/processed/authors.csv', index=False)
    
    print("Final datasets created in data/processed/")

if __name__ == "__main__":
    build_final_dataset()
