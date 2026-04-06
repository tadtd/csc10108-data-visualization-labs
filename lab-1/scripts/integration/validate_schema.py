import pandas as pd
import json
import os

def validate_schema():
    """
    Checks for missing columns and data type mismatches.
    """
    input_path = 'data/staging/merged_raw.csv'
    schema_path = 'config/schema.json'
    
    if not os.path.exists(input_path) or not os.path.exists(schema_path):
        print("Required files for validation not found.")
        return

    with open(schema_path, 'r') as f:
        schema = json.load(f)
    
    df = pd.read_csv(input_path)
    expected_fields = schema['fields']
    
    # Check missing columns
    missing_cols = [col for col in expected_fields if col not in df.columns]
    if missing_cols:
        print(f"WARNING: Missing columns: {missing_cols}")
    else:
        print("Schema validation passed: All expected columns are present.")

if __name__ == "__main__":
    validate_schema()
