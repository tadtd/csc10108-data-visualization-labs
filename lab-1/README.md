# Tiki Data Engineering & Analysis Project

This project aims to crawl product data from Tiki, process it, and provide insights through a dashboard.

## 1. Directory Structure

- `data/`: Contains raw, staging, and processed data.
- `scripts/`: Python scripts for crawling, integration, and preprocessing.
- `config/`: JSON configuration files for categories and schema.
- `dashboard/`: Streamlit dashboard application.
- `notebooks/`: Jupyter notebooks for exploratory data analysis.
- `logs/`: Execution logs.

## 2. Workflow

### Step 1: Data Collection
Each member crawls their assigned categories:
```bash
python scripts/crawler/member_1/crawl_listing.py
python scripts/crawler/member_2/crawl_listing.py
```

### Step 2: Data Integration
Merge and validate the collected data:
```bash
python scripts/integration/merge_data.py
python scripts/integration/remove_duplicates.py
python scripts/integration/validate_schema.py
```

### Step 3: Data Preprocessing
Clean and transform the data:
```bash
python scripts/preprocessing/clean_missing.py
python scripts/preprocessing/handle_outliers.py
python scripts/preprocessing/transform_features.py
python scripts/preprocessing/build_final_dataset.py
```

### Step 4: Analysis & Dashboard
Run the dashboard:
```bash
streamlit run dashboard/app.py
```

## 3. Installation

```bash
pip install -r requirements.txt
```

## 4. Database
<image src="assets/[DV]-Book-DB.svg" alt="Database Schema" width="600"/>