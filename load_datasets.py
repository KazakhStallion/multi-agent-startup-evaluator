import os
import zipfile
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

# Define the datasets using their Kaggle 'username/dataset-name' format
DATASETS = {
    "yc_2025": "mohamedasak/y-combinator-startup-directory-2025",
    "yc_latest": "amanpriyanshu/latest-yc-data",
    "startup_failures": "dagloxkankwanda/startup-failures",
    "failure_prediction": "sakharebharat/startup-failure-prediction-dataset",
    "global_startups": "adarsh2626/startup-dataset",
    # For Market Analyst Agent
    "crunchbase_investments": "arindam235/startup-investments-crunchbase",
    "startup_failures": "dagloxkankwanda/startup-failures"

}

RAW_DATA_DIR = "data/raw"

def download_and_extract(api, dataset_path, dest_folder):
    """Downloads a Kaggle dataset and extracts its contents."""
    print(f"Downloading {dataset_path}...")
    
    # Download the dataset archive
    api.dataset_download_files(dataset_path, path=dest_folder, unzip=False)
    
    # The downloaded file is a zip archive named after the dataset slug
    dataset_slug = dataset_path.split("/")[-1]
    zip_path = os.path.join(dest_folder, f"{dataset_slug}.zip")
    
    # Extract and clean up the zip file
    if os.path.exists(zip_path):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_folder)
        os.remove(zip_path)
    print(f"Completed: {dataset_slug}\n")

def load_csv_files(directory):
    """Loads all CSV files in the directory into a dictionary of DataFrames."""
    dataframes = {}
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            try:
                # Load into a pandas DataFrame
                df = pd.read_csv(file_path)
                dataframes[filename] = df
                print(f"Loaded {filename} - Shape: {df.shape}")
            except Exception as e:
                print(f"Could not load {filename}: {e}")
    return dataframes

if __name__ == "__main__":
    # Ensure the target directory exists
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    # Initialize and authenticate the Kaggle API
    print("Authenticating with Kaggle...")
    api = KaggleApi()
    api.authenticate()
    
    # 1. Download all datasets
    for name, dataset_path in DATASETS.items():
        download_and_extract(api, dataset_path, RAW_DATA_DIR)
        
    # 2. Load them into Pandas for inspection/processing
    print("Loading extracted CSVs into Pandas...")
    dfs = load_csv_files(RAW_DATA_DIR)
    
    print("\n--- Data Explorer: Initial Inspection ---")
    for filename, df in dfs.items():
        print(f"\n{'='*40}")
        print(f"FILE: {filename}")
        print(f"{'='*40}")
        
        # Show column names and data types
        print("\n[Columns & Types]")
        print(df.dtypes)
        
        # Show a snippet of the actual data
        print("\n[First 3 Rows]")
        print(df.head(3))
        
        # Check for missing values - critical for your agents!
        null_count = df.isnull().sum().sum()
        print(f"\n[Quality Check] Total Missing Values: {null_count}")
    
    print("\n" + "="*40)
    print("All datasets are loaded, inspected, and ready for processing.")