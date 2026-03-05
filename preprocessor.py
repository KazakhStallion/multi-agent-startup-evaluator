import pandas as pd
import json
import os
import random

PROCESSED_DIR = "data/processed"
RAW_DIR = "data/raw"

def clean_value(val):
    if pd.isna(val): return "Unknown"
    return str(val).strip()

def save_json(data):
    # Standardize the filename: lowercase, no spaces, alphanumeric only
    name = data['identity']['name'].replace(" ", "_").lower()
    filename = "".join([c for c in name if c.isalnum() or c == '_']) 
    
    # Handle cases where name might be empty after cleaning
    if not filename:
        filename = "unknown_startup_" + str(random.randint(1000, 9999))
    
    full_path = os.path.join(PROCESSED_DIR, f"{filename}.json")
    
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def process_datasets():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    count = 0
    
    # --- 1. Process YC 2025 (Successes) ---
    yc_filename = "Y_Combinator_2025.csv"
    yc_path = os.path.join(RAW_DIR, yc_filename)
    
    if os.path.exists(yc_path):
        df_yc = pd.read_csv(yc_path)
        for _, row in df_yc.iterrows():
            case = {
                "metadata": {"source": yc_filename, "label": "Successful"},
                "identity": {
                    "name": clean_value(row.get('company_name', 'Unknown')), 
                    "sector": clean_value(row.get('industry_2', 'Unknown')),
                    "location": clean_value(row.get('location', 'Unknown'))
                },
                "business": {"description": clean_value(row.get('company_description', 'Unknown'))},
                "finances": {
                    "revenue": "Unknown",
                    "burn_rate": "Unknown",
                    "runway": "Unknown"
                }
            }
            save_json(case)
            count += 1

    # --- 2. Process Failure Sector Files (Failures) ---
    for file_name in os.listdir(RAW_DIR):
        if "Failure" in file_name and file_name.endswith(".csv"):
            file_path = os.path.join(RAW_DIR, file_name)
            df_fail = pd.read_csv(file_path)
            for _, row in df_fail.iterrows():
                case = {
                    "metadata": {"source": file_name, "label": "Failed"},
                    "identity": {
                        "name": clean_value(row.get('Name', 'Unknown')), 
                        "sector": clean_value(row.get('Sector', 'Unknown'))
                    },
                    "business": {"description": clean_value(row.get('What They Did', 'Unknown'))},
                    "finances": {
                        "revenue": "Unknown",
                        "burn_rate": "Unknown",
                        "runway": "Unknown"
                    },
                    "ground_truth": {
                        "reason": clean_value(row.get('Why They Failed', 'N/A')),
                        "takeaway": clean_value(row.get('Takeaway', 'N/A'))
                    }
                }
                save_json(case)
                count += 1
    return count

if __name__ == "__main__":
    print("Starting Pre-processing...")
    total_processed = process_datasets()
    print(f"Successfully processed {total_processed} startups into {PROCESSED_DIR}.")

    # --- Verification Block ---
    print("\n" + "="*50)
    print("VERIFICATION: Random Sample Results")
    print("="*50)
    
    processed_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith('.json')]
    
    if processed_files:
        # Pick 2 random samples to show the diversity
        samples = random.sample(processed_files, min(2, len(processed_files)))
        
        for sample_file in samples:
            print(f"\n[SAMPLE FILE: {sample_file}]")
            with open(os.path.join(PROCESSED_DIR, sample_file), 'r') as f:
                sample_data = json.load(f)
                print(json.dumps(sample_data, indent=2))
    else:
        print("Error: No processed files found. Check your RAW_DIR path.")

    print("\n" + "="*50)
    print("Pre-processing Complete.")