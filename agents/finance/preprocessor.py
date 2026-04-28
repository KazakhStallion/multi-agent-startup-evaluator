import pandas as pd
import json
import os
import random

PROCESSED_DIR = "data/finance/processed"
RAW_DIR = "data/finance/raw"

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

    # --- 3. Process Startup Failure Prediction (Quantitative Data) ---
    pred_path = os.path.join(RAW_DIR, "startup_failure_prediction.csv")
    if os.path.exists(pred_path):
        df_pred = pd.read_csv(pred_path)
        for _, row in df_pred.iterrows():
            case = {
                "metadata": {"source": "startup_failure_prediction.csv", "label": "Numerical Analysis"},
                "identity": {
                    "name": clean_value(row.get('Startup_Name', 'Unknown')),
                    "sector": clean_value(row.get('Industry', 'Unknown'))
                },
                "business": {
                    "description": "Unknown", # We will still need to hydrate the 'story' for these
                    "model": clean_value(row.get('Business_Model', 'Unknown'))
                },
                "finances": {
                    "revenue": clean_value(row.get('Revenue', 0)),
                    "burn_rate": clean_value(row.get('Burn_Rate', 0)),
                    "funding": clean_value(row.get('Funding_Amount', 0)),
                    "retention": clean_value(row.get('Customer_Retention_Rate', 0)),
                    "marketing_expense": clean_value(row.get('Marketing_Expense', 0)),
                    "employee_count": clean_value(row.get('Employees_Count', 0))
                }
            }
            save_json(case)
            count += 1

    # --- 4. Process Startup Dataset (Growth Data) ---
    growth_path = os.path.join(RAW_DIR, "Startup Dataset.csv")
    if os.path.exists(growth_path):
        df_growth = pd.read_csv(growth_path)
        for _, row in df_growth.iterrows():
            case = {
                "metadata": {"source": "Startup Dataset.csv", "label": clean_value(row.get('Current Status'))},
                "identity": {
                    "name": clean_value(row.get('Name', 'Unknown')),
                    "sector": "Unknown" # This dataset doesn't have a clear sector column
                },
                "business": {"description": clean_value(row.get('Description', 'Unknown'))},
                "finances": {
                    "revenue_y1": clean_value(row.get('Revenue Year 1')),
                    "revenue_y2": clean_value(row.get('Revenue Year 2')),
                    "revenue_y3": clean_value(row.get('Revenue Year 3')),
                    "burn_rate": "Unknown"
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