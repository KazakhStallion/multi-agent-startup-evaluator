import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def hydrate_startup(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. Safety Check: Ensure keys exist so we don't get KeyErrors later
    if 'finances' not in data:
        data['finances'] = {"revenue": "Unknown", "burn_rate": "Unknown", "runway": "Unknown"}
    if 'business' not in data:
        data['business'] = {"description": "Unknown"}

    # 2. Only call the API if we actually have "Unknown" data
    if data['business']['description'] == "Unknown":
        name = data['identity'].get('name', 'Unknown Startup')
        sector = data['identity'].get('sector', 'Technology')
        
        print(f"Hydrating {name} using Groq...")
        
        prompt = f"""
        Provide a concise 2-paragraph business description for the startup '{name}' in the '{sector}' sector. 
        Then, generate 3 realistic but fictional seed-stage financial metrics: 
        Monthly Recurring Revenue (MRR), Monthly Burn Rate, and Runway (in months).
        Return the result in this exact JSON format:
        {{
            "description": "text here",
            "mrr": "$value",
            "burn": "$value",
            "runway": "value months"
        }}
        """

        try:
            # Using the current Llama 3.3 70B model ID on Groq
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile", 
                messages=[{"role": "user", "content": prompt}],
                response_format={ "type": "json_object" } 
            )
            
            enriched = json.loads(response.choices[0].message.content)
            
            # 3. Apply the new data
            data['business']['description'] = enriched.get('description', "Description failed to generate.")
            data['finances']['revenue'] = enriched.get('mrr', "Unknown")
            data['finances']['burn_rate'] = enriched.get('burn', "Unknown")
            data['finances']['runway'] = enriched.get('runway', "Unknown")

            # 4. Save the file ONLY if we successfully hydrated
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            print(f"Successfully hydrated {name}.")

        except Exception as e:
            print(f"Failed to hydrate {name}: {e}")
    else:
        print(f"Skip: {data['identity'].get('name')} already has a description.")

if __name__ == "__main__":
    # Choose a specific file to test
    test_file = os.path.join("data", "finance", "processed", "everyme.json")
    
    if os.path.exists(test_file):
        print(f"--- Before Hydration: {test_file} ---")
        with open(test_file, 'r', encoding='utf-8') as f:
            print(json.dumps(json.load(f), indent=2))
        
        print("\nRunning Hydration...")
        hydrate_startup(test_file)
        
        print(f"\n--- After Hydration: {test_file} ---")
        with open(test_file, 'r', encoding='utf-8') as f:
            # Re-read the file to see the changes
            final_data = json.load(f)
            print(json.dumps(final_data, indent=2))
    
    else:
        print(f"File not found: {test_file}")