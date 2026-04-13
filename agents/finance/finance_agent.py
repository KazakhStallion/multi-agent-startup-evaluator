import os
import json
# from groq import Groq
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
vt_client = (
    OpenAI(
        api_key=api_key,
        base_url="https://llm-api.arc.vt.edu/api/v1/",
    )
    if api_key
    else None
)

class FinanceAgent:
    def __init__(self, client= vt_client,model="gpt-oss-120b"):
        self.client = client
        self.model = model
    
    def analyze(self, startup):
        try:
            # prompt = """
            # You are a Senior VC Financial Analyst. Analyze this startup:

            # ### STARTUP DATA:
            # {startup}

            # ### MISSION:
            # For each of the tasks below, follow this three-step process:
            # 1. DATA CHECK: Identify the specific values from the STARTUP DATA needed for this task. 
            # 2. MISSING INFO: If data is missing (Unknown), explicitly state what specific financial document or metric you would need from the founder (e.g., "Full 12-month P&L", "Cohorted Retention Data").
            # 3. CALCULATION: Perform the math using provided data. If data was missing, provide a "Best Estimate" based on the 'sector' and 'model' provided in the data, but clearly tag it as [ESTIMATED].

            # ### TASKS:
            # Tasks:
            # 1. Calculate the Monthly Burn vs Revenue efficiency.
            # 2. Identify the Capital Risk Level (Low, Medium, High).
            # 3. Calculate the Burn Multiple (Net Burn / Net New ARR) to assess spending efficiency.
            # 4. Perform a Zero Growth Runway Stress Test: months until $0 if growth stalls.
            # 5. Evaluate Unit Economics (LTV/CAC): Is the cost of acquisition sustainable?
            # 6. Estimate the revenue required for a 10x return based on sector multiples.
            # 7. Identify the Capital Intensity Rating (Low, Medium High): Is this business too expensive to scale?
            # 8. Provide one difficult question the founder must answer.

            # ### CONSTRAINTS:
            # - Use ONLY the numerical values provided in the STARTUP DATA.
            # - Do not use placeholder variable names in the output. Replace them with the actual numbers. For example, do not write "revenue" / 12, write "97866143 / 12 = 8155511".
            # - FORBIDDEN: Do not change provided numbers (e.g., if Revenue is 97000000, do not use 200000).
            # - FORMAT: Output MUST be a valid JSON object.

            # ### DATA MAPPING DICTIONARY (MANDATORY):
            # - 'revenue' = Total Revenue. (If > 5,000,000, treat as Annual; divide by 12 for Monthly).
            # - 'burn_rate' = Monthly Burn Rate. THIS IS NOT UNKNOWN. USE THIS VALUE.
            # - 'funding' = Total Cash/Capital Raised.
            # - 'marketing_expense' = Monthly S&M (Sales and Marketing) spend.
            # - 'retention' = Customer Retention Rate.

            # ### CRITICAL INSTRUCTION:
            # If a value is provided in the 'finances' object, you are FORBIDDEN from labeling it as 'Missing' or 'Unknown'. 
            # Look at 'burn_rate' before claiming you don't have the burn rate.

            # ### OUTPUT SCHEMA (Example for each task):
            # {{
            # "data_audit": {{
            #     "verified_annual_revenue": "...", 
            #     "verified_monthly_burn": "...",
            #     "verified_total_funding": "..."
            # }},
            # "task_name": {{
            #     "provided_data_used": {{ "key": "value" }},
            #     "missing_info_required": "string",
            #     "math": "string showing the steps",
            #     "estimation_logic": "string",
            #     "result": "string",
            #     "reasoning": "string"
            # }},
            # "final_verdict": "Go/Pivot/No-Go"
            # }}
            # """

            startup_json = startup.get('finances', {})

            annual_rev = float(startup_json.get('revenue',0))
            monthly_rev = annual_rev / 12
            monthly_burn = float(startup_json.get('burn_rate',0))
            cash = float(startup_json.get('funding',0))
            

            startup_data = f"""
            Company Name: {startup.get('identity', {}).get('name')}
            Annual Revenue: {annual_rev}
            Monthly Revenue: {monthly_rev:.2f}
            Monthly Burn: {monthly_burn}
            Cash: {cash}
            """

            prompt = """
            AUDIT TASK: Extract and calculate metrics based EXCLUSIVELY on the data in <data>.

            COMPANY DATA
            {startup_data}

            ### YOUR AUDIT WORKFLOW:
            For each task, provide a JSON object following this logic:
            1. Reference ONLY the exact numbers from COMPANY DATA.
            2. Perform the calculation.
            3. Explain how you came to your conclusion.

            ### TASKS TO COMPLETE:
            1. Burn Efficiency: Monthly Burn vs Monthly Revenue.
            2. Risk Level: Capital Risk (Low/Med/High).
            3. Burn Multiple: Net Burn / Net New ARR.
            4. Runway: Months of survival at zero growth.
            5. Unit Economics: LTV/CAC sustainability.
            6. 10x Goal: Revenue needed for 10x exit.
            7. Intensity: Capital Intensity Rating (Low/Med/High).
            8. Inquiry: One critical question for the founder.
            9. Total Rating: 1-10
            10. Final Decision: Go, No-Go, Pivot, with detailed reasoning


            Return ONLY a valid JSON object with exactly these keys:
            {{
            "agent": "Finance",
            "burn_efficiency": {{
                "calculation": "string showing the math",
                "result": number,
                "reasoning": "string"
            }},
            "capital_risk": {{
                "level": "low/medium/high",
                "reasoning": "string"
            }},
            "burn_multiple": {{
                "calculation": "string showing the math",
                "result": number,
                "reasoning": "string"
            }},
            "runway": {{
                "calculation": "cash / monthly_burn",
                "months": number,
                "reasoning": "string"
            }},
            "ten_x_goal": {{
                "target_revenue_usd": number,
                "calculation": "string"
            }},
            "capital_intensity": {{
                "rating": "low/medium/high",
                "reasoning": "string"
            }},
            "founder_inquiry": "string",
            "total_rating": 5,
            "final_decision": {{
                "decision": "Go/No-Go/Pivot",
                "recommendation": "string"
            }}
            }}""".strip()
            # ### OUTPUT FORMAT:
            # Return ONLY a valid JSON object. 
            # """

            # Example Task Format:
            # "runway": {{
            #     "data_points": {{"cash": "ACTUAL_CASH_ON_HAND", "burn": "ACTUAL_MONTHLY_BURN"}},
            #     "calculation": "cash / burn",
            #     "result": "RESULT STRING"
            # }}
            # """

            final_prompt = prompt.format(
                startup_data=startup_data
                )

            # print("\n--- DEBUG: PROMPT SENT TO GROQ ---")
            # print(final_prompt)
            # print("----------------------------------\n")

            response = self.client.chat.completions.create(
                model= self.model, 
                messages=[{"role": "user", "content": final_prompt}],
                response_format={ "type": "json_object" } 
            )

            print("Using Model: ", self.model)
            
            return response.choices[0].message.content

        except Exception as e:
            print(f"Failed to analyze {startup}: {e}")