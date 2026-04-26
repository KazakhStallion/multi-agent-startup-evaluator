# NEED TO IMPLEMENT OTHERS FIRST, SO WE CAN USE THEIR COMBINED OUTPUTS

# import os
# import json
# from dotenv import load_dotenv
# from openai import OpenAI

# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")
# vt_client = (
#     OpenAI(
#         api_key=api_key,
#         base_url="https://llm-api.arc.vt.edu/api/v1/",
#     )
#     if api_key
#     else None
# )

# class SkepticAgent:
#     def __init__(self, client= vt_client,model="gpt-oss-120b"):
#         self.client = client
#         self.model = model
    
#     def analyze(self, startup):
#         try:
#             startup_json = startup.get('finances', {})

#             annual_rev = float(startup_json.get('revenue',0))
#             monthly_rev = annual_rev / 12
#             monthly_burn = float(startup_json.get('burn_rate',0))
#             cash = float(startup_json.get('funding',0))
            

#             startup_data = f"""
#             Company Name: {startup.get('identity', {}).get('name')}
#             Annual Revenue: {annual_rev}
#             Monthly Revenue: {monthly_rev:.2f}
#             Monthly Burn: {monthly_burn}
#             Cash: {cash}
#             """

#             prompt = """
#             AUDIT TASK: Extract and calculate metrics based EXCLUSIVELY on the data in <data>.

#             COMPANY DATA
#             {startup_data}

#             ### YOUR AUDIT WORKFLOW:
#             For each task, provide a JSON object following this logic:
#             1. Reference ONLY the exact numbers from COMPANY DATA.
#             2. Perform the calculation.
#             3. Explain how you came to your conclusion.

#             ### TASKS TO COMPLETE:
#             1. Burn Efficiency: Monthly Burn vs Monthly Revenue.
#             2. Risk Level: Capital Risk (Low/Med/High).
#             3. Burn Multiple: Net Burn / Net New ARR.
#             4. Runway: Months of survival at zero growth.
#             5. Unit Economics: LTV/CAC sustainability.
#             6. 10x Goal: Revenue needed for 10x exit.
#             7. Intensity: Capital Intensity Rating (Low/Med/High).
#             8. Inquiry: One critical question for the founder.
#             9. Total Rating: 1-10
#             10. Final Decision: Go, No-Go, Pivot, with detailed reasoning


#             Return ONLY a valid JSON object with exactly these keys:
#             {{
#             "agent": "Finance",
#             "burn_efficiency": {{
#                 "calculation": "string showing the math",
#                 "result": number,
#                 "reasoning": "string"
#             }},
#             "capital_risk": {{
#                 "level": "low/medium/high",
#                 "reasoning": "string"
#             }},
#             "burn_multiple": {{
#                 "calculation": "string showing the math",
#                 "result": number,
#                 "reasoning": "string"
#             }},
#             "runway": {{
#                 "calculation": "cash / monthly_burn",
#                 "months": number,
#                 "reasoning": "string"
#             }},
#             "ten_x_goal": {{
#                 "target_revenue_usd": number,
#                 "calculation": "string"
#             }},
#             "capital_intensity": {{
#                 "rating": "low/medium/high",
#                 "reasoning": "string"
#             }},
#             "founder_inquiry": "string",
#             "total_rating": 5,
#             "final_decision": {{
#                 "decision": "Go/No-Go/Pivot",
#                 "recommendation": "string"
#             }}
#             }}""".strip()
#             # ### OUTPUT FORMAT:
#             # Return ONLY a valid JSON object. 
#             # """

#             # Example Task Format:
#             # "runway": {{
#             #     "data_points": {{"cash": "ACTUAL_CASH_ON_HAND", "burn": "ACTUAL_MONTHLY_BURN"}},
#             #     "calculation": "cash / burn",
#             #     "result": "RESULT STRING"
#             # }}
#             # """

#             final_prompt = prompt.format(
#                 startup_data=startup_data
#                 )


#             response = self.client.chat.completions.create(
#                 model= self.model, 
#                 messages=[{"role": "user", "content": final_prompt}],
#                 response_format={ "type": "json_object" } 
#             )

#             print("Using Model: ", self.model)
            
#             return response.choices[0].message.content

#         except Exception as e:
#             print(f"Failed to analyze {startup}: {e}")