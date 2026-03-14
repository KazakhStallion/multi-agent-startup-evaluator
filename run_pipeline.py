import pandas as pd
import json
import os

from agents.finance_agent import FinanceAgent


if __name__ == "__main__":
    # Choose a specific file to test
    file_name = "startup_3.json"
    test_file = os.path.join("data", "processed", file_name)

    finance_agent = FinanceAgent(use_local=False, model="openai/gpt-oss-120b")
    
    if os.path.exists(test_file):
        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(data)
        print(finance_agent.analyze(data))
    
    else:
        print(f"File not found: {test_file}")