import json
import os
import re
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "tech_lead"
OUTPUT_DIR = DATA_DIR / "outputs"


class TechnicalLeadAgent:
    def __init__(self, use_local=False, model="llama-3.3-70b-versatile"):
        self.use_local = use_local
        self.model = model
        if not self.use_local:
            self.client = client

    def analyze(self, startup):
        if self.use_local:
            return self._local_analysis(startup)

        try:
            identity = startup.get("identity", {})
            business = startup.get("business", {})
            finances = startup.get("finances", {})

            startup_data = f"""
            Company Name: {identity.get('name', 'Unknown')}
            Sector: {identity.get('sector', 'Unknown')}
            Location: {identity.get('location', 'Unknown')}
            Business Model: {business.get('model', 'Unknown')}
            Description: {business.get('description', 'Unknown')}
            Team Size (if provided): {finances.get('employee_count', 'Unknown')}
            """

            prompt = """
            You are a Technical Lead evaluating a startup pitch from a technology strategy perspective.
            Use only information in COMPANY DATA. If information is missing, mark it as Unknown and explain what to validate.

            COMPANY DATA
            {startup_data}

            Return ONLY valid JSON with this schema:
            {{
              "technical_summary": "2-4 sentence technical interpretation",
              "architecture_feasibility": {{
                "assessment": "Low/Medium/High",
                "reasoning": "why"
              }},
              "scalability_outlook": {{
                "assessment": "Low/Medium/High",
                "reasoning": "why"
              }},
              "security_and_reliability_risks": [
                "risk 1",
                "risk 2",
                "risk 3"
              ],
              "build_plan_90_days": [
                "milestone 1",
                "milestone 2",
                "milestone 3"
              ],
              "tech_due_diligence_questions": [
                "question 1",
                "question 2",
                "question 3"
              ],
              "technical_verdict": "Go/Pivot/No-Go",
              "confidence": "Low/Medium/High"
            }}
            """

            final_prompt = prompt.format(startup_data=startup_data)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": final_prompt}],
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content

        except Exception as e:
            return json.dumps(
                {
                    "error": f"Technical lead analysis failed: {str(e)}",
                    "technical_verdict": "Pivot",
                    "confidence": "Low",
                }
            )

    def analyze_and_save(self, startup):
        result = self.analyze(startup)
        parsed_result = self._coerce_json(result)

        identity = startup.get("identity", {})
        startup_name = identity.get("name", "unknown_startup")
        startup_slug = self._slugify(startup_name)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"{startup_slug}_technical_analysis.json"

        payload = {
            "agent": "Technical Lead",
            "startup_identity": identity,
            "analysis": parsed_result,
        }

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

        return {"output_path": str(output_path), "analysis": parsed_result}

    def _local_analysis(self, startup):
        identity = startup.get("identity", {})
        business = startup.get("business", {})
        description = str(business.get("description", "Unknown")).lower()
        sector = str(identity.get("sector", "Unknown")).lower()

        scalability = "Medium"
        feasibility = "Medium"
        verdict = "Pivot"

        if any(x in description for x in ["ai", "ml", "platform", "api", "automation"]):
            scalability = "High"
        if sector in {"biotech", "hardware", "aerospace", "deeptech"}:
            feasibility = "Low"
            verdict = "No-Go"
        elif scalability == "High":
            verdict = "Go"

        result = {
            "technical_summary": "Local heuristic analysis based on sector and description text.",
            "architecture_feasibility": {
                "assessment": feasibility,
                "reasoning": "Estimated from sector complexity and available description only.",
            },
            "scalability_outlook": {
                "assessment": scalability,
                "reasoning": "Estimated from product keywords and likely software leverage.",
            },
            "security_and_reliability_risks": [
                "No explicit architecture diagram or deployment model provided.",
                "No explicit security posture, compliance scope, or threat model provided.",
                "No reliability SLO/SLA, incident, or observability plan described.",
            ],
            "build_plan_90_days": [
                "Define system architecture and non-functional requirements.",
                "Ship MVP with telemetry, authentication, and CI/CD guardrails.",
                "Run load and security tests, then prioritize remediation backlog.",
            ],
            "tech_due_diligence_questions": [
                "What does your reference architecture and data flow look like?",
                "How will you handle security, privacy, and compliance from day one?",
                "What are your scalability bottlenecks and mitigation plan?",
            ],
            "technical_verdict": verdict,
            "confidence": "Low",
        }
        return json.dumps(result, indent=2)

    @staticmethod
    def _slugify(value):
        lowered = str(value).strip().lower()
        slug = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
        return slug or "unknown_startup"

    @staticmethod
    def _coerce_json(value):
        if isinstance(value, dict):
            return value
        try:
            return json.loads(value)
        except Exception:
            return {"raw_response": str(value)}
