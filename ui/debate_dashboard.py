import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_committee_pipeline import run_committee_pipeline

COMMITTEE_DIR = PROJECT_ROOT / "data" / "committee_pipeline"
MOD_EVAL_DIR = PROJECT_ROOT / "data" / "moderator" / "evaluation"
FIG_DIR = MOD_EVAL_DIR / "figures"


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _decision_badge(decision: str) -> str:
    if decision == "Go":
        return "🟢 Go"
    if decision == "No-Go":
        return "🔴 No-Go"
    return "🟡 Pivot"


def _latest_committee_files():
    return sorted(COMMITTEE_DIR.glob("*_committee_pipeline.json"))


def _build_startup_payload(
    name: str,
    sector: str,
    description: str,
    location: str,
    model: str,
    target_customer: str,
    pricing: str,
    traction: str,
    founders: str,
    revenue: str,
    burn_rate: str,
    funding: str,
    runway: str,
    employee_count: str,
):
    return {
        "name": name.strip(),
        "sector": sector.strip(),
        "description": description.strip(),
        "identity": {
            "name": name.strip(),
            "sector": sector.strip(),
            "location": location.strip() or "Unknown",
        },
        "business": {
            "description": description.strip(),
            "model": model.strip() or "Unknown",
            "problem": "Unknown",
            "solution": "Unknown",
            "target_customer": target_customer.strip() or "Unknown",
            "pricing": pricing.strip() or "Unknown",
            "traction": traction.strip() or "Unknown",
        },
        "team": {
            "founders": founders.strip() or "Unknown",
        },
        "finances": {
            "revenue": revenue.strip() or "Unknown",
            "burn_rate": burn_rate.strip() or "Unknown",
            "funding": funding.strip() or "Unknown",
            "runway": runway.strip() or "Unknown",
            "employee_count": employee_count.strip() or "Unknown",
        },
    }


def render_run_panel():
    st.sidebar.subheader("Run New Startup")
    with st.sidebar.form("new_run_form", clear_on_submit=False):
        name = st.text_input("Startup name", value="NovaPay")
        sector = st.text_input("Sector", value="fintech")
        description = st.text_area(
            "Description",
            value="B2B payment automation platform for small businesses.",
            height=80,
        )
        location = st.text_input("Location", value="US")
        model = st.text_input("Business model", value="SaaS")
        target_customer = st.text_input("Target customer", value="US SMBs with 10-200 employees")
        pricing = st.text_input("Pricing", value="$299/month + usage tiers")
        traction = st.text_input("Traction", value="8 pilot customers")
        founders = st.text_input("Founders", value="1 fintech engineer + 1 SMB operator")
        revenue = st.text_input("Revenue", value="1200000")
        burn_rate = st.text_input("Burn rate", value="85000")
        funding = st.text_input("Funding", value="2000000")
        runway = st.text_input("Runway", value="24 months")
        employee_count = st.text_input("Employee count", value="12")
        run_clicked = st.form_submit_button("Run full debate")

    if not run_clicked:
        return None

    if not name.strip() or not sector.strip() or not description.strip():
        st.sidebar.error("Startup name, sector, and description are required.")
        return None

    startup_payload = _build_startup_payload(
        name=name,
        sector=sector,
        description=description,
        location=location,
        model=model,
        target_customer=target_customer,
        pricing=pricing,
        traction=traction,
        founders=founders,
        revenue=revenue,
        burn_rate=burn_rate,
        funding=funding,
        runway=runway,
        employee_count=employee_count,
    )
    with st.spinner("Running all agents, debate round, and moderator synthesis..."):
        result = run_committee_pipeline(startup_payload)
    st.sidebar.success(f"Run complete: {result.get('decision', 'unknown')}")
    output_path = result.get("output_path")
    return Path(output_path) if output_path else None


def _metric_block(label: str, value: str):
    st.markdown(
        f"""
        <div style="padding:10px 12px;border:1px solid #2d2d2d;border-radius:10px;background:#111111;">
            <div style="font-size:12px;color:#b3b3b3;">{label}</div>
            <div style="font-size:18px;font-weight:600;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header(payload: dict):
    startup = payload.get("startup", {})
    moderator = payload.get("moderator_output", {})
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _metric_block("Startup", str(startup.get("name", "Unknown")))
    with col2:
        _metric_block("Sector", str(startup.get("sector", "Unknown")))
    with col3:
        _metric_block("Final Decision", _decision_badge(str(moderator.get("final_decision", "Pivot"))))
    with col4:
        _metric_block("Confidence", str(moderator.get("confidence", "Low")))


def render_startup_snapshot(payload: dict):
    startup = payload.get("startup", {})
    business = startup.get("business", {})
    finances = startup.get("finances", {})
    st.subheader("Startup Snapshot")
    st.write(str(business.get("description", startup.get("description", "No description provided."))))
    c1, c2, c3 = st.columns(3)
    c1.metric("Revenue", str(finances.get("revenue", "Unknown")))
    c2.metric("Burn Rate", str(finances.get("burn_rate", "Unknown")))
    c3.metric("Funding", str(finances.get("funding", "Unknown")))


def render_agent_cards(payload: dict):
    committee = payload.get("committee_inputs", [])
    st.subheader("Agent Positions")
    for row in committee:
        if not isinstance(row, dict):
            continue
        with st.expander(f"{row.get('agent', 'Agent')} · {_decision_badge(str(row.get('decision', 'Pivot')))}"):
            st.write(row.get("summary", "No summary"))
            st.markdown("**Core thesis**")
            st.write(row.get("debate", {}).get("core_thesis", "No thesis"))
            left, right = st.columns(2)
            with left:
                st.markdown("**Key strengths**")
                for item in row.get("key_strengths", [])[:4]:
                    st.write(f"- {item}")
            with right:
                st.markdown("**Key risks**")
                for item in row.get("key_risks", [])[:4]:
                    st.write(f"- {item}")


def render_debate_round(payload: dict):
    debate_round = payload.get("moderator_output", {}).get("debate_round", {})
    st.subheader("Debate Round")
    st.caption("Round 1 rebuttals before final moderator synthesis")

    rebuttals = debate_round.get("rebuttals", [])
    if not rebuttals:
        st.info("No rebuttals found in this run.")
        return

    for r in rebuttals:
        if not isinstance(r, dict):
            continue
        agent = r.get("agent", "Unknown")
        stance = r.get("stance", "Maintain")
        new_decision = r.get("new_decision", "Pivot")
        target = ", ".join(r.get("responds_to", []))
        st.markdown(f"**{agent}** · {stance} · {_decision_badge(str(new_decision))}")
        st.write(f"Responds to: {target}")
        st.write(r.get("rebuttal", "No rebuttal"))
        st.divider()

    shifts = debate_round.get("key_shifts", [])
    if shifts:
        st.markdown("**Key shifts**")
        for item in shifts:
            st.write(f"- {item}")


def render_moderator_summary(payload: dict):
    mod = payload.get("moderator_output", {})
    st.subheader("Moderator Decision")
    st.write(mod.get("decision_summary", "No summary"))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Consensus points**")
        for item in mod.get("consensus_points", []):
            st.write(f"- {item}")
    with c2:
        st.markdown("**Disagreements**")
        for item in mod.get("disagreements", []):
            st.write(f"- {item}")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Top risks**")
        for item in mod.get("top_risks", []):
            st.write(f"- {item}")
    with c4:
        st.markdown("**Required follow-ups**")
        for item in mod.get("required_follow_ups", []):
            st.write(f"- {item}")


def render_eval_panel():
    st.subheader("Moderator Evaluation")
    summary_path = MOD_EVAL_DIR / "moderator_eval_summary.json"
    if not summary_path.exists():
        st.info("No moderator eval summary found yet.")
        return

    summary = _read_json(summary_path)
    c1, c2, c3 = st.columns(3)
    c1.metric("Runs", int(summary.get("num_runs", 0)))
    c2.metric("Avg Disagreement", f"{float(summary.get('avg_disagreement_rate', 0))*100:.1f}%")
    c3.metric("Avg Polarization", f"{float(summary.get('avg_vote_entropy_normalized', 0))*100:.1f}%")

    chart_names = [
        "decision_distribution.png",
        "confidence_distribution.png",
        "disagreement_per_run.png",
        "vote_polarization_per_run.png",
        "sector_disagreement.png",
    ]
    for name in chart_names:
        path = FIG_DIR / name
        if path.exists():
            st.image(str(path), width="stretch")


def main():
    st.set_page_config(page_title="Startup Debate Dashboard", layout="wide")
    st.title("Startup Debate Dashboard")
    st.caption("Committee debate + moderator output viewer")

    new_output = render_run_panel()
    if new_output:
        st.session_state["selected_run"] = str(new_output)

    files = _latest_committee_files()
    if not files:
        st.error("No committee pipeline outputs found yet.")
        return

    default_index = len(files) - 1
    selected_from_state = st.session_state.get("selected_run")
    if selected_from_state:
        for i, file_path in enumerate(files):
            if str(file_path) == selected_from_state:
                default_index = i
                break

    selected = st.sidebar.selectbox(
        "Select startup run",
        files,
        index=default_index,
        format_func=lambda p: p.name,
        key="selected_run_path",
    )
    st.session_state["selected_run"] = str(selected)
    payload = _read_json(selected)

    render_header(payload)
    st.divider()
    render_startup_snapshot(payload)
    st.divider()
    render_agent_cards(payload)
    st.divider()
    render_debate_round(payload)
    st.divider()
    render_moderator_summary(payload)
    st.divider()
    render_eval_panel()


if __name__ == "__main__":
    main()

