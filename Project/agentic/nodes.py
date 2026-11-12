"""The ten workflow steps and the router out of the Clearance Controller's gate.

Each node takes the state and returns only the keys it wrote. A node that calls an agent records
the call against the run in the same database transaction as its other writes, as soon as the
agent returns; a reply the agent refuses raises before anything is recorded.

The models, the live queue and the knowledge base are loaded once, when this module is first
imported, and shared by every node and by the REST service.
"""

import secrets

import pandas as pd
from langgraph.types import interrupt

from agentic.agents import (
    advise_on_spot_schedule,
    assess_copy_risk,
    draft_clearance_certificate,
    draft_restriction_notice,
    identify_issues,
    plan_assessment,
    write_clearance_assessment,
)
from agentic.state import ClearanceAssessmentState
from config import config
from data_pipeline.dataset_loader import load_raw_dataset
from knowledge.code_reading import read_category_code
from knowledge.vector_index import open_vector_index
from modeling.model_registry import load_feature_schema, load_model
from persistence.database import session_factory
from persistence.repository import (
    insert_agent_run,
    insert_clearance_assessment,
    insert_controller_decision,
    insert_llm_call,
    record_clearance_outcome,
    update_agent_run,
)
from tools.campaign_data_tools import (
    fetch_campaign_notes,
    fetch_campaign_profile,
    fetch_campaign_schedules,
    fetch_campaign_spots,
)
from tools.prediction_tools import (
    build_amendment_facts,
    build_issue_facts,
    compute_campaign_exposure,
    forecast_programme_audiences,
)
from tools.retrieval_tools import (
    read_executive_checks,
    read_rebooking_steps,
    read_standing_restrictions,
)
from tools.schedule_tools import (
    check_campaign_spots,
    compute_clearance_outcome,
    place_spot,
    summarise_spot_checks,
)

# Loaded once, at import, and shared by every node and by the service.
amendment_model = load_model(config.AMENDMENT_MODEL_FILE)
exposure_model = load_model(config.EXPOSURE_MODEL_FILE)
issue_model = load_model(config.ISSUE_MODEL_FILE)
amendment_feature_names = load_feature_schema(config.AMENDMENT_SCHEMA_FILE)
exposure_feature_names = load_feature_schema(config.EXPOSURE_SCHEMA_FILE)
open_campaigns = load_raw_dataset(config.OPEN_CAMPAIGNS_FILE)
booked_spots = load_raw_dataset(config.BOOKED_SPOTS_FILE)
programme_schedule = load_raw_dataset(config.PROGRAMME_SCHEDULE_FILE)
open_campaign_notes = load_raw_dataset(config.OPEN_CAMPAIGN_NOTES_FILE)
knowledge_collection = open_vector_index(config.VECTOR_STORE_DIRECTORY)
programme_forecasts = forecast_programme_audiences(
    exposure_model, exposure_feature_names, programme_schedule
)


def new_identifier(prefix: str) -> str:
    """A prefix followed by random upper-case hexadecimal characters, e.g. RUN-3F9A0C12B7DE."""
    return prefix + secrets.token_hex(config.IDENTIFIER_HEX_LENGTH // 2).upper()


def labelled_text(fields: dict) -> str | None:
    """An agent's fields written as 'LABEL: text' joined by blank lines; None when empty."""
    if not fields:
        return None
    return "\n\n".join(f"{label}: {text}" for label, text in fields.items())


def _record_call(run_id: str, call_record: dict) -> None:
    """Record one agent's call against its run, in a transaction of its own."""
    with session_factory() as session:
        insert_llm_call(session, run_id, call_record)
        session.commit()


def load_campaign_context_node(state: ClearanceAssessmentState) -> dict:
    """Open the run and gather every fact about the campaign before any agent runs."""
    reference = state["campaign_reference"]
    campaign_profile = fetch_campaign_profile(open_campaigns, reference)
    campaign_spots = fetch_campaign_spots(booked_spots, reference)
    campaign_notes = fetch_campaign_notes(open_campaign_notes, reference)
    amendment_facts = build_amendment_facts(
        amendment_model, amendment_feature_names, pd.DataFrame([campaign_profile])
    )[0]
    issue_facts = build_issue_facts(issue_model, campaign_notes)

    run_id = new_identifier(config.RUN_ID_PREFIX)
    with session_factory() as session:
        insert_agent_run(
            session,
            {
                "run_id": run_id,
                "thread_id": state["thread_id"],
                "campaign_reference": reference,
                "executive_id": state["executive_id"],
                "executive_request": state["executive_request"],
                "status": config.AGENT_RUN_STATUS_RUNNING,
            },
        )
        session.commit()
    return {
        "run_id": run_id,
        "assessment_id": new_identifier(config.ASSESSMENT_ID_PREFIX),
        "campaign_profile": campaign_profile,
        "campaign_spots": campaign_spots,
        "campaign_notes": campaign_notes,
        "amendment_facts": amendment_facts,
        "issue_facts": issue_facts,
    }


def plan_assessment_node(state: ClearanceAssessmentState) -> dict:
    """Ask the intake planner what the assessment has to establish."""
    reply = plan_assessment(state["campaign_profile"], state["executive_request"])
    _record_call(state["run_id"], reply["call_record"])
    return {"assessment_plan": reply["assessment_plan"], "llm_call_count": 1}


def assess_copy_risk_node(state: ClearanceAssessmentState) -> dict:
    """Ask the copy risk analyst to explain the amendment score."""
    reply = assess_copy_risk(
        state["amendment_facts"], state["campaign_profile"], state["assessment_plan"]
    )
    _record_call(state["run_id"], reply["call_record"])
    return {"copy_assessment": reply["copy_assessment"], "llm_call_count": 1}


def identify_issues_node(state: ClearanceAssessmentState) -> dict:
    """Ask the issue analyst how much weight the notes deserve, with the issue guide's checks."""
    executive_checks = read_executive_checks(
        knowledge_collection, state["issue_facts"]["issue_category"]
    )
    reply = identify_issues(
        state["issue_facts"],
        state["campaign_notes"],
        executive_checks,
        state["assessment_plan"],
        state["copy_assessment"],
    )
    _record_call(state["run_id"], reply["call_record"])
    return {"issue_analysis": reply["issue_analysis"], "llm_call_count": 1}


def check_spot_schedule_node(state: ClearanceAssessmentState) -> dict:
    """Test every spot against its category's own code, then ask the advisor to explain."""
    profile = state["campaign_profile"]
    code_rules = read_category_code(knowledge_collection, profile[config.PRODUCT_CATEGORY_COLUMN])
    schedules = fetch_campaign_schedules(programme_schedule, state["campaign_spots"])
    placements = [place_spot(spot, schedules) for spot in state["campaign_spots"]]
    spot_results = check_campaign_spots(placements, code_rules)
    schedule_facts = {
        "code_rules": code_rules,
        "spot_results": spot_results,
        "summary": summarise_spot_checks(spot_results),
    }
    reply = advise_on_spot_schedule(schedule_facts, profile, state["assessment_plan"])
    _record_call(state["run_id"], reply["call_record"])
    return {
        "schedule_facts": schedule_facts,
        "schedule_advice": reply["schedule_advice"],
        "llm_call_count": 1,
    }


def breaching_spot_records(spot_results: list[dict]) -> list:
    """The breaching spots as the assessment stores them."""
    return [
        {
            "spot_reference": result["spot_reference"],
            "channel": result["channel"],
            "broadcast_day": result["broadcast_day"],
            "break_time": result["break_time"],
            "programmes": [
                {
                    "programme_title": programme["programme_title"],
                    "forecast_under18_index": programme["forecast_under18_index"],
                }
                for programme in result["programmes"]
            ],
            "breach_codes": result["breach_codes"],
        }
        for result in spot_results
        if result["breach_codes"]
    ]


def compose_clearance_assessment_node(state: ClearanceAssessmentState) -> dict:
    """Work out the exposure, have the assessment written, and record it for the Controller."""
    profile, schedule_facts = state["campaign_profile"], state["schedule_facts"]
    summary = schedule_facts["summary"]
    exposure = compute_campaign_exposure(schedule_facts["spot_results"], programme_forecasts)
    reply = write_clearance_assessment(
        profile,
        state["executive_request"],
        state["amendment_facts"],
        state["issue_facts"],
        schedule_facts,
        exposure,
        state["copy_assessment"],
        state["issue_analysis"],
        state["schedule_advice"],
    )
    written = reply["clearance_assessment"]
    with session_factory() as session:
        insert_llm_call(session, state["run_id"], reply["call_record"])
        insert_clearance_assessment(
            session,
            {
                "assessment_id": state["assessment_id"],
                "run_id": state["run_id"],
                "campaign_reference": state["campaign_reference"],
                "agency_code": profile[config.AGENCY_CODE_COLUMN],
                "agency_name": profile[config.AGENCY_NAME_COLUMN],
                "advertiser_name": profile[config.ADVERTISER_NAME_COLUMN],
                "copy_title": profile[config.COPY_TITLE_COLUMN],
                "clock_number": profile[config.CLOCK_NUMBER_COLUMN],
                "product_category": profile[config.PRODUCT_CATEGORY_COLUMN],
                "code_document": schedule_facts["code_rules"]["document_name"],
                "executive_request": state["executive_request"],
                "amendment_score": state["amendment_facts"]["amendment_score"],
                "amendment_band": state["amendment_facts"]["amendment_band"],
                "issue_category": state["issue_facts"]["issue_category"],
                "issue_agreement": state["issue_facts"]["weighted_agreement"],
                "spots_booked": summary["spots_booked"],
                "spots_compliant": summary["spots_compliant"],
                "spots_breaching": summary["spots_breaching"],
                "breaches_by_rule": summary["breaches_by_rule"],
                "breaching_spots": breaching_spot_records(schedule_facts["spot_results"]),
                "first_transmission": summary["first_transmission"],
                "exposure_thousands": exposure,
                "decisions_open": summary["decisions_open"],
                "assessment_plan_text": labelled_text(state["assessment_plan"]),
                "assessment_text": written["ASSESSMENT"],
                "recommended_decision": written["RECOMMENDED DECISION"],
                "status": config.CAMPAIGN_STATUS_PENDING,
            },
        )
        update_agent_run(
            session,
            state["run_id"],
            config.AGENT_RUN_STATUS_AWAITING_DECISION,
            state["llm_call_count"] + 1,
        )
        session.commit()
    return {"llm_call_count": 1}


def request_controller_decision_node(state: ClearanceAssessmentState) -> dict:
    """Pause until a Clearance Controller decides, then work out what the decision clears.

    interrupt() checkpoints the run and stops it here; the service resumes it later, possibly in
    another process after a restart, with the controller's decision record. No model is called.
    """
    controller_decision = interrupt({"assessment_id": state["assessment_id"]})
    clearance_outcome = compute_clearance_outcome(
        controller_decision["decision"], state["schedule_facts"]["spot_results"]
    )
    return {
        "controller_decision": controller_decision,
        "clearance_outcome": clearance_outcome,
        "restriction_notice": {},
        "clearance_certificate": {},
    }


def route_controller_decision(state: ClearanceAssessmentState) -> str:
    """Send the run down the path its decision calls for; the value is the decision itself."""
    return state["controller_decision"]["decision"]


def draft_restriction_notice_node(state: ClearanceAssessmentState) -> dict:
    """Have the restriction notice written, with the code's re-booking steps."""
    profile = state["campaign_profile"]
    rebooking_steps = read_rebooking_steps(
        knowledge_collection, profile[config.PRODUCT_CATEGORY_COLUMN]
    )
    reply = draft_restriction_notice(
        profile, state["schedule_facts"], state["clearance_outcome"], rebooking_steps
    )
    _record_call(state["run_id"], reply["call_record"])
    return {"restriction_notice": reply["restriction_notice"], "llm_call_count": 1}


def draft_clearance_certificate_node(state: ClearanceAssessmentState) -> dict:
    """Have the clearance certificate written, with the code's standing restrictions."""
    profile = state["campaign_profile"]
    standing_restrictions = read_standing_restrictions(
        knowledge_collection, profile[config.PRODUCT_CATEGORY_COLUMN]
    )
    reply = draft_clearance_certificate(
        profile,
        state["schedule_facts"],
        state["clearance_outcome"],
        standing_restrictions,
        state["restriction_notice"],
    )
    _record_call(state["run_id"], reply["call_record"])
    return {"clearance_certificate": reply["clearance_certificate"], "llm_call_count": 1}


def record_clearance_outcome_node(state: ClearanceAssessmentState) -> dict:
    """Close the campaign: record the decision, what it cleared and withdrew, and what was sent."""
    decision, outcome = state["controller_decision"], state["clearance_outcome"]
    with session_factory() as session:
        insert_controller_decision(
            session,
            {
                "assessment_id": state["assessment_id"],
                "run_id": state["run_id"],
                "decision": decision["decision"],
                "decided_by": decision["decided_by"],
                "decided_by_role": decision["decided_by_role"],
                "decision_reason": decision["decision_reason"],
            },
        )
        record_clearance_outcome(
            session,
            state["assessment_id"],
            {
                "status": config.CAMPAIGN_STATUS_BY_DECISION[decision["decision"]],
                "spots_cleared": outcome["spots_cleared"],
                "spots_withdrawn": outcome["spots_withdrawn"],
                "spots_cleared_count": outcome["spots_cleared_count"],
                "spots_withdrawn_count": outcome["spots_withdrawn_count"],
                "restriction_notice_text": labelled_text(state["restriction_notice"]),
                "clearance_certificate_text": labelled_text(state["clearance_certificate"]),
            },
        )
        update_agent_run(
            session, state["run_id"], config.AGENT_RUN_STATUS_COMPLETE, state["llm_call_count"]
        )
        session.commit()
    return {}
