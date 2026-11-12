"""The REST service: eleven endpoints in one FastAPI application, with one failure handler.

Every fact the dashboard shows comes from here. When this module loads it takes the models, the
live queue and the knowledge base from the workflow's nodes module (nothing is loaded twice) and
works out every campaign's facts once, with the very tools the workflow uses. Signing in and out
are public; every other endpoint sits behind authorize_request, which decides the area by route.
"""

import hmac
import secrets
import threading
import uuid
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from google.genai import errors as genai_errors
from langgraph.types import Command
from sqlalchemy import true
from sqlalchemy.orm import Session

from access.password_security import hash_password
from access.permissions import authorize_request
from agentic import nodes
from agentic.graph import build_workflow
from api.schemas import (
    AssessmentRequest,
    CallBudgetResponse,
    CampaignDetailResponse,
    CampaignStatusValue,
    CategoryFilter,
    ChatRequest,
    ChatResponse,
    ChatTurnResponse,
    ClearanceRecordResponse,
    DecisionRequest,
    QueueRow,
    SignedInUserResponse,
    SignedOutResponse,
    SignInRequest,
)
from chat.campaign_status_answer import answer_campaign_status
from chat.clearance_assistant import answer_from_documents
from chat.conversation_memory import recall_conversation
from config import config
from knowledge.code_reading import read_category_code
from persistence.database import session_factory
from persistence.models import ChatTurn, ClearanceAssessment, UserAccount
from persistence.repository import (
    fetch_assessment_records,
    fetch_chat_turns,
    fetch_llm_calls,
    fetch_user_account,
    insert_chat_turn,
    update_agent_run,
    update_session_token,
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
)
from tools.schedule_tools import (
    check_campaign_spots,
    place_spot,
    rank_clearance_queue,
    summarise_spot_checks,
)

# ------------------------------------------------------------------------------------------------
# Worked out once, at start-up
# ------------------------------------------------------------------------------------------------

assessment_graph = build_workflow(config.CHECKPOINT_DATABASE_FILE)

category_codes = {
    category: read_category_code(nodes.knowledge_collection, category)
    for category in config.CATEGORY_CODE_DOCUMENTS
}

live_references = list(nodes.open_campaigns[config.CAMPAIGN_REFERENCE_COLUMN])
campaign_profiles = {
    reference: fetch_campaign_profile(nodes.open_campaigns, reference)
    for reference in live_references
}
amendment_facts_by_reference = dict(
    zip(
        live_references,
        build_amendment_facts(
            nodes.amendment_model, nodes.amendment_feature_names, nodes.open_campaigns
        ),
    )
)
campaign_spots_by_reference = {
    reference: fetch_campaign_spots(nodes.booked_spots, reference) for reference in live_references
}
booked_schedules = fetch_campaign_schedules(
    nodes.programme_schedule,
    [spot for spots in campaign_spots_by_reference.values() for spot in spots],
)

spot_results_by_reference = {
    reference: check_campaign_spots(
        [place_spot(spot, booked_schedules) for spot in campaign_spots_by_reference[reference]],
        category_codes[campaign_profiles[reference][config.PRODUCT_CATEGORY_COLUMN]],
    )
    for reference in live_references
}
summaries_by_reference = {
    reference: summarise_spot_checks(results)
    for reference, results in spot_results_by_reference.items()
}
exposures_by_reference = {
    reference: compute_campaign_exposure(results, nodes.programme_forecasts)
    for reference, results in spot_results_by_reference.items()
}

ranked_campaigns = rank_clearance_queue(
    [
        {
            "campaign_reference": reference,
            "agency_name": campaign_profiles[reference][config.AGENCY_NAME_COLUMN],
            "advertiser_name": campaign_profiles[reference][config.ADVERTISER_NAME_COLUMN],
            "copy_title": campaign_profiles[reference][config.COPY_TITLE_COLUMN],
            "product_category": campaign_profiles[reference][config.PRODUCT_CATEGORY_COLUMN],
            "spots_booked": summaries_by_reference[reference]["spots_booked"],
            "spots_breaching": summaries_by_reference[reference]["spots_breaching"],
            "first_transmission": summaries_by_reference[reference]["first_transmission"],
            "exposure": exposures_by_reference[reference],
            **amendment_facts_by_reference[reference],
        }
        for reference in live_references
    ]
)
clearance_queue = [
    dict(entry, queue_position=position) for position, entry in enumerate(ranked_campaigns, start=1)
]
queue_rows_by_reference = {row["campaign_reference"]: row for row in clearance_queue}

assessment_start_locks = {reference: threading.Lock() for reference in live_references}
decision_lock = threading.Lock()

public_router = APIRouter()
guarded_router = APIRouter()

SessionTokenHeader = Annotated[str | None, Header(alias=config.SESSION_TOKEN_HEADER)]
SignedInUser = Annotated[dict, Depends(authorize_request)]


# ------------------------------------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------------------------------------


def _without_gaps(record: dict) -> dict:
    """A profile with every missing value as None, so it serialises as JSON null."""
    return {key: (None if value != value else value) for key, value in record.items()}


def _refuse(status_code: int, detail: str) -> None:
    """Stop the request with an HTTP error and a plain message."""
    raise HTTPException(status_code=status_code, detail=detail)


def _assessment_record(session: Session, assessment_id: str) -> dict:
    """One assessment record by its identifier, or 404."""
    records = fetch_assessment_records(session, ClearanceAssessment.assessment_id == assessment_id)
    if not records:
        _refuse(status.HTTP_404_NOT_FOUND, config.ASSESSMENT_NOT_FOUND_MESSAGE)
    return records[0]


def _thread_settings(thread_id: str) -> dict:
    """The LangGraph settings that select one run's checkpointed thread."""
    return {"configurable": {"thread_id": thread_id}}


def _require_live_campaign(campaign_reference: str) -> None:
    """Refuse with 404 a campaign that is not in the live queue."""
    if campaign_reference not in queue_rows_by_reference:
        _refuse(
            status.HTTP_404_NOT_FOUND,
            config.CAMPAIGN_NOT_IN_QUEUE_MESSAGE.format(campaign_reference=campaign_reference),
        )


# ------------------------------------------------------------------------------------------------
# Signing in and out
# ------------------------------------------------------------------------------------------------


@public_router.post("/auth/signin", response_model=SignedInUserResponse)
def sign_in_user(sign_in_request: SignInRequest):
    """Exchange a username and password for a fresh session token."""
    with session_factory() as session:
        account = fetch_user_account(session, UserAccount.username == sign_in_request.username)
        # The password is hashed even for an unknown username, so both refusals take as long.
        offered_hash = hash_password(
            sign_in_request.password, account["password_salt"] if account else ""
        )
        if account is None or not hmac.compare_digest(offered_hash, account["password_hash"]):
            _refuse(status.HTTP_401_UNAUTHORIZED, config.SIGN_IN_FAILED_MESSAGE)
        session_token = secrets.token_urlsafe(config.SESSION_TOKEN_BYTES)
        update_session_token(session, account["username"], session_token)
        session.commit()
    return {
        **account,
        "session_token": session_token,
        "pages": config.ROLE_PAGE_ACCESS[account["role"]],
        "areas": config.ROLE_AREA_ACCESS[account["role"]],
    }


@public_router.post("/auth/signout", response_model=SignedOutResponse)
def sign_out_user(session_token: SessionTokenHeader = None):
    """End a session; the token can never be used again."""
    with session_factory() as session:
        account = (
            fetch_user_account(session, UserAccount.session_token == session_token)
            if session_token
            else None
        )
        if account is None:
            _refuse(status.HTTP_401_UNAUTHORIZED, config.SIGN_IN_REQUIRED_MESSAGE)
        update_session_token(session, account["username"], None)
        session.commit()
    return {"username": account["username"], "signed_out": True}


# ------------------------------------------------------------------------------------------------
# The queue and one campaign
# ------------------------------------------------------------------------------------------------


@guarded_router.get("/queue", response_model=list[QueueRow])
def list_clearance_queue(product_category: Annotated[CategoryFilter, Query()]):
    """The ranked queue, whole or filtered to one category, positions unchanged."""
    if product_category == config.QUEUE_CATEGORY_ALL:
        return clearance_queue
    return [row for row in clearance_queue if row["product_category"] == product_category]


@guarded_router.get("/queue/{campaign_reference}", response_model=CampaignDetailResponse)
def get_campaign_detail(campaign_reference: str):
    """One campaign with every fact the assessment works from; no model is called."""
    _require_live_campaign(campaign_reference)
    profile = campaign_profiles[campaign_reference]
    campaign_notes = fetch_campaign_notes(nodes.open_campaign_notes, campaign_reference)
    touched = {
        (spot["channel"], spot["broadcast_day"])
        for spot in campaign_spots_by_reference[campaign_reference]
    }
    return {
        "queue_position": queue_rows_by_reference[campaign_reference]["queue_position"],
        "profile": _without_gaps(profile),
        "campaign_notes": campaign_notes,
        "amendment": amendment_facts_by_reference[campaign_reference],
        "issue": build_issue_facts(nodes.issue_model, campaign_notes),
        "code_rules": category_codes[profile[config.PRODUCT_CATEGORY_COLUMN]],
        "schedules": [
            schedule
            for schedule in booked_schedules
            if (schedule["channel"], schedule["broadcast_day"]) in touched
        ],
        "spot_results": spot_results_by_reference[campaign_reference],
        "summary": summaries_by_reference[campaign_reference],
        "exposure": exposures_by_reference[campaign_reference],
    }


# ------------------------------------------------------------------------------------------------
# Assessments and decisions
# ------------------------------------------------------------------------------------------------


@guarded_router.post(
    "/assessments", response_model=ClearanceRecordResponse, status_code=status.HTTP_201_CREATED
)
def start_clearance_assessment(
    assessment_request: AssessmentRequest, signed_in_user: SignedInUser, request: Request
):
    """Run the five assessing agents on one campaign and leave it waiting for a Controller."""
    reference = assessment_request.campaign_reference
    _require_live_campaign(reference)
    with assessment_start_locks[reference]:
        with session_factory() as session:
            if fetch_assessment_records(
                session, ClearanceAssessment.campaign_reference == reference
            ):
                _refuse(
                    status.HTTP_409_CONFLICT,
                    config.CAMPAIGN_ALREADY_ASSESSED_MESSAGE.format(campaign_reference=reference),
                )
        thread_id = str(uuid.uuid4())
        request.state.thread_id = thread_id
        settings = _thread_settings(thread_id)
        assessment_graph.invoke(
            {
                "campaign_reference": reference,
                "executive_request": assessment_request.executive_request,
                "executive_id": signed_in_user["username"],
                "thread_id": thread_id,
                "llm_call_count": 0,
            },
            settings,
        )
        assessment_id = assessment_graph.get_state(settings).values["assessment_id"]
    with session_factory() as session:
        return _assessment_record(session, assessment_id)


@guarded_router.get("/clearances", response_model=list[ClearanceRecordResponse])
def list_clearance_records(status_value: Annotated[list[CampaignStatusValue], Query()]):
    """Every assessment whose status is one of those given, newest first."""
    with session_factory() as session:
        return fetch_assessment_records(session, ClearanceAssessment.status.in_(status_value))


def _decision_matches(sent: dict, decision_record: dict) -> bool:
    """True when a decision sent again is the same decision, person and reason as before."""
    fields = ["decision", "decided_by", "decision_reason"]
    return all(sent.get(field) == decision_record[field] for field in fields)


@guarded_router.post("/clearances/{assessment_id}/decision", response_model=ClearanceRecordResponse)
def submit_controller_decision(
    assessment_id: str, decision_request: DecisionRequest, signed_in_user: SignedInUser
):
    """Record a Controller's decision and finish the run down the path it calls for.

    Every refusal is settled under decision_lock before the run resumes, and the lock is held
    while the run finishes, so two simultaneous decisions can never both be recorded.
    """
    decision_record = {
        "decision": decision_request.decision,
        "decided_by": signed_in_user["display_name"],
        "decided_by_role": signed_in_user["role"],
        "decision_reason": decision_request.decision_reason,
    }
    with decision_lock:
        with session_factory() as session:
            record = _assessment_record(session, assessment_id)
        if record["executive_id"] == signed_in_user["username"]:
            _refuse(status.HTTP_403_FORBIDDEN, config.OWN_ASSESSMENT_REFUSED_MESSAGE)
        if record["status"] != config.CAMPAIGN_STATUS_PENDING:
            _refuse(status.HTTP_409_CONFLICT, config.ALREADY_DECIDED_MESSAGE)

        settings = _thread_settings(record["thread_id"])
        already_sent = assessment_graph.get_state(settings).values.get("controller_decision")
        if already_sent:
            # A decision whose documents failed to draft: only the same decision can finish it.
            if not _decision_matches(already_sent, decision_record):
                _refuse(
                    status.HTTP_409_CONFLICT,
                    config.DECISION_IN_PROGRESS_MESSAGE.format(
                        decided_by=already_sent["decided_by"], decision=already_sent["decision"]
                    ),
                )
            resume_with = None
        else:
            if decision_request.decision not in record["decisions_open"]:
                _refuse(
                    status.HTTP_409_CONFLICT,
                    config.DECISION_NOT_OPEN_MESSAGE.format(
                        decision=decision_request.decision,
                        spots_breaching=record["spots_breaching"],
                        spots_booked=record["spots_booked"],
                        spots_compliant=record["spots_compliant"],
                    ),
                )
            resume_with = Command(resume=decision_record)
        assessment_graph.invoke(resume_with, settings)
    with session_factory() as session:
        return _assessment_record(session, assessment_id)


@guarded_router.get("/clearances/{assessment_id}/calls", response_model=CallBudgetResponse)
def get_call_budget(assessment_id: str):
    """A run's language model calls against the count expected at its stage."""
    with session_factory() as session:
        record = _assessment_record(session, assessment_id)
        calls = fetch_llm_calls(session, record["run_id"])
    expected = (
        config.LLM_CALLS_BEFORE_DECISION
        if record["decision"] is None
        else config.LLM_CALLS_BY_DECISION[record["decision"]]
    )
    return {
        "assessment_id": assessment_id,
        "run_id": record["run_id"],
        "call_count": record["llm_call_count"],
        "expected_call_count": expected,
        "total_latency_milliseconds": sum(call["latency_milliseconds"] for call in calls),
        "calls": calls,
    }


# ------------------------------------------------------------------------------------------------
# The clearance chat
# ------------------------------------------------------------------------------------------------


@guarded_router.post("/chat/ask", response_model=ChatResponse)
def ask_clearance_assistant(chat_request: ChatRequest, signed_in_user: SignedInUser):
    """Answer one question in one model call, from the documents or the campaign's record."""
    question = chat_request.question
    with session_factory() as session:
        conversation = recall_conversation(session, signed_in_user["username"], question)
        reply = answer_from_documents(nodes.knowledge_collection, question, conversation)
        if reply["question_type"] == config.CHAT_TYPE_CAMPAIGN_STATUS:
            reply["answer"] = answer_campaign_status(
                session, nodes.open_campaigns, signed_in_user, reply["campaign_reference"]
            )
        insert_chat_turn(
            session,
            {
                "username": signed_in_user["username"],
                "user_role": signed_in_user["role"],
                "question_text": question,
                "question_type": reply["question_type"],
                "campaign_reference": reply["campaign_reference"],
                "answer_text": reply["answer"],
                "source_documents": ", ".join(reply["source_documents"]),
                "closest_distance": reply["closest_distance"],
                "was_refused": reply["was_refused"],
            },
        )
        session.commit()
    return reply


@guarded_router.get("/chat/turns", response_model=list[ChatTurnResponse])
def list_own_chat_turns(
    signed_in_user: SignedInUser,
    turn_limit: Annotated[int, Query(ge=1, le=config.CHAT_AUDIT_MAX_ROWS)],
):
    """The asker's own latest chat turns, newest first."""
    with session_factory() as session:
        return fetch_chat_turns(
            session, ChatTurn.username == signed_in_user["username"], turn_limit
        )


@guarded_router.get("/audit/chat-turns", response_model=list[ChatTurnResponse])
def list_recent_chat_turns(turn_limit: Annotated[int, Query(ge=1, le=config.CHAT_AUDIT_MAX_ROWS)]):
    """Everyone's latest chat turns, newest first, for the audit trail."""
    with session_factory() as session:
        return fetch_chat_turns(session, true(), turn_limit)


# ------------------------------------------------------------------------------------------------
# The failure handler and the application
# ------------------------------------------------------------------------------------------------


def report_failed_request(request: Request, failure: Exception):
    """Turn any failure into one plain 502 reply, and close the run a failed request opened."""
    thread_id = getattr(request.state, "thread_id", None)
    if thread_id:
        run_id = assessment_graph.get_state(_thread_settings(thread_id)).values.get("run_id")
        if run_id:
            with session_factory() as session:
                calls = fetch_llm_calls(session, run_id)
                update_agent_run(session, run_id, config.AGENT_RUN_STATUS_FAILED, len(calls))
                session.commit()
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"detail": f"{request.method} {request.url.path} failed: {failure}"},
    )


service_application = FastAPI(title=config.API_TITLE)
service_application.include_router(public_router)
service_application.include_router(guarded_router, dependencies=[Depends(authorize_request)])
for failure_type in (ValueError, genai_errors.APIError, httpx.HTTPError):
    service_application.add_exception_handler(failure_type, report_failed_request)
