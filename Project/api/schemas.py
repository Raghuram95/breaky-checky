"""Every request and reply shape of the REST service, declared once with Pydantic.

A request failing its conditions is refused with 422 before any work is done, and a reply carries
exactly the fields listed here: FastAPI filters every reply through its response model.
"""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, StringConstraints

from config import config

CategoryFilter = Literal[tuple(config.QUEUE_CATEGORY_OPTIONS)]
CampaignStatusValue = Literal[tuple(config.CAMPAIGN_STATUS_OPTIONS)]
DecisionValue = Literal[tuple(config.DECISIONS)]

ExecutiveRequestText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=config.EXECUTIVE_REQUEST_MINIMUM_CHARACTERS
    ),
]
DecisionReasonText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=config.DECISION_REASON_MINIMUM_CHARACTERS),
]
QuestionText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=config.CHAT_QUESTION_MINIMUM_CHARACTERS),
]


class SignInRequest(BaseModel):
    """A username and password exchanged for a session token."""

    username: str
    password: str


class SignedInUserResponse(BaseModel):
    """Who signed in, what they may reach, and the token every later request carries."""

    username: str
    display_name: str
    role: str
    assigned_agencies: list[str]
    session_token: str
    pages: list[str]
    areas: list[str]


class SignedOutResponse(BaseModel):
    """Confirmation that a session is over."""

    username: str
    signed_out: bool


class QueueRow(BaseModel):
    """One campaign in the ranked clearance queue."""

    queue_position: int
    campaign_reference: str
    agency_name: str
    advertiser_name: str
    copy_title: str
    product_category: str
    spots_booked: int
    spots_breaching: int
    first_transmission: dict
    exposure: float
    amendment_score: float
    amendment_band: str


class CampaignDetailResponse(BaseModel):
    """One campaign with every fact the assessment works from."""

    queue_position: int
    profile: dict
    campaign_notes: list[dict]
    amendment: dict
    issue: dict
    code_rules: dict
    schedules: list[dict]
    spot_results: list[dict]
    summary: dict
    exposure: float


class AssessmentRequest(BaseModel):
    """The campaign to assess and what the person opening the assessment wants looked at."""

    campaign_reference: str
    executive_request: ExecutiveRequestText


class DecisionRequest(BaseModel):
    """A Clearance Controller's decision and their reason, in their own words."""

    decision: DecisionValue
    decision_reason: DecisionReasonText


class ClearanceRecordResponse(BaseModel):
    """An assessment as stored, with its run and, once decided, its decision."""

    assessment_id: str
    run_id: str
    campaign_reference: str
    agency_code: str
    agency_name: str
    advertiser_name: str
    copy_title: str
    clock_number: str
    product_category: str
    code_document: str
    executive_request: str
    amendment_score: float
    amendment_band: str
    issue_category: str
    issue_agreement: float
    spots_booked: int
    spots_compliant: int
    spots_breaching: int
    breaches_by_rule: dict
    breaching_spots: list[dict]
    first_transmission: dict
    exposure_thousands: float
    decisions_open: list[str]
    assessment_plan_text: str
    assessment_text: str
    recommended_decision: str
    status: str
    spots_cleared: list[str] | None
    spots_withdrawn: list[str] | None
    spots_cleared_count: int | None
    spots_withdrawn_count: int | None
    restriction_notice_text: str | None
    clearance_certificate_text: str | None
    created_at: datetime
    status_changed_at: datetime
    thread_id: str
    executive_id: str
    llm_call_count: int
    decision: str | None
    decided_by: str | None
    decided_by_role: str | None
    decision_reason: str | None
    decided_at: datetime | None


class CallBudgetResponse(BaseModel):
    """A run's language model calls against the number expected."""

    assessment_id: str
    run_id: str
    call_count: int
    expected_call_count: int
    total_latency_milliseconds: int
    calls: list[dict]


class ChatRequest(BaseModel):
    """One question for the clearance assistant."""

    question: QuestionText


class ChatResponse(BaseModel):
    """The assistant's answer, its type and the documents it was given."""

    question_type: str
    campaign_reference: str | None
    answer: str
    source_documents: list[str]
    was_refused: bool


class ChatTurnResponse(BaseModel):
    """One stored chat turn."""

    turn_id: int
    username: str
    user_role: str
    question_text: str
    question_type: str
    campaign_reference: str | None
    answer_text: str
    source_documents: str
    closest_distance: float
    was_refused: bool
    asked_at: datetime
