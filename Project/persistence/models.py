"""The six tables the application writes to, declared with SQLAlchemy's typed ORM mapping.

Timestamps are set by the database itself, in UTC with millisecond precision, so that "newest
first" is decided by the database clock rather than by whichever Python process wrote the row.
Lists and dictionaries (agencies, spot references, breaches by rule) are stored as JSON.
"""

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

DATABASE_NOW = "STRFTIME('%Y-%m-%d %H:%M:%f', 'now')"


def database_now():
    """The current UTC time as the database writes it, for use in an UPDATE."""
    return func.strftime("%Y-%m-%d %H:%M:%f", "now")


def _written_at():
    """A timestamp column the database fills in when the row is written."""
    return mapped_column(DateTime, server_default=text(f"({DATABASE_NOW})"), nullable=False)


class Base(DeclarativeBase):
    """The declarative base every table inherits from; setup creates the tables from it."""


class UserAccount(Base):
    """One person who can sign in, with their role and, for a liaison, their agencies."""

    __tablename__ = "user_accounts"

    username: Mapped[str] = mapped_column(String, primary_key=True)
    display_name: Mapped[str] = mapped_column(String)
    role: Mapped[str] = mapped_column(String)
    assigned_agencies: Mapped[list] = mapped_column(JSON)
    password_hash: Mapped[str] = mapped_column(String)
    password_salt: Mapped[str] = mapped_column(String)
    session_token: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    created_at: Mapped[datetime] = _written_at()


class AgentRun(Base):
    """One run of the assessment workflow, from start through the pause to its end."""

    __tablename__ = "agent_runs"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    thread_id: Mapped[str] = mapped_column(String, index=True)
    campaign_reference: Mapped[str] = mapped_column(String, index=True)
    executive_id: Mapped[str] = mapped_column(String)
    executive_request: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String)
    llm_call_count: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime] = _written_at()
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class ClearanceAssessment(Base):
    """The finished assessment a Clearance Controller decides from, and what the decision did."""

    __tablename__ = "clearance_assessments"

    assessment_id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.run_id"))
    campaign_reference: Mapped[str] = mapped_column(String, index=True)
    agency_code: Mapped[str] = mapped_column(String)
    agency_name: Mapped[str] = mapped_column(String)
    advertiser_name: Mapped[str] = mapped_column(String)
    copy_title: Mapped[str] = mapped_column(String)
    clock_number: Mapped[str] = mapped_column(String)
    product_category: Mapped[str] = mapped_column(String)
    code_document: Mapped[str] = mapped_column(String)
    executive_request: Mapped[str] = mapped_column(Text)
    amendment_score: Mapped[float] = mapped_column(Float)
    amendment_band: Mapped[str] = mapped_column(String)
    issue_category: Mapped[str] = mapped_column(String)
    issue_agreement: Mapped[float] = mapped_column(Float)
    spots_booked: Mapped[int] = mapped_column(Integer)
    spots_compliant: Mapped[int] = mapped_column(Integer)
    spots_breaching: Mapped[int] = mapped_column(Integer)
    breaches_by_rule: Mapped[dict] = mapped_column(JSON)
    breaching_spots: Mapped[list] = mapped_column(JSON)
    first_transmission: Mapped[dict] = mapped_column(JSON)
    exposure_thousands: Mapped[float] = mapped_column(Float)
    decisions_open: Mapped[list] = mapped_column(JSON)
    assessment_plan_text: Mapped[str] = mapped_column(Text)
    assessment_text: Mapped[str] = mapped_column(Text)
    recommended_decision: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, index=True)
    spots_cleared: Mapped[list | None] = mapped_column(JSON, nullable=True)
    spots_withdrawn: Mapped[list | None] = mapped_column(JSON, nullable=True)
    spots_cleared_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    spots_withdrawn_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    restriction_notice_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    clearance_certificate_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = _written_at()
    status_changed_at: Mapped[datetime] = _written_at()


class ControllerDecision(Base):
    """A Clearance Controller's decision on one assessment: what, who, in which role and why."""

    __tablename__ = "controller_decisions"

    decision_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    assessment_id: Mapped[str] = mapped_column(
        ForeignKey("clearance_assessments.assessment_id"), unique=True
    )
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.run_id"))
    decision: Mapped[str] = mapped_column(String)
    decided_by: Mapped[str] = mapped_column(String)
    decided_by_role: Mapped[str] = mapped_column(String)
    decision_reason: Mapped[str] = mapped_column(Text)
    decided_at: Mapped[datetime] = _written_at()


class LlmCall(Base):
    """What one language model call cost: which agent, which model, how long and how much text."""

    __tablename__ = "llm_calls"

    call_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.run_id"), index=True)
    agent_name: Mapped[str] = mapped_column(String)
    model_name: Mapped[str] = mapped_column(String)
    prompt_characters: Mapped[int] = mapped_column(Integer)
    response_characters: Mapped[int] = mapped_column(Integer)
    latency_milliseconds: Mapped[int] = mapped_column(Integer)
    called_at: Mapped[datetime] = _written_at()


class ChatTurn(Base):
    """One question asked of the clearance assistant and the answer it gave."""

    __tablename__ = "chat_turns"

    turn_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String, index=True)
    user_role: Mapped[str] = mapped_column(String)
    question_text: Mapped[str] = mapped_column(Text)
    question_type: Mapped[str] = mapped_column(String)
    campaign_reference: Mapped[str | None] = mapped_column(String, nullable=True)
    answer_text: Mapped[str] = mapped_column(Text)
    source_documents: Mapped[str] = mapped_column(Text)
    closest_distance: Mapped[float] = mapped_column(Float)
    was_refused: Mapped[bool] = mapped_column(Boolean)
    asked_at: Mapped[datetime] = _written_at()
