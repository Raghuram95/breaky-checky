"""Every read and write against the six tables.

Nothing else in the project builds a query. Each function takes an open session from
session_factory, writes and flushes but never commits (the caller commits, so several writes can
form one transaction), and returns plain dictionaries rather than database objects.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from config import config
from persistence.models import (
    AgentRun,
    ChatTurn,
    ClearanceAssessment,
    ControllerDecision,
    LlmCall,
    UserAccount,
    database_now,
)

DECISION_FIELDS = ["decision", "decided_by", "decided_by_role", "decision_reason", "decided_at"]
RUN_FIELDS = ["thread_id", "executive_id", "llm_call_count"]


def _as_record(row: object) -> dict:
    """A database row as a plain dictionary of every column."""
    return {column.name: getattr(row, column.name) for column in row.__table__.columns}


def _get_or_refuse(session: Session, table: type, key: str, description: str):
    """Load one row by its key, refusing a key no row holds."""
    row = session.get(table, key)
    if row is None:
        raise ValueError(f"No {description} has the identifier {key}")
    return row


def insert_user_account(session: Session, account_record: dict) -> None:
    """Create one sign-in account; the password arrives already hashed."""
    session.add(UserAccount(**account_record))
    session.flush()


def fetch_user_account(session: Session, condition: ColumnElement[bool]) -> dict | None:
    """Find one account matching a condition, or None."""
    row = session.scalars(select(UserAccount).where(condition)).first()
    return None if row is None else _as_record(row)


def update_session_token(session: Session, username: str, session_token) -> None:
    """Start a session (a new token replaces any earlier one) or end it (None)."""
    _get_or_refuse(session, UserAccount, username, "account").session_token = session_token
    session.flush()


def insert_agent_run(session: Session, run_record: dict) -> None:
    """Record that a run has started, with no calls counted yet."""
    session.add(AgentRun(**run_record, llm_call_count=0))
    session.flush()


def update_agent_run(session: Session, run_id: str, status_value: str, llm_call_count: int) -> None:
    """Move a run on to its next status and record its calls; stamp the finish when it ends."""
    run = _get_or_refuse(session, AgentRun, run_id, "agent run")
    run.status = status_value
    run.llm_call_count = llm_call_count
    if status_value in config.AGENT_RUN_FINISHED_STATUSES:
        run.finished_at = database_now()
    session.flush()


def insert_llm_call(session: Session, run_id: str, call_record: dict) -> None:
    """Log what one language model call cost, against its run."""
    session.add(LlmCall(run_id=run_id, **call_record))
    session.flush()


def fetch_llm_calls(session: Session, run_id: str) -> list:
    """Every call a run made, in the order the calls were made."""
    rows = session.scalars(
        select(LlmCall).where(LlmCall.run_id == run_id).order_by(LlmCall.call_id)
    )
    return [_as_record(row) for row in rows]


def insert_clearance_assessment(session: Session, assessment_record: dict) -> None:
    """Record the finished assessment, waiting at the gate for its decision."""
    session.add(ClearanceAssessment(**assessment_record))
    session.flush()


def record_clearance_outcome(session: Session, assessment_id: str, outcome_record: dict) -> None:
    """Close an assessment with what its decision cleared, withdrew and sent."""
    assessment = _get_or_refuse(session, ClearanceAssessment, assessment_id, "clearance assessment")
    for column_name, value in outcome_record.items():
        setattr(assessment, column_name, value)
    assessment.status_changed_at = database_now()
    session.flush()


def fetch_assessment_records(session: Session, condition: ColumnElement[bool]) -> list:
    """Assessments matching a condition, each with its run and its decision, newest first."""
    query = (
        select(ClearanceAssessment, AgentRun, ControllerDecision)
        .join(AgentRun, ClearanceAssessment.run_id == AgentRun.run_id)
        .outerjoin(
            ControllerDecision,
            ControllerDecision.assessment_id == ClearanceAssessment.assessment_id,
        )
        .where(condition)
        .order_by(ClearanceAssessment.created_at.desc(), ClearanceAssessment.assessment_id.desc())
    )
    records = []
    for assessment, run, decision in session.execute(query):
        record = _as_record(assessment)
        record.update({field: getattr(run, field) for field in RUN_FIELDS})
        record.update(
            {field: getattr(decision, field) if decision else None for field in DECISION_FIELDS}
        )
        records.append(record)
    return records


def insert_controller_decision(session: Session, decision_record: dict) -> None:
    """Record a Clearance Controller's decision, who took it and why."""
    session.add(ControllerDecision(**decision_record))
    session.flush()


def insert_chat_turn(session: Session, turn_record: dict) -> None:
    """Record one chat question and its answer."""
    session.add(ChatTurn(**turn_record))
    session.flush()


def fetch_chat_turns(session: Session, condition: ColumnElement[bool], turn_limit: int) -> list:
    """At most turn_limit matching chat turns, newest first."""
    rows = session.scalars(
        select(ChatTurn).where(condition).order_by(ChatTurn.turn_id.desc()).limit(turn_limit)
    )
    return [_as_record(row) for row in rows]
