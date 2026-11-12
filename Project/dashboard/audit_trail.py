"""The audit trail page: assessed campaigns, the record behind each, model calls and the chat."""

import pandas as pd
import streamlit as st

from config import config
from dashboard.api_client import call_service
from dashboard.ui_components import as_dashboard_time, badge_markdown, page_header

AUDIT_EXPLANATION = """
**Everything is on the record.** Every assessment, every decision (who took it, in which role, when
and why), every document sent to an agency, every language model call and every chat question is
stored in the database, and can be inspected here afterwards.

* **Assessed campaigns**: choose statuses, then select a row to open its full record, including the
  model calls the run made against the number the workflow allows, and how long they took.
* **Clearance chat log**: the latest questions everyone asked the assistant, and how each was
  answered.
"""

STATUS_LABELS = {
    status: status.replace("_", " ").capitalize() for status in config.CAMPAIGN_STATUS_OPTIONS
}


def _render_call_budget(assessment_id: str) -> None:
    """The run's model calls against the count expected, with the total latency and every call."""
    budget = call_service("GET", f"/clearances/{assessment_id}/calls", None, None, None)
    columns = st.columns(3, border=True)
    columns[0].metric("Model calls recorded", budget["call_count"])
    columns[1].metric("Calls expected", budget["expected_call_count"])
    columns[2].metric("Total latency", f"{budget['total_latency_milliseconds'] / 1000:,.1f} s")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Agent": call["agent_name"],
                    "Model": call["model_name"],
                    "Prompt chars": call["prompt_characters"],
                    "Reply chars": call["response_characters"],
                    "Latency (ms)": call["latency_milliseconds"],
                    "Called": as_dashboard_time(call["called_at"]),
                }
                for call in budget["calls"]
            ]
        ),
        hide_index=True,
        width="stretch",
    )


def _render_record(record: dict) -> None:
    """Everything behind one assessment."""
    st.subheader(f"{record['campaign_reference']} · {record['assessment_id']}")
    st.markdown(
        " ".join(
            [
                badge_markdown(record["status"], config.STATUS_BADGE_COLOURS[record["status"]]),
                badge_markdown(
                    f"Recommended {record['recommended_decision']}",
                    config.DECISION_BADGE_COLOURS[record["recommended_decision"]],
                ),
            ]
        )
    )
    st.markdown(
        f"Started by **{record['executive_id']}** on {as_dashboard_time(record['created_at'])}"
        f", asking: *{record['executive_request']}*"
    )
    if record["decision"]:
        st.markdown(
            f"Decided **{record['decision']}** by **{record['decided_by']}** "
            f"({record['decided_by_role']}) on {as_dashboard_time(record['decided_at'])}: "
            f"\"{record['decision_reason']}\""
        )
        st.caption(
            f"Cleared: {', '.join(record['spots_cleared']) or 'none'}  \n"
            f"Withdrawn: {', '.join(record['spots_withdrawn']) or 'none'}"
        )
    else:
        st.info("Waiting for a Clearance Controller's decision.", icon=":material/hourglass_top:")
    st.markdown("**Language model calls**")
    _render_call_budget(record["assessment_id"])
    documents = st.tabs(["Assessment", "Restriction notice", "Clearance certificate"])
    texts = [
        record["assessment_text"],
        record["restriction_notice_text"],
        record["clearance_certificate_text"],
    ]
    for tab, text in zip(documents, texts):
        with tab:
            st.text(text or "None was drafted for this decision.")


def _render_assessments() -> None:
    """Assessments by the statuses chosen, with counts, and the record behind the one selected."""
    chosen_statuses = st.pills(
        "Status",
        config.CAMPAIGN_STATUS_OPTIONS,
        selection_mode="multi",
        default=config.DECIDED_CAMPAIGN_STATUSES,
        key="audit_statuses",
    )
    if not chosen_statuses:
        st.info("Choose at least one status.")
        return
    records = call_service("GET", "/clearances", {"status_value": chosen_statuses}, None, None)
    columns = st.columns(len(chosen_statuses), border=True)
    for column, status in zip(columns, chosen_statuses):
        column.metric(
            STATUS_LABELS[status], sum(1 for record in records if record["status"] == status)
        )
    if not records:
        st.caption("No assessment has these statuses yet.")
        return
    selection = st.dataframe(
        pd.DataFrame(
            [
                {
                    "Campaign": record["campaign_reference"],
                    "Status": record["status"],
                    "Decided": record["decision"],
                    "By": record["decided_by"],
                    "Cleared": record["spots_cleared_count"],
                    "Withdrawn": record["spots_withdrawn_count"],
                    "Model calls": record["llm_call_count"],
                }
                for record in records
            ]
        ),
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key="audit_table",
    )
    st.caption(
        "Select a row to open its record: the decision, the documents sent to the agency and "
        "the language model calls the run recorded."
    )
    if selection.selection.rows:
        with st.container(border=True):
            _render_record(records[selection.selection.rows[0]])


def _render_chat_log() -> None:
    """Everyone's most recent chat turns, newest first."""
    turns = call_service(
        "GET", "/audit/chat-turns", {"turn_limit": config.CHAT_AUDIT_MAX_ROWS}, None, None
    )
    if not turns:
        st.caption("Nobody has asked the clearance assistant anything yet.")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Asked": as_dashboard_time(turn["asked_at"]),
                    "Who": turn["username"],
                    "Role": turn["user_role"],
                    "Type": turn["question_type"],
                    "Campaign": turn["campaign_reference"],
                    "Refused": turn["was_refused"],
                    "Question": turn["question_text"],
                    "Sources": turn["source_documents"],
                }
                for turn in turns
            ]
        ),
        hide_index=True,
        width="stretch",
    )


def render_audit_trail_page() -> None:
    """Show the assessments, the record behind each, and the chat log."""
    page_header(
        config.PAGE_AUDIT,
        "Every assessment, decision, document, model call and chat "
        "question, as the database recorded it.",
        AUDIT_EXPLANATION,
    )
    assessments_tab, chat_tab = st.tabs(["Assessed campaigns", "Clearance chat log"])
    with assessments_tab:
        _render_assessments()
    with chat_tab:
        _render_chat_log()
