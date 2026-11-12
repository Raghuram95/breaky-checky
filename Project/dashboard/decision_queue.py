"""The decision queue page: assessments waiting at the gate, and the Controller's decision form."""

import pandas as pd
import streamlit as st

from config import config
from dashboard.api_client import call_service, signed_in_user
from dashboard.ui_components import as_dashboard_time, badge_markdown, first_spot_text, page_header

DECISION_EXPLANATION = """
**This is the human-in-the-loop gate.** The AI workflow has stopped here and saved its state to
disk; it will wait as long as it takes, even across a restart. Only a **Clearance Controller** who
did **not** open the assessment may decide it (the Advertising Clearance Policy separates reviewing
a campaign from deciding it).

**The three decisions**
* **CLEAR**: every spot airs. Open only when no spot breaks a rule. One more AI call writes the
  clearance certificate.
* **CLEAR WITH RESTRICTIONS**: compliant spots air, breaching spots are withdrawn. Open when some
  spots break and some comply. Two more AI calls: the restriction notice, then the certificate.
* **REJECT**: nothing airs and nothing is sent; the agency may resubmit. Always open. No AI call.

Which spots are cleared is worked out by code from the spot results, never chosen by an agent.
"""


def _breaching_table(record: dict) -> pd.DataFrame:
    """The assessment's breaching spots, with their programmes and codes."""
    return pd.DataFrame(
        [
            {
                "Breaching spot": spot["spot_reference"],
                "Channel": spot["channel"],
                "Day": spot["broadcast_day"],
                "Time": spot["break_time"],
                "Codes": ", ".join(spot["breach_codes"]),
                "In or between": " and ".join(
                    f"{programme['programme_title']} " f"({programme['forecast_under18_index']})"
                    for programme in spot["programmes"]
                ),
            }
            for spot in record["breaching_spots"]
        ]
    )


def _render_assessment(record: dict) -> None:
    """The chosen assessment in full."""
    st.markdown(
        " ".join(
            [
                badge_markdown(
                    f"Recommended: {record['recommended_decision']}",
                    config.DECISION_BADGE_COLOURS[record["recommended_decision"]],
                ),
                badge_markdown(
                    f"Amendment {record['amendment_score']:.4f}, {record['amendment_band']}",
                    config.AMENDMENT_BAND_BADGE_COLOURS[record["amendment_band"]],
                ),
                badge_markdown(record["product_category"], "gray"),
                badge_markdown(record["issue_category"], "gray"),
            ]
        )
    )
    columns = st.columns(4, border=True)
    columns[0].metric("Spots booked", record["spots_booked"])
    columns[1].metric("Compliant", record["spots_compliant"])
    columns[2].metric("Breaching", record["spots_breaching"])
    columns[3].metric("Exposure", f"{record['exposure_thousands']:,.1f} thousand")
    st.markdown("**The AI-written assessment** *(a recommendation, not a decision)*")
    st.write(record["assessment_text"])
    if record["breaching_spots"]:
        st.dataframe(_breaching_table(record), hide_index=True, width="stretch")
    st.caption(
        f"Read from {record['code_document']} · first spot "
        f"{first_spot_text(record['first_transmission'])} · notes agree "
        f"{record['issue_agreement']:.0%} · assessed by {record['executive_id']} on "
        f"{as_dashboard_time(record['created_at'])}"
    )
    with st.expander("What the opener asked, and what the planner decided to check"):
        st.markdown(f"**Asked:** {record['executive_request']}")
        st.text(record["assessment_plan_text"])


def _render_decision_form(record: dict) -> None:
    """Offer exactly the decisions open, each with what it costs in further model calls."""

    def describe(decision: str) -> str:
        further = config.LLM_CALLS_BY_DECISION[decision] - config.LLM_CALLS_BEFORE_DECISION
        return f"{decision}  ·  {further} further model call{'s' if further != 1 else ''}"

    with st.form("decision_form", border=True):
        decision = st.radio(
            "Decision",
            record["decisions_open"],
            format_func=describe,
            index=record["decisions_open"].index(record["recommended_decision"]),
        )
        reason = st.text_area(
            "Your reason, in your own words", placeholder="This is recorded in the audit trail."
        )
        submitted = st.form_submit_button(
            "Record the decision", type="primary", icon=":material/gavel:"
        )
    if not submitted:
        return
    if len(reason.strip()) < config.DECISION_REASON_MINIMUM_CHARACTERS:
        st.warning(
            f"Write a reason of at least {config.DECISION_REASON_MINIMUM_CHARACTERS} characters."
        )
        return
    decided = call_service(
        "POST",
        f"/clearances/{record['assessment_id']}/decision",
        None,
        {"decision": decision, "decision_reason": reason.strip()},
        "Recording the decision and drafting the documents it calls for...",
    )
    st.success(
        f"{decided['campaign_reference']} is now {decided['status']}: "
        f"{decided['spots_cleared_count']} spots cleared and "
        f"{decided['spots_withdrawn_count']} withdrawn.",
        icon=":material/task_alt:",
    )
    st.caption(
        f"Cleared: {', '.join(decided['spots_cleared']) or 'none'}  \n"
        f"Withdrawn: {', '.join(decided['spots_withdrawn']) or 'none'}"
    )


def render_decision_queue_page() -> None:
    """Show the assessments waiting at the gate and let a Clearance Controller decide one."""
    page_header(
        config.PAGE_DECISION,
        "Assessments the AI workflow has finished, paused and waiting for a Clearance "
        "Controller's decision.",
        DECISION_EXPLANATION,
    )
    pending = call_service(
        "GET", "/clearances", {"status_value": [config.CAMPAIGN_STATUS_PENDING]}, None, None
    )
    if not pending:
        st.info(
            "No assessment is waiting for a decision. Start one from the Campaign detail page.",
            icon=":material/inbox:",
        )
        return
    labels = {
        f"{record['campaign_reference']} · {record['assessment_id']} · recommends "
        f"{record['recommended_decision']}": record
        for record in pending
    }
    chosen = labels[st.selectbox(f"{len(pending)} waiting for a decision", list(labels))]

    assessment_column, decision_column = st.columns([3, 2], gap="large")
    with assessment_column:
        _render_assessment(chosen)
    with decision_column:
        st.subheader("Decide")
        if chosen["executive_id"] == signed_in_user()["username"]:
            st.warning(config.OWN_ASSESSMENT_REFUSED_MESSAGE, icon=":material/block:")
        else:
            _render_decision_form(chosen)
