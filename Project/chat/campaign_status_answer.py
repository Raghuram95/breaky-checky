"""Answering a question about one named campaign from its record, across the ownership line.

No language model is involved: the answer is assembled from the database in config's own words.
  * An Agency Liaison hears only about their own agencies' campaigns: the status, when it last
    changed and, once decided, the decision, the spot counts and the documents sent. Another
    agency's campaign is answered exactly like a reference that does not exist.
  * The desk (Clearance Executives and Controllers) also hears the full record behind it.
"""

import pandas as pd
from sqlalchemy.orm import Session

from config import config
from persistence.models import ClearanceAssessment
from persistence.repository import fetch_assessment_records


def _desk_record_lines(record: dict) -> list:
    """The assessment's record, which only the desk's roles hear."""
    breaches = ", ".join(f"{code} {count}" for code, count in record["breaches_by_rule"].items())
    breaching = (
        "; ".join(
            f"{spot['spot_reference']} on {spot['channel']}, {spot['broadcast_day']} "
            f"{spot['break_time']}, "
            + " and ".join(
                f"{programme['programme_title']} (index {programme['forecast_under18_index']})"
                for programme in spot["programmes"]
            )
            + f", breaking {', '.join(spot['breach_codes'])}"
            for spot in record["breaching_spots"]
        )
        or "none"
    )
    first = record["first_transmission"]
    return [
        f"Assessment {record['assessment_id']} was started by {record['executive_id']}.",
        f"Copy \"{record['copy_title']}\", clock number {record['clock_number']}, from "
        f"{record['agency_name']} ({record['agency_code']}), "
        f"category {record['product_category']}, "
        f"checked against {record['code_document']}.",
        f"Amendment score {record['amendment_score']}, band {record['amendment_band']}; the notes "
        f"name {record['issue_category']} with weighted agreement {record['issue_agreement']}.",
        f"{record['spots_booked']} spots booked, {record['spots_compliant']} compliant and "
        f"{record['spots_breaching']} breaching; breaches by rule: {breaches}.",
        f"Breaching spots: {breaching}.",
        f"First transmission {first['broadcast_day']} {first['break_time']}; exposure "
        f"{record['exposure_thousands']} thousand under-18 viewers.",
        f"Decisions open: {', '.join(record['decisions_open'])}; recommended: "
        f"{record['recommended_decision']}.",
    ]


def _desk_decision_lines(record: dict) -> list:
    """Who decided, why, and which spots were cleared and withdrawn: the desk only."""
    return [
        f"It was decided by {record['decided_by']}, {record['decided_by_role']}, who wrote: "
        f"\"{record['decision_reason']}\".",
        f"Spots cleared: {', '.join(record['spots_cleared']) or 'none'}. "
        f"Spots withdrawn: {', '.join(record['spots_withdrawn']) or 'none'}.",
    ]


def answer_campaign_status(
    session: Session,
    open_campaigns: pd.DataFrame,
    signed_in_user: dict,
    campaign_reference: str | None,
) -> str:
    """Answer a question about one campaign in the way the asker's role and agencies allow."""
    if campaign_reference is None:
        return config.CHAT_CAMPAIGN_NOT_FOUND_MESSAGE
    rows = open_campaigns[open_campaigns[config.CAMPAIGN_REFERENCE_COLUMN] == campaign_reference]
    if rows.empty:
        return config.CHAT_CAMPAIGN_NOT_FOUND_MESSAGE
    hears_record = signed_in_user["role"] in config.CHAT_RECORD_ROLES
    agency = rows.iloc[0][config.AGENCY_CODE_COLUMN]
    if not hears_record and agency not in signed_in_user["assigned_agencies"]:
        return config.CHAT_CAMPAIGN_NOT_FOUND_MESSAGE

    records = fetch_assessment_records(
        session, ClearanceAssessment.campaign_reference == campaign_reference
    )
    if not records:
        return config.CHAT_NOT_ASSESSED_TEMPLATE.format(campaign_reference=campaign_reference)
    record = records[0]

    lines = [
        config.CHAT_STATUS_TEMPLATE.format(
            campaign_reference=campaign_reference,
            status_phrase=config.CHAT_STATUS_PHRASES[record["status"]],
            changed_at=record["status_changed_at"].strftime(config.STATUS_CHANGE_TIME_FORMAT),
        )
    ]
    if hears_record:
        lines += _desk_record_lines(record)
    if record["decision"] is not None:
        lines.append(
            config.CHAT_DECIDED_TEMPLATE.format(
                decision=record["decision"],
                spots_cleared=record["spots_cleared_count"],
                spots_withdrawn=record["spots_withdrawn_count"],
            )
        )
        if hears_record:
            lines += _desk_decision_lines(record)
        lines.append(config.CHAT_DOCUMENTS_SENT[record["status"]])
    return " ".join(lines)
