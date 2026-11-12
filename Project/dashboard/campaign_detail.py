"""The campaign detail page: one campaign's facts, its broadcast days, and the assessment form."""

import altair as alt
import pandas as pd
import streamlit as st

from config import config
from dashboard.api_client import call_service, signed_in_user
from dashboard.clearance_queue import SELECTED_CAMPAIGN_KEY
from dashboard.ui_components import as_dashboard_time, badge_markdown, first_spot_text, page_header

DETAIL_EXPLANATION = """
**What you are looking at.** One campaign, with every fact the AI agents will be given. Nothing on
this page is produced by a language model: it comes from the deterministic tools and the trained
models, through the REST service.

* **Broadcast days** draws each day the campaign is booked on, channel by channel, from 06:00 to
  30:00 (the small hours after midnight belong to the day before). Grey bars are programmes, blue
  bars live sport, orange bars programmes made for children; a green circle is a spot that may air
  and a red cross a spot that breaks a rule.
* **Spots and rules** shows the five rules exactly as the category's code states them, and every
  spot's result with the rule it breaks.
* **Copy and notes** shows the advert itself and the desk's notes, with the issue an NLP model reads
  in each note.
* **Assess** starts the AI assessment: five agents run in turn (one language model call each), then
  the campaign waits for a Clearance Controller. *No spot is cleared and nothing is sent to the
  agency on the word of a model.*
"""

BAR_HEIGHT_SHARE = 0.55
DAY_START = config.BROADCAST_DAY_START_MINUTE
DAY_END = config.BROADCAST_DAY_END_MINUTE
CLOCK_LABEL = (
    "(floor(datum.value / 60) < 10 ? '0' : '') + floor(datum.value / 60) + ':' + "
    "(datum.value % 60 < 10 ? '0' : '') + (datum.value % 60)"
)


def _programme_kind(programme: dict) -> str:
    """Which kind of bar a programme is drawn as."""
    if programme["live_sport"]:
        return config.PROGRAMME_KIND_LIVE_SPORT
    if programme["made_for_children"]:
        return config.PROGRAMME_KIND_CHILDREN
    return config.PROGRAMME_KIND_OTHER


def _programmes_text(programmes: list[dict]) -> str:
    """'Title (index)' for each programme a spot is in or between."""
    return " and ".join(
        f"{programme['programme_title']} ({programme['forecast_under18_index']})"
        for programme in programmes
    )


def _day_chart(schedules: list[dict], spot_results: list[dict]) -> alt.LayerChart:
    """One broadcast day: programmes as bars and spots as points, channel by channel."""
    programmes = pd.DataFrame(
        [
            {
                "Channel": schedule["channel"],
                "Programme": programme["programme_title"],
                "Start": programme["start_minute"],
                "End": programme["end_minute"],
                "Times": f"{programme['start_time']} to {programme['end_time']}",
                "Index": programme["forecast_under18_index"],
                "Kind": _programme_kind(programme),
            }
            for schedule in schedules
            for programme in schedule["programmes"]
        ]
    )
    spots = pd.DataFrame(
        [
            {
                "Channel": result["channel"],
                "Spot": result["spot_reference"],
                "Minute": result["break_minute"],
                "Time": result["break_time"],
                "Breaks": ", ".join(result["breach_codes"]) or "nothing",
                "Result": (
                    config.SPOT_RESULT_BREACHING
                    if result["breach_codes"]
                    else config.SPOT_RESULT_COMPLIANT
                ),
            }
            for result in spot_results
        ]
    )

    x_axis = alt.Axis(
        values=list(range(DAY_START, DAY_END + 1, config.BROADCAST_CLOCK_TICK_MINUTES)),
        labelExpr=CLOCK_LABEL,
        title=None,
        grid=True,
        gridColor=config.CHART_GRID_COLOUR,
        domainColor=config.CHART_AXIS_COLOUR,
        tickColor=config.CHART_AXIS_COLOUR,
        labelColor=config.CHART_LABEL_COLOUR,
    )
    y_axis = alt.Axis(
        title=None,
        labelLimit=config.STRIP_LABEL_WIDTH,
        minExtent=config.STRIP_LABEL_WIDTH,
        labelColor=config.CHART_LABEL_COLOUR,
        domainColor=config.CHART_AXIS_COLOUR,
    )
    x_scale = alt.Scale(domain=[DAY_START, DAY_END])
    kinds = list(config.PROGRAMME_KIND_COLOURS)
    results = list(config.SPOT_RESULT_COLOURS)

    bars = (
        alt.Chart(programmes)
        .mark_bar(
            height=config.STRIP_ROW_HEIGHT * BAR_HEIGHT_SHARE,
            stroke=config.CHART_SURFACE_COLOUR,
            strokeWidth=1,
        )
        .encode(
            x=alt.X("Start:Q", scale=x_scale, axis=x_axis),
            x2="End:Q",
            y=alt.Y("Channel:N", axis=y_axis),
            color=alt.Color(
                "Kind:N",
                scale=alt.Scale(domain=kinds, range=list(config.PROGRAMME_KIND_COLOURS.values())),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=["Programme", "Times", "Index", "Kind"],
        )
    )
    points = (
        alt.Chart(spots)
        .mark_point(size=110, filled=True, opacity=1)
        .encode(
            x=alt.X("Minute:Q", scale=x_scale),
            y=alt.Y("Channel:N"),
            shape=alt.Shape(
                "Result:N",
                scale=alt.Scale(domain=results, range=list(config.SPOT_RESULT_SHAPES.values())),
                legend=alt.Legend(title=None, orient="top"),
            ),
            fill=alt.Fill(
                "Result:N",
                scale=alt.Scale(domain=results, range=list(config.SPOT_RESULT_COLOURS.values())),
                legend=None,
            ),
            stroke=alt.value(config.CHART_SURFACE_COLOUR),
            tooltip=["Spot", "Time", "Result", "Breaks"],
        )
    )
    channel_count = programmes["Channel"].nunique()
    return (
        alt.layer(bars, points)
        .resolve_scale(color="independent", shape="independent")
        .properties(
            height=config.STRIP_ROW_HEIGHT * channel_count, background=config.CHART_SURFACE_COLOUR
        )
        .configure_view(stroke=None)
    )


def _render_broadcast_days(detail: dict) -> None:
    """Draw every broadcast day the campaign is booked on."""
    st.caption(
        "Each broadcast day runs from 06:00 to 30:00 on the broadcast clock: the small hours "
        "after midnight are written 24:00 to 29:59 and belong to the day before. Hover over "
        "a programme or a spot for its details."
    )
    for broadcast_day in config.WEEKDAYS:
        schedules = [
            schedule
            for schedule in detail["schedules"]
            if schedule["broadcast_day"] == broadcast_day
        ]
        if not schedules:
            continue
        st.subheader(broadcast_day)
        day_spots = [
            result for result in detail["spot_results"] if result["broadcast_day"] == broadcast_day
        ]
        st.altair_chart(_day_chart(schedules, day_spots), width="stretch")


def _rule_readings(code_rules: dict) -> dict:
    """Each rule as the service read it, in short words for the rules table."""
    hours, exception = code_rules["restricted_hours"], code_rules["live_sport_exception"]
    window = f"{hours['start']} to {hours['end']}" if hours else "any time of the broadcast day"
    if exception:
        window += (
            f"; live sport centre breaks over {exception['minutes_after_start']} min in and "
            f"{exception['minutes_before_end']} min from the end excepted"
        )
    return {
        config.BREACH_WINDOW: window,
        config.BREACH_INDEX: f"index {code_rules['index_threshold']} or more",
        config.BREACH_CHILDREN: code_rules["children_rule"],
        config.BREACH_FREQUENCY: f"{code_rules['daily_cap']} a channel a broadcast day",
        config.BREACH_SEPARATION: f"{code_rules['separation_minutes']} minutes apart",
    }


def _render_spots_and_rules(detail: dict) -> None:
    """The five rules beside the code's own sentences, then every spot in transmission order."""
    code_rules, summary = detail["code_rules"], detail["summary"]
    st.markdown(f"**The five rules, as {code_rules['document_name']} states them**")
    readings = _rule_readings(code_rules)
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Rule": code,
                    "Read as": readings[code],
                    "The code's own sentence": code_rules["rule_sentences"][code],
                }
                for code in config.BREACH_CODES
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.markdown(
        f"**Every spot, in transmission order** · {summary['spots_compliant']} compliant, "
        f"{summary['spots_breaching']} breaching"
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Spot": result["spot_reference"],
                    "Channel": result["channel"],
                    "Day": result["broadcast_day"],
                    "Time": result["break_time"],
                    "Break": "centre" if result["break_position"] == config.BREAK_IN else "end",
                    "In or between": _programmes_text(result["programmes"]),
                    "Result": ", ".join(result["breach_codes"]) or "compliant",
                }
                for result in detail["spot_results"]
            ]
        ),
        hide_index=True,
        width="stretch",
    )


def _render_copy_and_notes(detail: dict) -> None:
    """The copy, its claims and offers, its history, and its notes with how each was read."""
    profile, amendment, issue = detail["profile"], detail["amendment"], detail["issue"]

    def shown(column_name: str) -> str:
        value = profile.get(column_name)
        return "not applicable" if value is None else str(value)

    copy_card, claims_card, history_card = st.columns(3, border=True)
    copy_card.markdown(
        f"**The copy**\n\n{shown('product_category')}, {shown('sub_category')}  \n"
        f"{shown('copy_length_seconds')} seconds, {shown('supers_count')} supers, legal text on "
        f"screen {shown('legal_text_seconds')} seconds  \nVoiceover {shown('voiceover_flag')} · "
        f"humour {shown('humour_flag')}  \nLead performer {shown('lead_performer_age_band')} · "
        f"celebrity {shown('celebrity_flag')} · sports personality "
        f"{shown('sports_personality_flag')}  \nChild-appeal elements "
        f"{shown('child_appeal_elements_count')}"
    )
    claims_card.markdown(
        f"**Claims and offers**\n\nObjective claims {shown('claims_count')}  \nPrice "
        f"{shown('price_claim_flag')} · health {shown('health_claim_flag')} · environmental "
        f"{shown('environmental_claim_flag')} · comparative {shown('comparative_claim_flag')}  \n"
        f"Offer terms on screen: {shown('offer_terms_shown_flag')}  \nBonus or free bet "
        f"{shown('bonus_or_free_bet_flag')}  \nRepresentative APR on screen: "
        f"{shown('representative_apr_shown_flag')}"
    )
    history_card.markdown(
        f"**Booking and history**\n\n{shown('spots_booked')} spots on {shown('channels_booked')} "
        f"channels  \n{shown('share_before_21_pct')}% before 21:00, "
        f"{shown('share_in_live_sport_pct')}% in live sport  \nFirst spot "
        f"{shown('days_to_first_transmission')} days after submission  \nAdvertiser: "
        f"{shown('advertiser_amendments_past_year')} of {shown('advertiser_campaigns_past_year')} "
        f"campaigns amended in the past year · new advertiser {shown('new_advertiser_flag')}  \n"
        f"Agency: {shown('agency_amendment_rate_pct')}% of its copy amended"
    )

    st.markdown(
        f"The amendment model scores the copy **{amendment['amendment_score']:.4f}**, band "
        f"**{amendment['amendment_band']}**. The issue model names "
        f"**{issue['issue_category']}**, weighted agreement "
        f"{issue['weighted_agreement']:.0%} across the {issue['notes_read']} notes it read, "
        "a legal review counting twice."
    )
    readings = {
        reading["note_sequence"]: reading["issue_category"] for reading in issue["note_readings"]
    }
    for note in detail["campaign_notes"]:
        with st.container(border=True):
            read_as = readings.get(note["note_sequence"], "set aside: too short to read")
            st.caption(f"Note {note['note_sequence']} · {note['note_source']} · read as {read_as}")
            st.write(note["note_text"])


def _render_assess(campaign_reference: str, existing: dict | None) -> None:
    """The assessment form, or the existing assessment when the campaign is already assessed."""
    st.caption(
        "Say what you want looked at. Five agents run in turn, one language model call each, "
        "and the campaign then waits for a Clearance Controller who did not start the "
        "assessment. No spot is cleared and nothing is sent to the agency."
    )
    if existing:
        st.info(
            f"{existing['assessment_id']} was started by {existing['executive_id']} on "
            f"{as_dashboard_time(existing['created_at'])} and stands at {existing['status']}. "
            "A campaign is assessed once.",
            icon=":material/assignment_turned_in:",
        )
        return
    with st.form("assessment_form", border=True):
        request_text = st.text_area(
            "What do you want looked at?",
            placeholder="e.g. Check the offer terms and every booked spot.",
        )
        started = st.form_submit_button(
            "Start the assessment", type="primary", icon=":material/play_arrow:"
        )
    if not started:
        return
    if len(request_text.strip()) < config.EXECUTIVE_REQUEST_MINIMUM_CHARACTERS:
        st.warning(f"Write at least {config.EXECUTIVE_REQUEST_MINIMUM_CHARACTERS} characters.")
        return
    record = call_service(
        "POST",
        "/assessments",
        None,
        {"campaign_reference": campaign_reference, "executive_request": request_text.strip()},
        "Five agents are assessing the campaign, one language model call each...",
    )
    st.success(
        f"{record['assessment_id']} is waiting for a Clearance Controller, recommending "
        f"{record['recommended_decision']}.",
        icon=":material/hourglass_top:",
    )
    st.write(record["assessment_text"])


def _headline(detail: dict, existing: dict | None, queue_size: int) -> None:
    """The campaign's badges, one-line description and key figures."""
    profile, amendment, summary = detail["profile"], detail["amendment"], detail["summary"]
    badges = [
        badge_markdown(profile["product_category"], "gray"),
        badge_markdown(
            f"Amendment {amendment['amendment_score']:.4f}, " f"{amendment['amendment_band']}",
            config.AMENDMENT_BAND_BADGE_COLOURS[amendment["amendment_band"]],
        ),
    ]
    badges += [
        badge_markdown(f"{decision} open", config.DECISION_BADGE_COLOURS[decision])
        for decision in summary["decisions_open"]
    ]
    badges.append(
        badge_markdown(f"Queue position {detail['queue_position']} of {queue_size}", "gray")
    )
    if existing:
        badges.append(
            badge_markdown(existing["status"], config.STATUS_BADGE_COLOURS[existing["status"]])
        )
    st.markdown(" ".join(badges))
    st.caption(
        f"{profile['advertiser_name']}'s {profile['copy_length_seconds']}-second copy "
        f"\"{profile['copy_title']}\", clock number `{profile['clock_number']}`, booked by "
        f"{profile['agency_name']}; every rule read from {detail['code_rules']['document_name']}."
    )
    columns = st.columns(5, border=True)
    columns[0].metric("Spots booked", summary["spots_booked"])
    columns[1].metric("Compliant", summary["spots_compliant"])
    columns[2].metric(
        "Breaching",
        summary["spots_breaching"],
        help="Spots that break at least one of the five rules of the code.",
    )
    columns[3].metric(
        "Exposure",
        f"{detail['exposure']:,.1f} thousand",
        help="Forecast viewers under 18 that the breaching spots would reach.",
    )
    columns[4].metric("First spot", first_spot_text(summary["first_transmission"]))


def render_campaign_detail_page() -> None:
    """Show one campaign's facts and let an executive or controller start its assessment."""
    page_header(
        config.PAGE_CAMPAIGN,
        "Every spot and result on this page comes from the same tools the agents are given, "
        "so what you read here is what the assessment will work from.",
        DETAIL_EXPLANATION,
    )
    queue = call_service(
        "GET", "/queue", {"product_category": config.QUEUE_CATEGORY_ALL}, None, None
    )
    references = [row["campaign_reference"] for row in queue]
    remembered = st.session_state.get(SELECTED_CAMPAIGN_KEY)
    chosen = st.selectbox(
        "Campaign",
        references,
        index=references.index(remembered) if remembered in references else 0,
        help="In queue order. The campaign you picked on the queue page opens first.",
    )
    st.session_state[SELECTED_CAMPAIGN_KEY] = chosen

    detail = call_service("GET", f"/queue/{chosen}", None, None, None)
    assessed = call_service(
        "GET", "/clearances", {"status_value": config.CAMPAIGN_STATUS_OPTIONS}, None, None
    )
    existing = next((record for record in assessed if record["campaign_reference"] == chosen), None)

    st.header(chosen)
    _headline(detail, existing, len(queue))
    days_tab, rules_tab, copy_tab, assess_tab = st.tabs(
        ["Broadcast days", "Spots and rules", "Copy and notes", "Assess"]
    )
    with days_tab:
        _render_broadcast_days(detail)
    with rules_tab:
        _render_spots_and_rules(detail)
    with copy_tab:
        _render_copy_and_notes(detail)
    with assess_tab:
        if config.AREA_ASSESSMENT in signed_in_user()["areas"]:
            _render_assess(chosen, existing)
