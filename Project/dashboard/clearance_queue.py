"""The clearance queue page: open campaigns ranked by the young audience their breaches reach."""

import pandas as pd
import streamlit as st

from config import config
from dashboard.api_client import call_service
from dashboard.ui_components import first_spot_text, go_to_page, page_header

SELECTED_CAMPAIGN_KEY = "selected_campaign"

QUEUE_EXPLANATION = """
**What you are looking at.** Every advertising campaign waiting for clearance this week. Before any
AI agent has run, the service has already placed each booked spot on its *broadcast day* and tested
it against the five rules of its category's own code (restricted hours, the under-18 audience index,
children's programmes, the daily limit and the separation between spots).

**Why this order.** The campaign whose rule-breaking spots would reach the most viewers under 18
comes
first. That figure, the **exposure**, is forecast by a machine-learning model from each programme's
slot, genre, channel and audience index. Ties go to the campaign that airs first.

**The amendment score** is a second model's estimate of how likely the advert's *copy* is to need
changes. It does not say whether a spot breaks a rule; it helps an executive decide what to read
first.

**What to do next.** Pick a campaign in the table and open it on the *Campaign detail* page.
"""


def _queue_metrics(whole_queue: list[dict]) -> None:
    """Counts along the top, worked out from the whole queue."""
    columns = st.columns(5, border=True)
    columns[0].metric("Campaigns", len(whole_queue))
    columns[1].metric(
        "With a breaching spot", sum(1 for row in whole_queue if row["spots_breaching"])
    )
    columns[2].metric(
        "Every spot breaching",
        sum(1 for row in whole_queue if row["spots_breaching"] == row["spots_booked"]),
    )
    columns[3].metric("Spots booked", f"{sum(row['spots_booked'] for row in whole_queue):,}")
    columns[4].metric(
        "Largest exposure",
        f"{max((row['exposure'] for row in whole_queue), default=0):,.1f} thousand",
        help="The forecast under-18 audience the breaching spots of the top campaign "
        "would reach, in thousands of viewers.",
    )


def _queue_table(rows: list[dict]) -> pd.DataFrame:
    """The queue rows as the table shows them."""
    return pd.DataFrame(
        [
            {
                "#": row["queue_position"],
                "Campaign": row["campaign_reference"],
                "Advertiser": row["advertiser_name"],
                "Category": row["product_category"],
                "Spots": row["spots_booked"],
                "Breaching": row["spots_breaching"],
                "First spot": first_spot_text(row["first_transmission"]),
                "Exposure (k)": row["exposure"],
                "Amendment score": row["amendment_score"],
                "Band": row["amendment_band"],
            }
            for row in rows
        ]
    )


def render_clearance_queue_page() -> None:
    """Show the ranked queue and let the reader pick a campaign for the detail page."""
    page_header(
        config.PAGE_QUEUE,
        "Every open campaign, each spot already tested against its category's own code. "
        "The largest forecast under-18 exposure comes first.",
        QUEUE_EXPLANATION,
    )
    whole_queue = call_service(
        "GET", "/queue", {"product_category": config.QUEUE_CATEGORY_ALL}, None, None
    )
    _queue_metrics(whole_queue)

    category = st.pills(
        "Category",
        config.QUEUE_CATEGORY_OPTIONS,
        default=config.QUEUE_CATEGORY_ALL,
        key="queue_category",
    )
    category = category or config.QUEUE_CATEGORY_ALL
    rows = (
        whole_queue
        if category == config.QUEUE_CATEGORY_ALL
        else call_service("GET", "/queue", {"product_category": category}, None, None)
    )

    selection = st.dataframe(
        _queue_table(rows),
        hide_index=True,
        width="stretch",
        height=520,
        on_select="rerun",
        selection_mode="single-row",
        key="queue_table",
        column_config={
            "Exposure (k)": st.column_config.NumberColumn(
                format="%.1f", help="Thousands of viewers under 18"
            ),
            "Amendment score": st.column_config.ProgressColumn(
                min_value=0.0, max_value=1.0, format="%.4f"
            ),
        },
    )
    st.caption(
        f"{len(rows)} campaigns shown. Select one to open it on the "
        f"{config.PAGE_CAMPAIGN} page."
    )

    chosen_rows = selection.selection.rows
    if chosen_rows:
        chosen = rows[chosen_rows[0]]["campaign_reference"]
        st.session_state[SELECTED_CAMPAIGN_KEY] = chosen
        if st.button(
            f"Open {chosen} on the {config.PAGE_CAMPAIGN} page",
            type="primary",
            icon=":material/arrow_forward:",
        ):
            go_to_page(config.PAGE_CAMPAIGN)
