"""Small building blocks the dashboard pages share: fonts, badges, times and the workflow guide.

Keeping them in one place means every page badges a band, a decision or a status the same way,
writes times the same way, and explains the clearance workflow in the same plain words.
Colours come only from config (badge colours) or from the theme in .streamlit/config.toml.
"""

from datetime import datetime

import streamlit as st

from config import config
from dashboard.api_client import PAGES_KEY

FONT_URL = ("https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;"
            "9..144,650&family=Manrope:wght@400;500;600;700&display=swap")
FONT_STYLESHEET = """
<style>
@import url('FONT_URL_PLACEHOLDER');
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp td, .stApp th,
.stApp div[data-testid="stMarkdownContainer"], .stApp div[data-testid="stCaptionContainer"],
.stApp div[data-testid="stMetricLabel"], .stApp div[data-testid="stMetricValue"],
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span:not([data-testid="stIconMaterial"]) {
    font-family: 'Manrope', sans-serif;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .brand-title {
    font-family: 'Fraunces', serif !important;
    letter-spacing: -0.01em;
}
.stApp h1 { font-weight: 650; }
.brand-title { font-size: 2.6rem; font-weight: 650; line-height: 1.1; margin: 0; }
.brand-subtitle { font-size: 1.02rem; opacity: 0.72; margin-top: 0.35rem; }
.step-guide { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.2rem 0 0.9rem 0; }
.step-guide span { border: 1px solid rgba(120, 113, 108, 0.35); border-radius: 999px;
                   padding: 0.18rem 0.75rem; font-size: 0.82rem; opacity: 0.6; }
.step-guide span.current { opacity: 1; font-weight: 700; border-width: 2px; }
</style>
"""

WORKFLOW_STEPS = [
    (config.PAGE_QUEUE, "1 · Rank the queue"),
    (config.PAGE_CAMPAIGN, "2 · Inspect & assess"),
    (config.PAGE_DECISION, "3 · Controller decides"),
    (config.PAGE_AUDIT, "4 · Audit the record"),
    (config.PAGE_CHAT, "Ask the assistant"),
]


def apply_fonts() -> None:
    """Load the dashboard's typefaces: Fraunces for titles, Manrope for everything else."""
    st.html(FONT_STYLESHEET.replace("FONT_URL_PLACEHOLDER", FONT_URL))


def badge(text: str, colour_map: dict, key: str) -> None:
    """A coloured badge whose colour config assigns to the value."""
    st.badge(text, color=colour_map.get(key, "gray"))


def badge_markdown(text: str, colour: str) -> str:
    """Badge markup to place inline in markdown text, e.g. several badges in a row."""
    return f":{colour}-badge[{text}]"


def as_dashboard_time(timestamp: str | None) -> str:
    """An ISO timestamp from the service written in the dashboard's time format."""
    if not timestamp:
        return "—"
    return datetime.fromisoformat(str(timestamp)).strftime(config.DASHBOARD_TIME_FORMAT)


def first_spot_text(first_transmission: dict) -> str:
    """A first transmission as 'Saturday 18:20'."""
    return f"{first_transmission['broadcast_day']} {first_transmission['break_time']}"


def page_header(page_name: str, introduction: str, explanation: str) -> None:
    """Title, where this page sits in the workflow, a one-line summary and a plain-words guide."""
    st.title(page_name)
    steps = "".join(
        f'<span class="{"current" if name == page_name else ""}">{label}</span>'
        for name, label in WORKFLOW_STEPS
    )
    st.html(f'<div class="step-guide">{steps}</div>')
    st.caption(introduction)
    with st.expander("New here? What this page does, in plain words", icon=":material/help:"):
        st.markdown(explanation)


def go_to_page(page_name: str) -> None:
    """Switch to another page the signed-in person is offered."""
    pages = st.session_state.get(PAGES_KEY, {})
    if page_name in pages:
        st.switch_page(pages[page_name])
