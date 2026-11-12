"""Entry point 6: the Streamlit dashboard - signing in, the navigation and the five pages.

Run from the Project folder, with the service already running:
    python -m streamlit run run_dashboard.py
"""

import requests
import streamlit as st

from config import config
from dashboard.api_client import (
    FLASH_MESSAGE_KEY,
    PAGES_KEY,
    SESSION_KEY,
    call_service,
    end_dashboard_session,
    signed_in_user,
)
from dashboard.audit_trail import render_audit_trail_page
from dashboard.campaign_detail import render_campaign_detail_page
from dashboard.clearance_chat import render_clearance_chat_page
from dashboard.clearance_queue import render_clearance_queue_page
from dashboard.decision_queue import render_decision_queue_page
from dashboard.ui_components import apply_fonts

PAGE_RENDERERS = {
    config.PAGE_QUEUE: (render_clearance_queue_page, ":material/format_list_numbered:"),
    config.PAGE_CAMPAIGN: (render_campaign_detail_page, ":material/smart_display:"),
    config.PAGE_DECISION: (render_decision_queue_page, ":material/gavel:"),
    config.PAGE_AUDIT: (render_audit_trail_page, ":material/manage_search:"),
    config.PAGE_CHAT: (render_clearance_chat_page, ":material/forum:"),
}

HEALTH_CHECK_SECONDS = 5
ROLE_DESCRIPTIONS = {
    config.ROLE_CONTROLLER: "Sees every page and is the only role that may decide an assessment "
    "(never one they opened themselves).",
    config.ROLE_EXECUTIVE: "Works the queue, inspects campaigns, starts assessments, reads the "
    "audit trail and uses the chat. Cannot decide.",
    config.ROLE_LIAISON: "Uses the chat only, and hears only about the campaigns of the four "
    "agencies assigned to them.",
}


def _service_is_listening() -> bool:
    """True when something answers at the service's address."""
    try:
        requests.get(config.API_BASE_URL + "/openapi.json", timeout=HEALTH_CHECK_SECONDS)
    except requests.RequestException:
        return False
    return True


def _render_brand() -> None:
    """The dashboard's title and subtitle, set in the display typeface."""
    st.html(
        f'<p class="brand-title">{config.DASHBOARD_TITLE}</p>'
        f'<p class="brand-subtitle">{config.DASHBOARD_SUBTITLE}</p>'
    )


def _render_sample_credentials() -> None:
    """A drop-down of one sample account per role, ready to copy (for demonstrations only)."""
    with st.expander("Sample credentials for the three roles", icon=":material/key:"):
        samples = {}
        for account in config.SEED_USER_ACCOUNTS:
            samples.setdefault(account["role"], account)
        role = st.selectbox("Role", list(samples), key="sample_role")
        account = samples[role]
        st.caption(ROLE_DESCRIPTIONS[role])
        username_column, password_column = st.columns(2)
        username_column.caption("Username")
        username_column.code(account["username"], language=None)
        password_column.caption("Password")
        password_column.code(account["password"], language=None)
        if account["assigned_agencies"]:
            st.caption("Assigned agencies: " + ", ".join(account["assigned_agencies"]))


def _render_sign_in() -> None:
    """The sign-in form, under the title and subtitle; a refusal shows the service's message."""
    _, centre, _ = st.columns([1, 1.4, 1])
    with centre:
        _render_brand()
        flash_message = st.session_state.pop(FLASH_MESSAGE_KEY, None)
        if flash_message:
            st.warning(flash_message, icon=":material/logout:")
        with st.form("sign_in_form", border=True):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button(
                "Sign in", type="primary", width="stretch", icon=":material/login:"
            )
        _render_sample_credentials()
        if submitted:
            reply = call_service(
                "POST",
                "/auth/signin",
                None,
                {"username": username.strip(), "password": password},
                None,
            )
            st.session_state[SESSION_KEY] = reply
            st.rerun()


def _render_sidebar(user: dict) -> None:
    """Who is signed in, their role and agencies, and the way to sign out."""
    with st.sidebar:
        _render_brand()
        with st.container(border=True):
            st.markdown(f"**{user['display_name']}**  \n{user['role']}")
            if user["assigned_agencies"]:
                st.caption("Agencies: " + ", ".join(user["assigned_agencies"]))
        if st.button("Sign out", icon=":material/logout:", width="stretch"):
            call_service("POST", "/auth/signout", None, None, None)
            end_dashboard_session(None)
            st.rerun()
        st.caption(
            "AI agents draft and explain; a Clearance Controller decides. Nothing is sent "
            "to an agency on the word of a model."
        )


def main() -> None:
    """Draw the dashboard with only the pages the signed-in role is offered."""
    st.set_page_config(
        page_title=config.DASHBOARD_TITLE, page_icon=":material/smart_display:", layout="wide"
    )
    apply_fonts()
    if not _service_is_listening():
        st.error(
            f"The BreakCheck AI service is not answering at {config.API_BASE_URL}. Start it in "
            "another terminal with `python run_api_service.py`, then reload this page.",
            icon=":material/cloud_off:",
        )
        st.stop()

    user = signed_in_user()
    if user is None:
        _render_sign_in()
        return
    pages = [
        st.Page(PAGE_RENDERERS[name][0], title=name, icon=PAGE_RENDERERS[name][1])
        for name in user["pages"]
    ]
    st.session_state[PAGES_KEY] = {page.title: page for page in pages}
    _render_sidebar(user)
    st.navigation(pages).run()


if __name__ == "__main__":
    main()
