"""The one door between the dashboard and the REST service.

Every request a page sends passes through call_service: no page holds an address, a timeout or a
session token. The session (who is signed in, their token, their pages) lives in Streamlit's
st.session_state, which belongs to one browser tab and survives reruns of the script.
"""

import requests
import streamlit as st

from config import config

SESSION_KEY = "signed_in_user"
FLASH_MESSAGE_KEY = "flash_message"
PAGES_KEY = "pages_by_name"


def signed_in_user() -> dict | None:
    """The signed-in person's sign-in reply, or None before anyone signs in."""
    return st.session_state.get(SESSION_KEY)


def end_dashboard_session(message: str | None) -> None:
    """Forget the signed-in person in this browser tab, keeping a message for the sign-in form."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    if message:
        st.session_state[FLASH_MESSAGE_KEY] = message


def _show_navigation_back() -> None:
    """After an error, offer a way back to the first page the person may use."""
    pages = st.session_state.get(PAGES_KEY) or {}
    if pages:
        first_page = next(iter(pages.values()))
        st.page_link(first_page, label="Back to " + first_page.title, icon=":material/arrow_back:")


def _error_detail(response: requests.Response) -> str:
    """The service's own message, or the status and text when the reply is not JSON."""
    try:
        detail = response.json().get("detail")
    except ValueError:
        return f"{response.status_code}: {response.text[:300]}"
    if isinstance(detail, list):  # FastAPI's 422 lists every field that failed validation
        return "; ".join(
            f"{'.'.join(str(part) for part in item.get('loc', []))}: {item.get('msg')}"
            for item in detail
        )
    return str(detail)


def _send(method: str, path: str, query: dict | None, payload: dict | None) -> requests.Response:
    """Send one request to the service, with the session token whenever there is a session."""
    user = signed_in_user()
    headers = {config.SESSION_TOKEN_HEADER: user["session_token"]} if user else {}
    return requests.request(
        method,
        config.API_BASE_URL + path,
        params=query,
        json=payload,
        headers=headers,
        timeout=config.API_REQUEST_TIMEOUT_SECONDS,
    )


def call_service(
    method: str, path: str, query: dict | None, payload: dict | None, waiting_message: str | None
):
    """Make one call to the service and return its JSON reply; stop the page on any refusal."""
    try:
        if waiting_message:
            with st.spinner(waiting_message, show_time=True):
                response = _send(method, path, query, payload)
        else:
            response = _send(method, path, query, payload)
    except requests.RequestException as failure:
        st.error(
            f"The service at {config.API_BASE_URL} could not be reached "
            f"({type(failure).__name__}). "
            "Check that run_api_service.py is running, then try again.",
            icon=":material/cloud_off:",
        )
        st.stop()

    if response.status_code == requests.codes.unauthorized and signed_in_user():
        end_dashboard_session(config.SESSION_ENDED_MESSAGE)
        st.rerun()
    if response.status_code >= requests.codes.bad_request:
        st.error(_error_detail(response), icon=":material/error:")
        _show_navigation_back()
        st.stop()
    return response.json()
