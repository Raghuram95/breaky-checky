"""The clearance chat page: ask the assistant, and read your own conversation."""

import streamlit as st

from config import config
from dashboard.api_client import call_service, signed_in_user
from dashboard.ui_components import page_header

CHAT_EXPLANATION = f"""
**An assistant that only answers from the company's documents.** Each question costs exactly one
language model call. The service first *retrieves* the passages of the codes, policy and guides
closest in meaning to your question (it never reads the casebook of past clearances), then the model
answers from those passages alone and names the documents it used. If the documents do not cover
the question, it **refuses rather than guesses**.

**Questions about one campaign** (name it by its reference, e.g. `{config.CAMPAIGN_REFERENCE_FORM}`)
are answered from the campaign's record by the service itself, never by the model. An Agency Liaison
hears only about their own agencies' campaigns.

**Memory.** The assistant remembers your last {config.CHAT_HISTORY_TURN_LIMIT} exchanges, so you can
ask a follow-up such as "and for alcohol?".
"""


def _render_side_panel(user: dict) -> None:
    """What the assistant answers, what it remembers and, for a liaison, whose campaigns."""
    with st.container(border=True):
        st.markdown("**What the assistant answers**")
        for question_type in config.CHAT_TYPE_OPTIONS:
            st.badge(question_type, color=config.CHAT_TYPE_BADGE_COLOURS[question_type])
        st.caption(
            f"Answers come only from {config.ORGANISATION_NAME}'s codes, policies and guides, "
            "never from the casebook of past clearances, and each lists the documents it "
            "was given. All five codes are read in full for every question. The assistant "
            f"remembers your last {config.CHAT_HISTORY_TURN_LIMIT} exchanges."
        )
        if user["role"] not in config.CHAT_RECORD_ROLES:
            agencies = ", ".join(user["assigned_agencies"])
            st.caption(
                "About a campaign you hear its status and when it last changed, and once it "
                "is decided the decision, the spots cleared and withdrawn and the documents "
                f"sent - only for the agencies assigned to you: {agencies}. Any other "
                "campaign cannot be found."
            )


def _render_turn(turn: dict) -> None:
    """One exchange: the question, then the answer with its type, refusal mark and sources."""
    with st.chat_message("user", avatar=":material/person:"):
        st.write(turn["question_text"])
    with st.chat_message("assistant", avatar=":material/smart_display:"):
        badges = [
            f":{config.CHAT_TYPE_BADGE_COLOURS[turn['question_type']]}-badge"
            f"[{turn['question_type']}]"
        ]
        if turn["was_refused"]:
            badges.append(":gray-badge[refused]")
        st.markdown(" ".join(badges))
        st.write(turn["answer_text"])
        if turn["source_documents"]:
            st.caption("From " + turn["source_documents"])


def render_clearance_chat_page() -> None:
    """Ask the clearance assistant a question and show the signed-in person's conversation."""
    page_header(
        config.PAGE_CHAT,
        "Ask about a code's rules, how spots are scheduled and checked, "
        "what an agency can do, or where a campaign has got to by its reference.",
        CHAT_EXPLANATION,
    )
    user = signed_in_user()
    conversation_column, panel_column = st.columns([3, 1], gap="large")
    with panel_column:
        _render_side_panel(user)
    with conversation_column:
        turns = call_service(
            "GET", "/chat/turns", {"turn_limit": config.CHAT_PAGE_TURN_LIMIT}, None, None
        )
        if not turns:
            st.caption(
                'No questions yet. Try: "How many spots of one alcohol campaign may go out '
                'on one channel in one broadcast day?"'
            )
        for turn in reversed(turns):
            _render_turn(turn)

    question = st.chat_input(
        "Ask about a code, scheduling, what an agency can do, or a campaign " "by its reference"
    )
    if question is None:
        return
    if len(question.strip()) < config.CHAT_QUESTION_MINIMUM_CHARACTERS:
        st.warning(f"Write at least {config.CHAT_QUESTION_MINIMUM_CHARACTERS} characters.")
        return
    call_service(
        "POST",
        "/chat/ask",
        None,
        {"question": question.strip()},
        "Searching the documents and answering in one language model call...",
    )
    st.rerun()
