"""Recalling the last few exchanges of one person's own conversation.

Memory lives in the chat_turns table, so it survives restarts and belongs to the person who asked:
nobody's questions ever reach somebody else's prompt. A campaign status answer is never replayed,
because it was built from a campaign's record and nothing from a record is put before the model.
"""

from sqlalchemy.orm import Session

from config import config
from persistence.models import ChatTurn
from persistence.repository import fetch_chat_turns

FIRST_QUESTION_TEXT = (
    "This is the first question of the conversation; there are no earlier exchanges."
)


def recall_conversation(session: Session, username: str, question_text: str) -> dict:
    """The person's recent exchanges as history text, and the text to search the documents with."""
    recent_turns = fetch_chat_turns(
        session, ChatTurn.username == username, config.CHAT_HISTORY_TURN_LIMIT
    )
    if not recent_turns:
        return {"history_text": FIRST_QUESTION_TEXT, "search_text": question_text}

    exchanges = []
    for turn in reversed(recent_turns):
        answer = (
            config.CHAT_STATUS_ANSWER_IN_HISTORY
            if turn["question_type"] == config.CHAT_TYPE_CAMPAIGN_STATUS
            else turn["answer_text"]
        )
        exchanges.append(f"Question: {turn['question_text']}\nAnswer: {answer}")
    return {
        "history_text": "\n\n".join(exchanges),
        "search_text": f"{recent_turns[0]['question_text']} {question_text}",
    }
