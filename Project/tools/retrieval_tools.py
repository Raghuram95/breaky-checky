"""Reading the passage an agent quotes, from the one document that states it.

Each tool searches a single document, so a guide's checks always come from that issue's guide and
a code's steps from that category's code, never from a look-alike passage of another document.
"""

from chromadb.api.models.Collection import Collection

from config import config
from knowledge.vector_index import find_nearest_passage


def _configured_document(documents, subject, kind):
    """The document config names for a subject; a subject with none is refused."""
    if subject not in documents:
        raise ValueError(f"No {kind} is configured for {subject!r}")
    return documents[subject]


def read_executive_checks(collection: Collection, issue_category: str) -> str:
    """The passage of the issue's guide that says what the executive checks."""
    guide = _configured_document(config.GUIDE_DOCUMENT_BY_ISSUE, issue_category, "guide")
    return find_nearest_passage(collection, config.EXECUTIVE_CHECKS_QUESTION, guide)


def read_rebooking_steps(collection: Collection, product_category: str) -> str:
    """The passage of the category's code that says how a withdrawn spot is re-booked."""
    code = _configured_document(config.CATEGORY_CODE_DOCUMENTS, product_category, "code")
    return find_nearest_passage(collection, config.REBOOKING_QUESTION, code)


def read_standing_restrictions(collection: Collection, product_category: str) -> str:
    """The passage of the category's code that lists the standing restrictions on cleared spots."""
    code = _configured_document(config.CATEGORY_CODE_DOCUMENTS, product_category, "code")
    return find_nearest_passage(collection, config.STANDING_RESTRICTIONS_QUESTION, code)
