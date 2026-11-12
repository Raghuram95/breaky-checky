"""Reading one category's five rules out of that category's own code.

No hour, threshold, limit or separation is written into the solution: each is read at run time
from the sentences of the category's own code document, with the patterns in config. A code is
read whole (every one of its passages), and each rule must have exactly one value. The same
sentence met in two overlapping passages is one statement; two different values are refused.
"""

import re

from chromadb.api.models.Collection import Collection

from config import config
from knowledge.vector_index import read_document_chunks


def _statements(passages: list[str], pattern: str):
    """Every distinct statement of one rule across the passages: {matched sentence: its groups}."""
    return {
        found.group(0): found.groups()
        for passage in passages
        for found in re.finditer(pattern, passage)
    }


def _single_statement(statements: dict, rule_name: str, document_name: str):
    """The one value a rule states; no value or two different values are refused."""
    values = set(statements.values())
    if len(values) != 1:
        raise ValueError(f"{document_name} states {len(values)} values for {rule_name}, not one")
    return next(iter(statements.items()))


def _read_window(passages: list[str], document_name: str):
    """The restricted hours (or None for 'any time'), the live sport exception, and its sentence."""
    hours = _statements(passages, config.RESTRICTED_HOURS_PATTERN)
    any_time = _statements(passages, config.ANY_TIME_PATTERN)
    exceptions = _statements(passages, config.LIVE_SPORT_EXCEPTION_PATTERN)
    if bool(hours) == bool(any_time):
        raise ValueError(
            f"{document_name} must state either restricted hours or that spots may "
            "go out at any time, and states both or neither"
        )
    if any_time:
        if exceptions:
            raise ValueError(
                f"{document_name} states a live sport exception without restricted hours"
            )
        return None, None, next(iter(any_time))

    hours_sentence, (start, end) = _single_statement(hours, "restricted hours", document_name)
    if start == end:
        raise ValueError(f"{document_name} states restricted hours that begin and end at {start}")
    restricted_hours = {"start": start, "end": end}
    if not exceptions:
        return restricted_hours, None, hours_sentence
    exception_sentence, (after, before) = _single_statement(
        exceptions, "the live sport exception", document_name
    )
    exception = {"minutes_after_start": int(after), "minutes_before_end": int(before)}
    return restricted_hours, exception, hours_sentence + " " + exception_sentence


def _read_children_rule(passages: list[str], document_name: str):
    """Which reach of the children's rule the code states, and its sentence; exactly one."""
    stated = {
        rule: _statements(passages, pattern)
        for rule, pattern in config.CHILDREN_RULE_PATTERNS.items()
    }
    stated = {rule: statements for rule, statements in stated.items() if statements}
    if len(stated) != 1:
        raise ValueError(f"{document_name} states {len(stated)} children's rules, not one")
    rule, statements = next(iter(stated.items()))
    return rule, next(iter(statements))


def read_category_code(collection: Collection, product_category: str) -> dict:
    """Read the five rules a campaign's spots answer to from its category's own code."""
    document_name = config.CATEGORY_CODE_DOCUMENTS.get(product_category)
    if document_name is None:
        raise ValueError(f"No code is configured for the category {product_category!r}")
    passages = [
        re.sub(config.WHITESPACE_RUN_PATTERN, " ", passage)
        for passage in read_document_chunks(collection, document_name)
    ]

    restricted_hours, exception, window_sentence = _read_window(passages, document_name)
    children_rule, children_sentence = _read_children_rule(passages, document_name)
    figures, figure_sentences = {}, {}
    for figure_name, pattern in config.CODE_FIGURE_PATTERNS.items():
        sentence, (value,) = _single_statement(
            _statements(passages, pattern), figure_name, document_name
        )
        figures[figure_name], figure_sentences[figure_name] = int(value), sentence
    if figures["daily_cap"] == 0:
        raise ValueError(f"{document_name} states a daily limit of nought")

    return {
        "document_name": document_name,
        "restricted_hours": restricted_hours,
        "live_sport_exception": exception,
        "children_rule": children_rule,
        "index_threshold": figures["index_threshold"],
        "daily_cap": figures["daily_cap"],
        "separation_minutes": figures["separation_minutes"],
        "rule_sentences": {
            config.BREACH_WINDOW: window_sentence,
            config.BREACH_INDEX: figure_sentences["index_threshold"],
            config.BREACH_CHILDREN: children_sentence,
            config.BREACH_FREQUENCY: figure_sentences["daily_cap"],
            config.BREACH_SEPARATION: figure_sentences["separation_minutes"],
        },
    }
