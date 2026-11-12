"""Cutting the documents into numbered sections, and the sections into overlapping passages.

Two rules keep every passage meaningful on its own:
  * a passage never crosses from one section into the next, and
  * a passage is cut only between sentences, so every sentence (and so every rule) lies whole
    inside at least one passage.
Neighbouring passages share their last whole sentences, up to config.CHUNK_OVERLAP_CHARACTERS,
so a sentence that sits at a boundary still has its context on both sides.
"""

import re

from config import config


def _flatten_whitespace(text: str) -> str:
    """Replace control characters with spaces and every run of whitespace with one space."""
    without_controls = re.sub(config.CONTROL_CHARACTER_PATTERN, " ", text)
    return re.sub(config.WHITESPACE_RUN_PATTERN, " ", without_controls).strip()


def _split_one_document(document_name: str, document_text: str) -> list:
    """Drop a document's single header line and cut it at its numbered section headings."""
    headers = re.findall(config.DOCUMENT_HEADER_PATTERN, document_text)
    if len(headers) != 1:
        raise ValueError(
            f"{document_name} carries {len(headers)} header lines where it carries one"
        )
    text = re.sub(config.DOCUMENT_HEADER_PATTERN, "\n", document_text)

    headings = list(re.finditer(config.SECTION_HEADING_PATTERN, text))
    numbers = [int(heading.group(1)) for heading in headings]
    if not numbers or numbers != list(range(1, len(numbers) + 1)):
        raise ValueError(f"{document_name} has sections numbered {numbers}, not 1, 2, 3 and on")

    # The text before the second heading (the title included) belongs to the first section.
    starts = [0] + [heading.start() for heading in headings[1:]]
    ends = starts[1:] + [len(text)]
    return [
        {"document_name": document_name, "section_text": _flatten_whitespace(text[start:end])}
        for start, end in zip(starts, ends)
    ]


def split_documents_into_sections(document_entries: list[dict]) -> list:
    """Divide every document at its section headings, documents and sections in their own order."""
    return [
        section
        for entry in document_entries
        for section in _split_one_document(entry["document_name"], entry["document_text"])
    ]


def _overlap_tail(sentences: list[str]) -> list[str]:
    """The last whole sentences of a passage that fit within the overlap allowance."""
    tail = []
    for sentence in reversed(sentences):
        if len(" ".join([sentence] + tail)) > config.CHUNK_OVERLAP_CHARACTERS:
            break
        tail.insert(0, sentence)
    return tail


def _section_passages(section_text: str) -> list[str]:
    """Pack a section's sentences into passages of about config.CHUNK_SIZE_CHARACTERS."""
    sentences = [
        sentence
        for sentence in re.split(config.SENTENCE_BOUNDARY_PATTERN, section_text)
        if sentence
    ]
    passages, current = [], []
    for sentence in sentences:
        if current and len(" ".join(current + [sentence])) > config.CHUNK_SIZE_CHARACTERS:
            passages.append(" ".join(current))
            current = _overlap_tail(current)
        current.append(sentence)
    if current:
        passages.append(" ".join(current))
    return passages


def build_chunk_records(section_entries: list[dict]) -> list:
    """Cut every section into passages, numbering them within their document from 0."""
    chunk_records = []
    passages_numbered = {}
    for section in section_entries:
        document_name = section["document_name"]
        for passage in _section_passages(section["section_text"]):
            number = passages_numbered.get(document_name, 0)
            passages_numbered[document_name] = number + 1
            chunk_records.append(
                {
                    "chunk_id": f"{document_name}-{number}",
                    "chunk_text": passage,
                    "document_name": document_name,
                }
            )
    return chunk_records
