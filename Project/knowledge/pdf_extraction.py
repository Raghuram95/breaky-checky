"""Getting the text out of the clearance documents.

A page that yields no text could be the page that states a rule, so a document with such a page
is refused rather than indexed with a hole in it.
"""

import os

from pypdf import PdfReader


def extract_text_from_directory(docs_directory: str) -> list:
    """Read every PDF in the folder, in file-name order, skipping anything that is not a PDF."""
    document_entries = []
    for file_name in sorted(os.listdir(docs_directory)):
        if not file_name.lower().endswith(".pdf"):
            continue
        reader = PdfReader(os.path.join(docs_directory, file_name))
        page_texts = []
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if not page_text.strip():
                raise ValueError(f"{file_name}: page {page_number} gives up no text")
            page_texts.append(page_text)
        document_entries.append(
            {"document_name": file_name, "document_text": "\n".join(page_texts)}
        )
    return document_entries
