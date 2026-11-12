"""Entry point 4: build the searchable knowledge base and read every category's rules back from it.

Run from the Project folder:  python run_knowledge_indexing.py
"""

from config import config
from knowledge.chunk_splitting import build_chunk_records, split_documents_into_sections
from knowledge.code_reading import read_category_code
from knowledge.pdf_extraction import extract_text_from_directory
from knowledge.vector_index import build_vector_index, open_vector_index


def describe_rules(rules: dict) -> list:
    """The lines that print one category's rules, as read from its code."""
    hours = rules["restricted_hours"]
    lines = [
        (
            f"       restricted hours: {hours['start']} to {hours['end']}"
            if hours
            else "       restricted hours: none, spots at any time of the broadcast day"
        )
    ]
    exception = rules["live_sport_exception"]
    if exception:
        lines.append(
            f"       live sport exception: centre breaks more than "
            f"{exception['minutes_after_start']} minutes after the start and more than "
            f"{exception['minutes_before_end']} minutes before the end"
        )
    lines.append(
        f"       under-18 index threshold {rules['index_threshold']}, children's "
        f"programmes {rules['children_rule']}, at most {rules['daily_cap']} spots a "
        f"channel a broadcast day, {rules['separation_minutes']} minutes apart"
    )
    return lines


def main() -> None:
    """Rebuild the store from the sixteen documents, then read every code back from it."""
    print("building the clearance knowledge index")
    documents = extract_text_from_directory(config.DOCS_DIRECTORY)
    sections = split_documents_into_sections(documents)
    chunks = build_chunk_records(sections)
    stored = build_vector_index(chunks, config.VECTOR_STORE_DIRECTORY)
    print(f"       documents read: {len(documents)}")
    print(f"       sections found: {len(sections)}")
    print(f"       chunks stored: {stored}")
    print(f"       store location: {config.VECTOR_STORE_DIRECTORY}")

    collection = open_vector_index(config.VECTOR_STORE_DIRECTORY)
    print("rules read from each category's own code")
    for category in config.CATEGORY_CODE_DOCUMENTS:
        rules = read_category_code(collection, category)
        print(f"{category:<22} {rules['document_name']}")
        for line in describe_rules(rules):
            print(line)


if __name__ == "__main__":
    main()
