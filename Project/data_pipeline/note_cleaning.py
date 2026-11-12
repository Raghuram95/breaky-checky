"""The cleaning run for the free-text clearance notes.

Notes are written at speed: wrapped in markup, full of escaped characters, sometimes shouted in
capitals and sometimes too short to mean anything. The classifier reads words, so every note is
reduced to the same plain words however it was written.
"""

import re

import pandas as pd

from config import config
from data_pipeline.common_cleaning import (
    convert_markup_to_plain_text,
    convert_whole_numbers,
    normalize_category_values,
    refuse_values,
    remove_duplicate_records,
    remove_rows_missing_target,
)
from data_pipeline.dataset_loader import load_raw_dataset, save_processed_dataset


def is_shouted(text: str) -> bool:
    """True when a note's letters are all capitals, so it can only be restored in lower case."""
    return text == text.upper() and text != text.lower()


def word_count(text: str) -> int:
    """Count the runs of characters between whitespace."""
    return len(str(text).split())


def restore_capitals(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Write every note that arrived wholly in capitals in lower case."""
    restored = dataframe.copy()
    column_name = config.NOTE_TEXT_COLUMN
    restored[column_name] = restored[column_name].map(
        lambda text: text.lower() if is_shouted(text) else text, na_action="ignore"
    )
    return restored


def normalize_note_timestamps(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Write every note time as the weekday in full, a space and the time, e.g. 'Thursday 09:05'."""
    normalized = dataframe.copy()
    column_name = config.NOTE_TIMESTAMP_COLUMN

    def read_timestamp(value):
        found = re.fullmatch(config.NOTE_TIMESTAMP_PATTERN, str(value).strip())
        if not found:
            return None
        weekday = config.WEEKDAY_MAPPING.get(found.group(config.NOTE_WEEKDAY_PART).lower())
        return f"{weekday} {found.group(config.NOTE_TIME_PART)}" if weekday else None

    readings = normalized[column_name].map(read_timestamp, na_action="ignore")
    unreadable = normalized[column_name].notna() & readings.isna()
    if unreadable.any():
        refuse_values(
            column_name,
            normalized.loc[unreadable, column_name],
            "holds note times that cannot be read",
        )
    normalized[column_name] = readings
    return normalized


def clean_note_dataset(input_file_path: str, output_file_path: str) -> pd.DataFrame:
    """Produce a notes table the issue model can learn from, and save it with the raw columns."""
    raw_frame = load_raw_dataset(input_file_path)
    cleaned = remove_duplicate_records(raw_frame)
    cleaned = remove_rows_missing_target(cleaned, config.NOTE_TARGET_COLUMN)
    cleaned = normalize_category_values(cleaned, config.NOTE_VALUE_MAPPINGS)
    cleaned = convert_whole_numbers(cleaned, config.NOTE_WHOLE_NUMBER_PATTERNS)
    cleaned = normalize_note_timestamps(cleaned)
    cleaned = convert_markup_to_plain_text(cleaned)
    cleaned = restore_capitals(cleaned)
    words = cleaned[config.NOTE_TEXT_COLUMN].map(word_count, na_action="ignore").fillna(0)
    cleaned = cleaned[words >= config.MINIMUM_NOTE_WORDS].reset_index(drop=True)
    cleaned[config.NOTE_LENGTH_COLUMN] = cleaned[config.NOTE_TEXT_COLUMN].str.len().astype("Int64")
    cleaned = cleaned[list(raw_frame.columns)]
    save_processed_dataset(cleaned, output_file_path)
    return cleaned
