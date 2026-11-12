"""Evidence that each raw extract needed cleaning, and that its cleaned table is clean.

Every count here is measured, never assumed: the raw frame as load_raw_dataset read it is
compared with the cleaned table the pipeline saved. The report is written to
output/data_quality_report.json so the cleaning can be defended row by row.
"""

import json
import os
import re

import pandas as pd

from config import config
from data_pipeline.common_cleaning import (
    convert_audiences_to_thousands,
    convert_clock_times_to_minutes,
    convert_decimal_numbers,
    convert_markup_to_plain_text,
    convert_spot_lengths_to_seconds,
    convert_to_broadcast_clock,
    convert_whole_numbers,
    normalize_category_values,
)
from data_pipeline.note_cleaning import is_shouted, word_count
from data_pipeline.submission_cleaning import clock_number_lengths


def _count(mask: pd.Series) -> int:
    """Count the True values of a boolean mask, gaps counting as False."""
    return int(mask.fillna(False).astype(bool).sum())


def _gaps_by_column(dataframe: pd.DataFrame) -> dict:
    """Each column holding at least one gap, against its number of gaps."""
    gaps = dataframe.isna().sum()
    return {column_name: int(count) for column_name, count in gaps.items() if count > 0}


def _not_plain_numbers(series: pd.Series) -> int:
    """Count the values written as something other than a plain number."""
    present = series.dropna().astype(str)
    return int((~present.str.fullmatch(config.PLAIN_NUMBER_PATTERN)).sum())


def _out_of_range(dataframe: pd.DataFrame, range_limits: dict[str, tuple]) -> dict:
    """Each limited column against how many of its readings fall outside the limits."""
    counts = {}
    for column_name, (lowest, highest) in range_limits.items():
        numbers = pd.to_numeric(dataframe[column_name])
        counts[column_name] = _count((numbers < lowest) | (numbers > highest))
    return counts


def _target_shares(clean_frame: pd.DataFrame, target_column: str) -> dict:
    """Each answer's share of the cleaned rows."""
    shares = clean_frame[target_column].value_counts(normalize=True)
    return {
        str(answer): round(float(share), config.TARGET_SHARE_DECIMALS)
        for answer, share in shares.sort_index().items()
    }


def _first_copies(raw_frame: pd.DataFrame) -> pd.DataFrame:
    """The raw rows that are not repeats of an earlier row."""
    return raw_frame[~raw_frame.duplicated(keep="first")]


def profile_dataset(raw_frame: pd.DataFrame, clean_frame: pd.DataFrame, target_column: str) -> dict:
    """Account for every row an extract arrived with: kept, repeated or missing its answer."""
    first_copies = _first_copies(raw_frame)
    return {
        "rows_received": int(len(raw_frame)),
        "rows_kept": int(len(clean_frame)),
        "rows_removed": int(len(raw_frame) - len(clean_frame)),
        "duplicate_rows": int(raw_frame.duplicated(keep="first").sum()),
        "rows_missing_target": int(first_copies[target_column].isna().sum()),
        "columns": int(clean_frame.shape[1]),
        "missing_cells_as_received": _gaps_by_column(raw_frame),
        "missing_cells_after_cleaning": _gaps_by_column(clean_frame),
    }


def _contradictions(frame: pd.DataFrame, copy_lengths: pd.Series) -> tuple[int, int]:
    """Count legal text longer than the copy, and amendments exceeding campaigns."""
    legal_too_long = frame[config.LEGAL_TEXT_SECONDS_COLUMN] > copy_lengths
    too_many_amendments = (
        frame[config.ADVERTISER_AMENDMENTS_COLUMN] > frame[config.ADVERTISER_CAMPAIGNS_COLUMN]
    )
    return _count(legal_too_long), _count(too_many_amendments)


def profile_submission_details(raw_frame: pd.DataFrame, clean_frame: pd.DataFrame) -> dict:
    """Record the defects peculiar to the submissions."""
    number_columns = list(config.SUBMISSION_WHOLE_NUMBER_PATTERNS) + list(
        config.SUBMISSION_DECIMAL_NUMBER_PATTERNS
    )
    read_raw = convert_whole_numbers(raw_frame, config.SUBMISSION_WHOLE_NUMBER_PATTERNS)
    read_raw = convert_decimal_numbers(read_raw)
    read_raw = convert_spot_lengths_to_seconds(read_raw, [config.COPY_LENGTH_COLUMN])
    clock_lengths = clock_number_lengths(raw_frame[config.CLOCK_NUMBER_COLUMN])
    written_lengths = raw_frame[config.COPY_LENGTH_COLUMN].dropna().astype(str)

    legal_as_received, amendments_as_received = _contradictions(read_raw, clock_lengths)
    legal_after, amendments_after = _contradictions(
        clean_frame, clean_frame[config.COPY_LENGTH_COLUMN]
    )
    return {
        "copy_lengths_not_written_in_seconds": int(
            (~written_lengths.str.fullmatch(config.WHOLE_NUMBER_PATTERN)).sum()
        ),
        "copy_lengths_disagreeing_with_clock_number": _count(
            read_raw[config.COPY_LENGTH_COLUMN] != clock_lengths
        ),
        "numbers_written_with_a_unit_or_sign": {
            column_name: _not_plain_numbers(raw_frame[column_name])
            for column_name in number_columns
        },
        "out_of_range_by_column": _out_of_range(read_raw, config.SUBMISSION_RANGE_LIMITS),
        "legal_text_longer_than_copy_as_received": legal_as_received,
        "legal_text_longer_than_copy_after_cleaning": legal_after,
        "amendments_exceeding_campaigns_as_received": amendments_as_received,
        "amendments_exceeding_campaigns_after_cleaning": amendments_after,
        "target_shares": _target_shares(clean_frame, config.SUBMISSION_TARGET_COLUMN),
    }


def profile_audience_details(raw_frame: pd.DataFrame, clean_frame: pd.DataFrame) -> dict:
    """Record the audiences' own defects: two clocks, odd notations, impossible readings."""
    start_column, end_column = config.START_TIME_COLUMN, config.END_TIME_COLUMN
    target_column = config.AUDIENCE_TARGET_COLUMN
    folded = normalize_category_values(raw_frame, config.AUDIENCE_VALUE_MAPPINGS)
    on_calendar_clock = folded[config.SCHEDULING_SYSTEM_COLUMN].isin(config.CALENDAR_CLOCK_SYSTEMS)
    as_written = convert_clock_times_to_minutes(folded, [start_column, end_column])
    calendar_rows = as_written[on_calendar_clock]

    read_raw = convert_to_broadcast_clock(folded, start_column, [end_column])
    read_raw = convert_audiences_to_thousands(read_raw)
    read_raw = convert_whole_numbers(read_raw, config.AUDIENCE_WHOLE_NUMBER_PATTERNS)

    first_copies = read_raw[~raw_frame.duplicated(keep="first")]
    lowest_target, highest_target = config.AUDIENCE_RANGE_LIMITS[target_column]
    impossible_target = (first_copies[target_column] < lowest_target) | (
        first_copies[target_column] > highest_target
    )
    written_days = raw_frame[config.TRANSMISSION_DAY_COLUMN].dropna()
    return {
        "times_written_on_the_calendar_clock": int(on_calendar_clock.sum()),
        "small_hours_starts_moved_to_the_previous_weekday": _count(
            calendar_rows[start_column] < config.BROADCAST_DAY_START_MINUTE
        ),
        "ends_written_before_their_starts": _count(
            calendar_rows[end_column] < calendar_rows[start_column]
        ),
        "weekdays_not_written_in_full": int((~written_days.isin(config.WEEKDAYS)).sum()),
        "audiences_not_written_in_thousands": {
            column_name: _not_plain_numbers(raw_frame[column_name])
            for column_name in config.AUDIENCE_COLUMNS_IN_THOUSANDS
        },
        "out_of_range_by_column": _out_of_range(read_raw, config.AUDIENCE_RANGE_LIMITS),
        "rows_with_impossible_target": _count(impossible_target),
        "median_under18_viewers": float(clean_frame[target_column].median()),
    }


def profile_note_details(raw_frame: pd.DataFrame, clean_frame: pd.DataFrame) -> dict:
    """Record the defects peculiar to free text: markup, escapes, capitals, stale lengths."""
    text_column, length_column = config.NOTE_TEXT_COLUMN, config.NOTE_LENGTH_COLUMN
    raw_text = raw_frame[text_column].astype(object)
    plain = convert_markup_to_plain_text(raw_frame)[text_column]
    plain_length = plain.str.len()
    raw_length = pd.to_numeric(raw_frame[length_column])

    weekdays = (
        raw_frame[config.NOTE_TIMESTAMP_COLUMN]
        .dropna()
        .map(lambda value: re.fullmatch(config.NOTE_TIMESTAMP_PATTERN, value))
    )
    written_weekdays = weekdays.map(
        lambda found: found.group(config.NOTE_WEEKDAY_PART) if found else None
    )

    first = ~raw_frame.duplicated(keep="first")
    answered_first_copies = first & raw_frame[config.NOTE_TARGET_COLUMN].notna()
    too_few_words = plain.map(word_count, na_action="ignore").fillna(0) < config.MINIMUM_NOTE_WORDS
    clean_words = clean_frame[text_column].map(word_count)
    return {
        "weekdays_not_written_in_full": int((~written_weekdays.isin(config.WEEKDAYS)).sum()),
        "notes_with_markup": _count(raw_text.str.contains(config.MARKUP_PATTERN, regex=True)),
        "notes_with_encoding_artifacts": _count(
            raw_text.str.contains(config.ENCODING_ARTIFACT_PATTERN, regex=True)
        ),
        "notes_in_capitals": _count(plain.map(is_shouted, na_action="ignore")),
        "note_lengths_disagreeing_with_plain_text_as_received": _count(
            raw_length.notna() & plain_length.notna() & (raw_length != plain_length)
        ),
        "note_lengths_disagreeing_with_plain_text_after_cleaning": _count(
            clean_frame[length_column] != clean_frame[text_column].str.len()
        ),
        "rows_with_too_few_words": _count(answered_first_copies & too_few_words),
        "shortest_note_words": int(clean_words.min()),
        "target_shares": _target_shares(clean_frame, config.NOTE_TARGET_COLUMN),
    }


def write_quality_report(profiles: dict[str, dict]) -> None:
    """Write every extract's profile as indented JSON with sorted keys."""
    folder = os.path.dirname(config.DATA_QUALITY_REPORT_FILE)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(config.DATA_QUALITY_REPORT_FILE, "w", encoding="utf-8") as report_file:
        json.dump(profiles, report_file, indent=2, sort_keys=True)
