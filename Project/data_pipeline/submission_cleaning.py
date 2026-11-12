"""The cleaning run for the campaign submission extracts.

Besides the repairs every extract needs, a submission can contradict itself in two ways only it
can hold: legal small print on screen for longer than the copy runs, and more amended campaigns
than campaigns. The clock number is the copy's identity in the playout system, so when the
written copy length disagrees with the length in the clock number, the clock number wins.
"""

import re

import pandas as pd

from config import config
from data_pipeline.common_cleaning import (
    convert_decimal_numbers,
    convert_spot_lengths_to_seconds,
    convert_whole_numbers,
    normalize_category_values,
    refuse_values,
    remove_duplicate_records,
    remove_rows_missing_target,
    treat_out_of_range_as_missing,
)
from data_pipeline.dataset_loader import load_raw_dataset, save_processed_dataset


def clock_number_lengths(clock_numbers: pd.Series) -> pd.Series:
    """Read the copy length in seconds from each clock number's last three digits.

    A clock number that carries no length, or a length that is not a standard spot length,
    is refused: the copy's length could not then be known for certain.
    """

    def read_length(clock_number):
        found = re.fullmatch(config.CLOCK_NUMBER_PATTERN, str(clock_number).strip())
        return int(found.group(1)) if found else None

    lengths = clock_numbers.map(read_length)
    unreadable = lengths.isna()
    if unreadable.any():
        refuse_values(
            config.CLOCK_NUMBER_COLUMN,
            clock_numbers[unreadable],
            "holds clock numbers that carry no length",
        )
    non_standard = ~lengths.isin(config.STANDARD_SPOT_LENGTHS)
    if non_standard.any():
        refuse_values(
            config.CLOCK_NUMBER_COLUMN,
            clock_numbers[non_standard],
            "holds clock numbers whose length is not a standard spot length",
        )
    return lengths.astype("Int64")


def blank_contradictions(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Blank the two readings that contradict another fact of the same submission."""
    repaired = dataframe.copy()
    legal_too_long = (
        (repaired[config.LEGAL_TEXT_SECONDS_COLUMN] > repaired[config.COPY_LENGTH_COLUMN])
        .fillna(False)
        .astype(bool)
    )
    repaired.loc[legal_too_long, config.LEGAL_TEXT_SECONDS_COLUMN] = None
    too_many_amendments = (
        (
            repaired[config.ADVERTISER_AMENDMENTS_COLUMN]
            > repaired[config.ADVERTISER_CAMPAIGNS_COLUMN]
        )
        .fillna(False)
        .astype(bool)
    )
    repaired.loc[too_many_amendments, config.ADVERTISER_AMENDMENTS_COLUMN] = None
    return repaired


def clean_submission_dataset(input_file_path: str, output_file_path: str) -> pd.DataFrame:
    """Produce a submissions table a model can be fitted on, and save it with the raw columns."""
    raw_frame = load_raw_dataset(input_file_path)
    cleaned = remove_duplicate_records(raw_frame)
    cleaned = remove_rows_missing_target(cleaned, config.SUBMISSION_TARGET_COLUMN)
    cleaned = normalize_category_values(cleaned, config.SUBMISSION_VALUE_MAPPINGS)
    cleaned = convert_whole_numbers(cleaned, config.SUBMISSION_WHOLE_NUMBER_PATTERNS)
    cleaned = convert_decimal_numbers(cleaned)
    cleaned = convert_spot_lengths_to_seconds(cleaned, [config.COPY_LENGTH_COLUMN])
    cleaned[config.COPY_LENGTH_COLUMN] = clock_number_lengths(cleaned[config.CLOCK_NUMBER_COLUMN])
    cleaned = treat_out_of_range_as_missing(cleaned, config.SUBMISSION_RANGE_LIMITS)
    cleaned = blank_contradictions(cleaned)
    cleaned = cleaned[list(raw_frame.columns)]
    save_processed_dataset(cleaned, output_file_path)
    return cleaned
