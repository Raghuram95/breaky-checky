"""The cleaning run for the programme audience extracts.

Four channels are scheduled on three systems with two different clocks. Every programme is moved
onto its broadcast day and the broadcast clock, so that 01:10 on a Sunday written by one system
and 25:10 on the Saturday written by another become the same moment.
"""

import pandas as pd

from config import config
from data_pipeline.common_cleaning import (
    convert_audiences_to_thousands,
    convert_to_broadcast_clock,
    convert_whole_numbers,
    format_broadcast_clock,
    normalize_category_values,
    remove_duplicate_records,
    remove_rows_missing_target,
    treat_out_of_range_as_missing,
)
from data_pipeline.dataset_loader import load_raw_dataset, save_processed_dataset


def clean_audience_dataset(input_file_path: str, output_file_path: str) -> pd.DataFrame:
    """Produce an audiences table a model can be fitted on, every programme on its broadcast day."""
    raw_frame = load_raw_dataset(input_file_path)
    time_columns = [config.START_TIME_COLUMN, config.END_TIME_COLUMN]
    cleaned = remove_duplicate_records(raw_frame)
    cleaned = remove_rows_missing_target(cleaned, config.AUDIENCE_TARGET_COLUMN)
    cleaned = normalize_category_values(cleaned, config.AUDIENCE_VALUE_MAPPINGS)
    cleaned = convert_to_broadcast_clock(
        cleaned, config.START_TIME_COLUMN, [config.END_TIME_COLUMN]
    )
    cleaned = convert_audiences_to_thousands(cleaned)
    cleaned = convert_whole_numbers(cleaned, config.AUDIENCE_WHOLE_NUMBER_PATTERNS)
    cleaned = treat_out_of_range_as_missing(cleaned, config.AUDIENCE_RANGE_LIMITS)
    cleaned = remove_rows_missing_target(cleaned, config.AUDIENCE_TARGET_COLUMN)
    cleaned = format_broadcast_clock(cleaned, time_columns)
    cleaned = cleaned[list(raw_frame.columns)]
    save_processed_dataset(cleaned, output_file_path)
    return cleaned
