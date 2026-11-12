"""Entry point 2: clean the five raw extracts and write the data quality report.

Run from the Project folder:  python run_data_pipeline.py
"""

import pathlib

from config import config
from data_pipeline.audience_cleaning import clean_audience_dataset
from data_pipeline.dataset_loader import load_raw_dataset
from data_pipeline.note_cleaning import clean_note_dataset
from data_pipeline.quality_profiler import (
    profile_audience_details,
    profile_dataset,
    profile_note_details,
    profile_submission_details,
    write_quality_report,
)
from data_pipeline.submission_cleaning import clean_submission_dataset

# Each job: (raw file, cleaned file, cleaning function, answer column, detail profiler).
DATASET_JOBS = [
    (
        config.RAW_SUBMISSIONS_TRAIN_FILE,
        config.CLEAN_SUBMISSIONS_TRAIN_FILE,
        clean_submission_dataset,
        config.SUBMISSION_TARGET_COLUMN,
        profile_submission_details,
    ),
    (
        config.RAW_SUBMISSIONS_EVAL_FILE,
        config.CLEAN_SUBMISSIONS_EVAL_FILE,
        clean_submission_dataset,
        config.SUBMISSION_TARGET_COLUMN,
        profile_submission_details,
    ),
    (
        config.RAW_AUDIENCES_TRAIN_FILE,
        config.CLEAN_AUDIENCES_TRAIN_FILE,
        clean_audience_dataset,
        config.AUDIENCE_TARGET_COLUMN,
        profile_audience_details,
    ),
    (
        config.RAW_AUDIENCES_EVAL_FILE,
        config.CLEAN_AUDIENCES_EVAL_FILE,
        clean_audience_dataset,
        config.AUDIENCE_TARGET_COLUMN,
        profile_audience_details,
    ),
    (
        config.RAW_CLEARANCE_NOTES_FILE,
        config.CLEAN_CLEARANCE_NOTES_FILE,
        clean_note_dataset,
        config.NOTE_TARGET_COLUMN,
        profile_note_details,
    ),
]


def main() -> None:
    """Clean every extract in order, profile it as received and as cleaned, and write the report."""
    profiles = {}
    for raw_path, clean_path, clean_dataset, target_column, profile_details in DATASET_JOBS:
        raw_frame = load_raw_dataset(raw_path)
        clean_frame = clean_dataset(raw_path, clean_path)
        profile = profile_dataset(raw_frame, clean_frame, target_column)
        profile.update(profile_details(raw_frame, clean_frame))
        name = pathlib.Path(raw_path).stem
        profiles[name] = profile
        print(
            f"{name:<28} {profile['rows_received']:>6} received -> "
            f"{profile['rows_kept']:>6} kept  ({profile['rows_removed']} removed)"
        )
    write_quality_report(profiles)
    print(f"quality report written to {config.DATA_QUALITY_REPORT_FILE}")


if __name__ == "__main__":
    main()
