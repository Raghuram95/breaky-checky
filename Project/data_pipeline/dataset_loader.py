"""Reading and writing the project's CSV files.

This is the only data-preparation module that touches CSV files on disk. Raw extracts are read
as plain text so that nothing is guessed about their contents; cleaned tables are read with
their column types inferred, because the pipeline has already made every value trustworthy.
"""

import os

import pandas as pd

from config import config


def load_raw_dataset(file_path: str) -> pd.DataFrame:
    """Read one raw extract as text, turning every configured null token into a missing value.

    Every value is stripped of surrounding whitespace first, so " N/A " and "n/a" are both seen
    as missing, while a value that merely contains a token (for example "Not recorded yet") is
    kept as written.
    """
    raw_frame = pd.read_csv(file_path, dtype=str, keep_default_na=False)
    stripped_frame = raw_frame.apply(lambda column: column.str.strip())
    null_tokens = {token.lower() for token in config.NULL_TOKENS}
    is_null_token = stripped_frame.apply(lambda column: column.str.lower().isin(null_tokens))
    return stripped_frame.mask(is_null_token)


def load_processed_dataset(file_path: str) -> pd.DataFrame:
    """Read a cleaned table with each column's type inferred; only an empty cell is missing."""
    return pd.read_csv(file_path, keep_default_na=False, na_values=[""], low_memory=False)


def save_processed_dataset(dataframe: pd.DataFrame, file_path: str) -> None:
    """Write a cleaned table without an index column, creating its folder when needed."""
    folder = os.path.dirname(file_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    dataframe.to_csv(file_path, index=False)
