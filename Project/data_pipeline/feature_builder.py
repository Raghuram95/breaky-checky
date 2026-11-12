"""Turning cleaned tables into model inputs.

The preparation a model needs (filling gaps, scaling numbers, one-hot encoding categories) is
built here as part of a scikit-learn Pipeline, so it is learned only from the training rows and
saved inside the model file. Scoring a live campaign then applies exactly the same preparation.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import config
from data_pipeline.common_cleaning import convert_clock_times_to_minutes

MISSING_CATEGORY = "Not recorded"


def add_slot_minutes(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add each programme's start minute on the broadcast clock and its length in minutes."""
    with_slot = dataframe.copy()
    minutes = convert_clock_times_to_minutes(
        dataframe[[config.START_TIME_COLUMN, config.END_TIME_COLUMN]],
        [config.START_TIME_COLUMN, config.END_TIME_COLUMN],
    )
    with_slot[config.START_MINUTE_COLUMN] = minutes[config.START_TIME_COLUMN]
    with_slot[config.DURATION_MINUTES_COLUMN] = (
        minutes[config.END_TIME_COLUMN] - minutes[config.START_TIME_COLUMN]
    )
    return with_slot


def _as_float(value):
    """A number as a float, and any kind of gap (None, pd.NA, NaN, '') as NaN."""
    if pd.isna(value) or value == "":
        return np.nan
    return float(value)


def _as_category(value):
    """A category as text, and any kind of gap as NaN."""
    if pd.isna(value) or value == "":
        return np.nan
    return str(value)


def build_feature_frame(
    dataframe: pd.DataFrame, numeric_columns: list[str], categorical_columns: list[str]
) -> pd.DataFrame:
    """Select a model's inputs: the numeric columns as floats, then the categories as text."""
    features = pd.DataFrame(index=dataframe.index)
    for column_name in numeric_columns:
        features[column_name] = dataframe[column_name].map(_as_float).astype(float)
    for column_name in categorical_columns:
        features[column_name] = dataframe[column_name].map(_as_category).astype(object)
    return features


def build_model_pipeline(
    estimator, numeric_columns: list[str], categorical_columns: list[str]
) -> Pipeline:
    """Join the input preparation and the estimator into one unfitted Pipeline.

    Numbers: a gap is filled with the training median and a flag column records that it was a
    gap, then the numbers are scaled. Categories: a gap becomes its own category, and a category
    never seen in training is ignored rather than refused.
    """
    numeric_preparation = Pipeline(
        [
            ("fill_gaps", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_preparation = Pipeline(
        [
            ("fill_gaps", SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY)),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preparation = ColumnTransformer(
        [
            ("numbers", numeric_preparation, list(numeric_columns)),
            ("categories", categorical_preparation, list(categorical_columns)),
        ]
    )
    return Pipeline([("preparation", preparation), ("estimator", estimator)])
