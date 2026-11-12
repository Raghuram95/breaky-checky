"""Saving and loading the models, their feature schemas and their scores.

Nothing else in the project writes to output/models/. A model is saved with joblib as the whole
fitted Pipeline, so its input preparation travels with it and loads back scoring exactly as saved.
"""

import json
import os

import joblib
from sklearn.pipeline import Pipeline

from config import config


def _ensure_folder(file_path: str) -> None:
    """Create the folder a file belongs in, if it does not exist yet."""
    folder = os.path.dirname(file_path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def _write_json(payload, file_path: str) -> None:
    """Write a JSON file, indented with sorted keys."""
    _ensure_folder(file_path)
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)


def save_model(model: Pipeline, file_path: str) -> None:
    """Save a fitted Pipeline, preparation included."""
    _ensure_folder(file_path)
    joblib.dump(model, file_path)


def save_feature_schema(feature_names: list[str], file_path: str) -> None:
    """Record the exact input columns a model was fitted on, in order."""
    _write_json(list(feature_names), file_path)


def save_metrics(metrics: dict[str, dict]) -> None:
    """Save every model's scores to the metrics file."""
    _write_json(metrics, config.MODEL_METRICS_FILE)


def load_model(file_path: str) -> Pipeline:
    """Load a saved Pipeline, ready to score."""
    return joblib.load(file_path)


def load_feature_schema(file_path: str) -> list[str]:
    """Load a model's input column names, in the order they were saved."""
    with open(file_path, encoding="utf-8") as schema_file:
        return json.load(schema_file)
