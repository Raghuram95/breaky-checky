"""Scoring fitted models on held-out rows and on their training rows.

No function here fits anything. Each reports the held-out score (how well the model generalises),
the training score, and the generalisation gap between them (how much it memorised). The gap is
worked out before rounding, and every score is rounded to config.METRIC_DECIMALS places.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from config import config


def _rounded(value) -> float:
    """Round a score to the configured number of places, as a plain float."""
    return round(float(value), config.METRIC_DECIMALS)


def evaluate_classifier(
    model: Pipeline,
    train_features: pd.DataFrame,
    train_target: pd.Series,
    held_out_features: pd.DataFrame,
    held_out_target: pd.Series,
) -> dict:
    """Score the amendment model: ranking quality (ROC AUC) plus the usual threshold metrics."""
    held_out_probability = model.predict_proba(held_out_features)[:, 1]
    train_probability = model.predict_proba(train_features)[:, 1]
    held_out_prediction = model.predict(held_out_features)
    held_out_auc = roc_auc_score(held_out_target, held_out_probability)
    train_auc = roc_auc_score(train_target, train_probability)
    return {
        "held_out_roc_auc": _rounded(held_out_auc),
        "train_roc_auc": _rounded(train_auc),
        "generalisation_gap": _rounded(train_auc - held_out_auc),
        "accuracy": _rounded(accuracy_score(held_out_target, held_out_prediction)),
        "precision": _rounded(
            precision_score(held_out_target, held_out_prediction, zero_division=0)
        ),
        "recall": _rounded(recall_score(held_out_target, held_out_prediction, zero_division=0)),
        "f1": _rounded(f1_score(held_out_target, held_out_prediction, zero_division=0)),
        "mean_predicted_probability": _rounded(np.mean(held_out_probability)),
        "positive_share": _rounded(np.mean(np.asarray(held_out_target, dtype=float))),
    }


def evaluate_regressor(
    model: Pipeline,
    train_features: pd.DataFrame,
    train_target: pd.Series,
    held_out_features: pd.DataFrame,
    held_out_target: pd.Series,
) -> dict:
    """Score the exposure model against the baseline of always answering the training mean."""
    held_out_prediction = np.asarray(model.predict(held_out_features), dtype=float)
    train_prediction = np.asarray(model.predict(train_features), dtype=float)
    held_out_r2 = r2_score(held_out_target, held_out_prediction)
    train_r2 = r2_score(train_target, train_prediction)
    baseline = np.full(len(held_out_target), float(np.mean(np.asarray(train_target, dtype=float))))
    return {
        "held_out_r2": _rounded(held_out_r2),
        "train_r2": _rounded(train_r2),
        "generalisation_gap": _rounded(train_r2 - held_out_r2),
        "mean_absolute_error": _rounded(mean_absolute_error(held_out_target, held_out_prediction)),
        "average_estimate_mean_absolute_error": _rounded(
            mean_absolute_error(held_out_target, baseline)
        ),
        "lowest_prediction": _rounded(held_out_prediction.min()),
    }


def evaluate_text_classifier(
    model: Pipeline,
    train_text: pd.Series,
    train_target: pd.Series,
    held_out_text: pd.Series,
    held_out_target: pd.Series,
) -> dict:
    """Score the issue model; macro F1 weighs the rarest issue as much as the commonest."""
    held_out_prediction = model.predict(held_out_text)
    train_prediction = model.predict(train_text)
    held_out_macro = f1_score(held_out_target, held_out_prediction, average="macro")
    train_macro = f1_score(train_target, train_prediction, average="macro")
    classes = [str(label) for label in model.classes_]
    per_class = f1_score(held_out_target, held_out_prediction, average=None, labels=model.classes_)
    return {
        "held_out_macro_f1": _rounded(held_out_macro),
        "train_macro_f1": _rounded(train_macro),
        "generalisation_gap": _rounded(train_macro - held_out_macro),
        "accuracy": _rounded(accuracy_score(held_out_target, held_out_prediction)),
        "weighted_f1": _rounded(f1_score(held_out_target, held_out_prediction, average="weighted")),
        "per_class_f1": {label: _rounded(score) for label, score in zip(classes, per_class)},
    }
