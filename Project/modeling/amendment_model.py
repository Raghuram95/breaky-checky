"""The model that predicts whether a campaign's copy will need changes before it can be cleared.

Logistic regression was chosen over gradient boosting after comparing both on the held-out
submissions: it ranked them slightly better (ROC AUC about 0.86 against 0.85), memorised less
(a smaller train/held-out gap), gives well calibrated probabilities (the score is banded at fixed
thresholds, so calibration matters), and each flag's effect can be read off its coefficient.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config import config
from data_pipeline.feature_builder import build_model_pipeline
from modeling.model_evaluation import evaluate_classifier

MAXIMUM_SOLVER_ITERATIONS = 2000


def train_amendment_model(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    held_out_features: pd.DataFrame,
    held_out_target: pd.Series,
) -> tuple[Pipeline, dict]:
    """Fit the amendment pipeline on the training rows only, and score it on both sets."""
    estimator = LogisticRegression(
        max_iter=MAXIMUM_SOLVER_ITERATIONS, random_state=config.RANDOM_SEED
    )
    model = build_model_pipeline(
        estimator, config.AMENDMENT_NUMERIC_FEATURES, config.AMENDMENT_CATEGORICAL_FEATURES
    )
    model.fit(train_features, train_target)
    scores = evaluate_classifier(
        model, train_features, train_target, held_out_features, held_out_target
    )
    return model, scores
