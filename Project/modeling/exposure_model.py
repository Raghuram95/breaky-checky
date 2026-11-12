"""The model that forecasts how many viewers under 18 a programme will have, in thousands.

A histogram gradient boosting regressor with Poisson loss. Audiences are non-negative and
heavily skewed (median about 12 thousand, a long tail past 1,500 thousand), which is the shape a
Poisson model is built for. Poisson loss predicts through an exponential, so every forecast is
above nought, which a squared-error model did not guarantee (it forecast negative audiences).
Boosted trees also capture interactions such as "late slot on a children's channel" on their own.
"""

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from config import config
from data_pipeline.feature_builder import build_model_pipeline
from modeling.model_evaluation import evaluate_regressor

BOOSTING_ROUNDS = 400
LEARNING_RATE = 0.05


def train_exposure_model(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    held_out_features: pd.DataFrame,
    held_out_target: pd.Series,
) -> tuple[Pipeline, dict]:
    """Fit the exposure pipeline on the training rows only, and score it on both sets."""
    estimator = HistGradientBoostingRegressor(
        loss="poisson",
        max_iter=BOOSTING_ROUNDS,
        learning_rate=LEARNING_RATE,
        random_state=config.RANDOM_SEED,
    )
    model = build_model_pipeline(
        estimator, config.EXPOSURE_NUMERIC_FEATURES, config.EXPOSURE_CATEGORICAL_FEATURES
    )
    model.fit(train_features, train_target)
    scores = evaluate_regressor(
        model, train_features, train_target, held_out_features, held_out_target
    )
    return model, scores
