"""Entry point 3: train, score and save the three models.

Run from the Project folder:  python run_model_training.py
"""

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config import config
from data_pipeline.dataset_loader import load_processed_dataset
from data_pipeline.feature_builder import add_slot_minutes, build_feature_frame
from modeling.amendment_model import train_amendment_model
from modeling.exposure_model import train_exposure_model
from modeling.issue_model import train_issue_model
from modeling.model_registry import save_feature_schema, save_metrics, save_model


def _metrics_record(
    model: Pipeline, scores, training_rows: int, held_out_rows: int, score_name: str, minimum: float
) -> dict:
    """The record kept for one model in the metrics file."""
    return {
        "algorithm": type(model.steps[-1][1]).__name__,
        "training_rows": int(training_rows),
        "held_out_rows": int(held_out_rows),
        "scores": scores,
        "minimum_required": {score_name: minimum},
        "meets_minimum": bool(scores[score_name] >= minimum),
        "scikit_learn_version": sklearn.__version__,
    }


def train_amendment() -> dict:
    """Fit the amendment model on the cleaned training submissions; score on the held-out ones."""
    train_frame = load_processed_dataset(config.CLEAN_SUBMISSIONS_TRAIN_FILE)
    held_out_frame = load_processed_dataset(config.CLEAN_SUBMISSIONS_EVAL_FILE)
    features = [
        build_feature_frame(
            frame, config.AMENDMENT_NUMERIC_FEATURES, config.AMENDMENT_CATEGORICAL_FEATURES
        )
        for frame in (train_frame, held_out_frame)
    ]
    targets = [
        frame[config.SUBMISSION_TARGET_COLUMN].map(config.SUBMISSION_TARGET_ENCODING).astype(int)
        for frame in (train_frame, held_out_frame)
    ]
    model, scores = train_amendment_model(features[0], targets[0], features[1], targets[1])
    save_model(model, config.AMENDMENT_MODEL_FILE)
    save_feature_schema(list(features[0].columns), config.AMENDMENT_SCHEMA_FILE)
    print(
        f"amendment held-out ROC AUC {scores['held_out_roc_auc']:.4f} "
        f"(train {scores['train_roc_auc']:.4f}, gap {scores['generalisation_gap']:+.4f})"
    )
    return _metrics_record(
        model,
        scores,
        len(train_frame),
        len(held_out_frame),
        "held_out_roc_auc",
        config.MINIMUM_AMENDMENT_ROC_AUC,
    )


def train_exposure() -> dict:
    """Fit the exposure model on the cleaned training audiences; score on the held-out weeks."""
    frames = [
        add_slot_minutes(load_processed_dataset(path))
        for path in (config.CLEAN_AUDIENCES_TRAIN_FILE, config.CLEAN_AUDIENCES_EVAL_FILE)
    ]
    features = [
        build_feature_frame(
            frame, config.EXPOSURE_NUMERIC_FEATURES, config.EXPOSURE_CATEGORICAL_FEATURES
        )
        for frame in frames
    ]
    targets = [frame[config.AUDIENCE_TARGET_COLUMN] for frame in frames]
    model, scores = train_exposure_model(features[0], targets[0], features[1], targets[1])
    save_model(model, config.EXPOSURE_MODEL_FILE)
    save_feature_schema(list(features[0].columns), config.EXPOSURE_SCHEMA_FILE)
    print(
        f"exposure  held-out R2 {scores['held_out_r2']:.4f} "
        f"MAE {scores['mean_absolute_error']:.2f} thousand "
        f"(train {scores['train_r2']:.4f}, gap {scores['generalisation_gap']:+.4f})"
    )
    return _metrics_record(
        model, scores, len(frames[0]), len(frames[1]), "held_out_r2", config.MINIMUM_EXPOSURE_R2
    )


def train_issue() -> dict:
    """Split the notes once, stratified by issue, and fit the issue model on the larger part."""
    notes = load_processed_dataset(config.CLEAN_CLEARANCE_NOTES_FILE)
    train_text, held_out_text, train_target, held_out_target = train_test_split(
        notes[config.NOTE_TEXT_COLUMN],
        notes[config.NOTE_TARGET_COLUMN],
        test_size=config.NOTE_HELD_OUT_SHARE,
        stratify=notes[config.NOTE_TARGET_COLUMN],
        random_state=config.RANDOM_SEED,
    )
    model, scores = train_issue_model(train_text, train_target, held_out_text, held_out_target)
    save_model(model, config.ISSUE_MODEL_FILE)
    print(
        f"issue     held-out macro F1 {scores['held_out_macro_f1']:.4f} "
        f"(train {scores['train_macro_f1']:.4f}, gap {scores['generalisation_gap']:+.4f})"
    )
    return _metrics_record(
        model,
        scores,
        len(train_text),
        len(held_out_text),
        "held_out_macro_f1",
        config.MINIMUM_ISSUE_MACRO_F1,
    )


def main() -> None:
    """Train the three models, save them with their schemas and scores, and report the minimums."""
    print("training three models")
    metrics = {"amendment": train_amendment(), "exposure": train_exposure(), "issue": train_issue()}
    save_metrics(metrics)
    print(f"metrics written to {config.MODEL_METRICS_FILE}")
    below_minimum = [name for name, record in metrics.items() if not record["meets_minimum"]]
    if below_minimum:
        print("below the minimum config sets: " + ", ".join(below_minimum))
    else:
        print("every model reaches the minimum config sets for it")


if __name__ == "__main__":
    main()
