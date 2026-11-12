"""The NLP model that reads a clearance note and names the copy issue it describes.

TF-IDF over words and word pairs feeds a multinomial logistic regression. Notes are short and
the issues are signalled by distinctive vocabulary ("free bet", "APR", "recyclable"), which is
exactly what TF-IDF captures; the bigrams keep phrases like "no evidence" together. Logistic
regression gives a probability for every issue, which the weighted vote and its tie-break need,
and balanced class weights stop the commonest issues from swamping the rare ones (macro F1).
The vectoriser is inside the Pipeline, so the saved model takes raw note text directly.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config import config
from modeling.model_evaluation import evaluate_text_classifier

WORD_NGRAM_RANGE = (1, 2)
MINIMUM_DOCUMENT_FREQUENCY = 2
MAXIMUM_SOLVER_ITERATIONS = 2000


def train_issue_model(
    train_text: pd.Series,
    train_target: pd.Series,
    held_out_text: pd.Series,
    held_out_target: pd.Series,
) -> tuple[Pipeline, dict]:
    """Fit the text pipeline on the training notes only, and score it on both sets."""
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=WORD_NGRAM_RANGE,
                    min_df=MINIMUM_DOCUMENT_FREQUENCY,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=MAXIMUM_SOLVER_ITERATIONS,
                    class_weight="balanced",
                    random_state=config.RANDOM_SEED,
                ),
            ),
        ]
    )
    model.fit(train_text, train_target)
    scores = evaluate_text_classifier(
        model, train_text, train_target, held_out_text, held_out_target
    )
    return model, scores
