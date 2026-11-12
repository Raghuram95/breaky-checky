"""Turning a model's raw output into the facts the clearance desk acts on."""

import pandas as pd
from sklearn.pipeline import Pipeline

from config import config
from data_pipeline.feature_builder import add_slot_minutes, build_feature_frame
from data_pipeline.note_cleaning import word_count


def amendment_band(score: float) -> str:
    """Band a rounded amendment score; a score sitting on a threshold belongs to the band above."""
    if score >= config.AMENDMENT_BAND_HIGH_THRESHOLD:
        return config.AMENDMENT_BAND_HIGH
    if score >= config.AMENDMENT_BAND_MEDIUM_THRESHOLD:
        return config.AMENDMENT_BAND_MEDIUM
    return config.AMENDMENT_BAND_LOW


def build_amendment_facts(
    amendment_model: Pipeline, amendment_feature_names: list[str], campaigns: pd.DataFrame
) -> list:
    """Score and band every campaign's chance of needing copy changes, in the table's order."""
    features = build_feature_frame(
        campaigns, config.AMENDMENT_NUMERIC_FEATURES, config.AMENDMENT_CATEGORICAL_FEATURES
    )[amendment_feature_names]
    probabilities = amendment_model.predict_proba(features)[:, 1]
    facts = []
    for probability in probabilities:
        score = round(float(probability), config.AMENDMENT_SCORE_DECIMALS)
        facts.append({"amendment_score": score, "amendment_band": amendment_band(score)})
    return facts


def build_issue_facts(issue_model: Pipeline, campaign_notes: list[dict]) -> dict:
    """Name the issue a campaign's notes describe by a weighted vote, and say how far they agree.

    Each readable note votes for the issue it most probably describes, with the weight of its
    source (a legal review counts twice). A tie goes to the tied issue whose probabilities,
    weighted the same way and summed across the notes, are greatest; never to whichever came first.
    """
    unknown_sources = sorted(
        {note["note_source"] for note in campaign_notes} - set(config.NOTE_SOURCE_WEIGHTS)
    )
    if unknown_sources:
        raise ValueError(f"Notes from sources config gives no weight: {unknown_sources}")
    readable = [
        note
        for note in campaign_notes
        if word_count(note["note_text"]) >= config.MINIMUM_NOTE_WORDS
    ]
    if not readable:
        raise ValueError("No note of this campaign is long enough to read")

    classes = [str(label) for label in issue_model.classes_]
    probabilities = issue_model.predict_proba([note["note_text"] for note in readable])
    votes, confidence, note_readings = {}, dict.fromkeys(classes, 0.0), []
    for note, note_probabilities in zip(readable, probabilities):
        weight = config.NOTE_SOURCE_WEIGHTS[note["note_source"]]
        issue = classes[int(note_probabilities.argmax())]
        votes[issue] = votes.get(issue, 0) + weight
        for label, probability in zip(classes, note_probabilities):
            confidence[label] += weight * float(probability)
        note_readings.append({"note_sequence": note["note_sequence"], "issue_category": issue})

    winner = max(votes, key=lambda issue: (votes[issue], confidence[issue]))
    ranked_votes = dict(sorted(votes.items(), key=lambda item: (-item[1], -confidence[item[0]])))
    return {
        "issue_category": winner,
        "weighted_votes": ranked_votes,
        "weighted_agreement": round(
            votes[winner] / sum(votes.values()), config.NOTE_AGREEMENT_DECIMALS
        ),
        "notes_read": len(readable),
        "notes_too_short_to_read": len(campaign_notes) - len(readable),
        "note_readings": note_readings,
    }


def forecast_programme_audiences(
    exposure_model: Pipeline, exposure_feature_names: list[str], programme_schedule: pd.DataFrame
) -> dict[str, float]:
    """Forecast every booking-week programme's under-18 audience once, in thousands."""
    entry_ids = programme_schedule[config.SCHEDULE_ENTRY_COLUMN]
    if entry_ids.duplicated().any():
        raise ValueError("A schedule entry appears more than once in the booking week")
    features = build_feature_frame(
        add_slot_minutes(programme_schedule),
        config.EXPOSURE_NUMERIC_FEATURES,
        config.EXPOSURE_CATEGORICAL_FEATURES,
    )[exposure_feature_names]
    forecasts = exposure_model.predict(features)
    return {str(entry_id): float(forecast) for entry_id, forecast in zip(entry_ids, forecasts)}


def compute_campaign_exposure(
    spot_results: list[dict], programme_forecasts: dict[str, float]
) -> float:
    """Add up the forecast under-18 viewers the breaching spots would reach, rounded once.

    A spot in a centre break reaches its programme's audience; a spot in an end break reaches the
    larger of its two programmes' audiences.
    """
    exposure = 0.0
    for result in spot_results:
        if not result["breach_codes"]:
            continue
        reached = []
        for programme in result["programmes"]:
            entry_id = programme["schedule_entry_id"]
            if entry_id not in programme_forecasts:
                raise ValueError(f"Programme {entry_id} has no audience forecast")
            reached.append(programme_forecasts[entry_id])
        exposure += max(reached)
    return round(exposure, config.EXPOSURE_DECIMALS)
