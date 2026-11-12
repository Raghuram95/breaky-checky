"""Serving single campaigns out of the live queue.

Each tool works on a table its caller loaded once at start-up, reads no file and changes nothing
it is given. A campaign that cannot be read for certain is refused with a ValueError, which stops
an assessment before it spends a language model call.
"""

import re

import pandas as pd

from config import config
from data_pipeline.common_cleaning import (
    clock_time_to_minutes,
    convert_spot_lengths_to_seconds,
    convert_to_broadcast_clock,
    convert_whole_numbers,
    format_broadcast_clock,
    normalize_category_values,
)

YES_NO_FLAGS = {config.BOOLEAN_YES: True, config.BOOLEAN_NO: False}


def _rows_of(table: type, campaign_reference: str):
    """The rows of a table that belong to one campaign."""
    return table[table[config.CAMPAIGN_REFERENCE_COLUMN] == campaign_reference]


def _is_blank(value) -> bool:
    """True for a missing value or text that is only whitespace."""
    return pd.isna(value) or not str(value).strip()


def fetch_campaign_profile(open_campaigns: pd.DataFrame, campaign_reference: str) -> dict:
    """One campaign's own row from the queue, keyed by column name."""
    rows = _rows_of(open_campaigns, campaign_reference)
    if rows.empty:
        raise ValueError(
            config.CAMPAIGN_NOT_IN_QUEUE_MESSAGE.format(campaign_reference=campaign_reference)
        )
    if len(rows) > 1:
        raise ValueError(f"Campaign {campaign_reference} appears {len(rows)} times in the queue")
    return rows.iloc[0].to_dict()


def fetch_campaign_spots(booked_spots: pd.DataFrame, campaign_reference: str) -> list:
    """Place one campaign's spots on their broadcast day and the broadcast clock."""
    repeated = booked_spots[config.SPOT_REFERENCE_COLUMN].duplicated(keep=False)
    if repeated.any():
        raise ValueError(
            "Spot references repeated in the booking: "
            f"{sorted(set(booked_spots.loc[repeated, config.SPOT_REFERENCE_COLUMN]))[:5]}"
        )
    spots = _rows_of(booked_spots, campaign_reference)
    if spots.empty:
        raise ValueError(f"Campaign {campaign_reference} has no booked spots")
    blank = spots[config.BOOKED_SPOT_REQUIRED_COLUMNS].map(_is_blank)
    if blank.any(axis=None):
        raise ValueError(
            f"Campaign {campaign_reference} has spots missing "
            f"{sorted(blank.columns[blank.any()])}"
        )

    spots = normalize_category_values(spots, config.BOOKED_SPOT_VALUE_MAPPINGS)
    spots = convert_spot_lengths_to_seconds(spots, [config.SPOT_LENGTH_COLUMN])
    non_standard = ~spots[config.SPOT_LENGTH_COLUMN].isin(config.STANDARD_SPOT_LENGTHS)
    if non_standard.any():
        raise ValueError(
            f"Campaign {campaign_reference} books spots of non-standard length: "
            f"{sorted(spots.loc[non_standard, config.SPOT_LENGTH_COLUMN].tolist())}"
        )

    placed = convert_to_broadcast_clock(spots, config.BREAK_TIME_COLUMN, [])
    written = format_broadcast_clock(placed, [config.BREAK_TIME_COLUMN])
    placed = placed.assign(written_break_time=written[config.BREAK_TIME_COLUMN])
    placed = placed.sort_values(config.SPOT_REFERENCE_COLUMN)
    return [
        {
            "spot_reference": row[config.SPOT_REFERENCE_COLUMN],
            "channel": row[config.CHANNEL_COLUMN],
            "broadcast_day": row[config.TRANSMISSION_DAY_COLUMN],
            "break_time": row["written_break_time"],
            "break_minute": int(row[config.BREAK_TIME_COLUMN]),
        }
        for row in placed.to_dict("records")
    ]


def _read_flag(value, column_name: str, entry_id: str):
    """A schedule flag as a bool; anything but Yes or No is refused."""
    if value not in YES_NO_FLAGS:
        raise ValueError(f"Schedule entry {entry_id} has {column_name} {value!r}, not Yes or No")
    return YES_NO_FLAGS[value]


def _read_programme(row: object) -> dict:
    """One schedule row as a programme on the broadcast clock, refusing anything unreadable."""
    entry_id = row[config.SCHEDULE_ENTRY_COLUMN]
    start_minute = clock_time_to_minutes(row[config.START_TIME_COLUMN])
    end_minute = clock_time_to_minutes(row[config.END_TIME_COLUMN])
    index = re.fullmatch(
        config.WHOLE_NUMBER_PATTERN, str(row[config.FORECAST_INDEX_COLUMN]).strip()
    )
    if start_minute is None or end_minute is None or index is None:
        raise ValueError(
            f"Schedule entry {entry_id} leaves a time or its index blank or unreadable"
        )
    if end_minute <= start_minute:
        raise ValueError(f"Schedule entry {entry_id} does not end after it starts")
    if (
        start_minute < config.BROADCAST_DAY_START_MINUTE
        or end_minute > config.BROADCAST_DAY_END_MINUTE
    ):
        raise ValueError(f"Schedule entry {entry_id} runs outside the broadcast day")
    return {
        "schedule_entry_id": entry_id,
        "programme_title": row[config.PROGRAMME_TITLE_COLUMN],
        "start_time": row[config.START_TIME_COLUMN],
        "end_time": row[config.END_TIME_COLUMN],
        "start_minute": start_minute,
        "end_minute": end_minute,
        "forecast_under18_index": int(index.group(1)),
        "live_sport": _read_flag(
            row[config.LIVE_SPORT_FLAG_COLUMN], config.LIVE_SPORT_FLAG_COLUMN, entry_id
        ),
        "made_for_children": _read_flag(
            row[config.MADE_FOR_CHILDREN_FLAG_COLUMN],
            config.MADE_FOR_CHILDREN_FLAG_COLUMN,
            entry_id,
        ),
    }


def fetch_campaign_schedules(programme_schedule: pd.DataFrame, campaign_spots: list[dict]) -> list:
    """The schedule of every channel and broadcast day the spots touch, in start order."""
    touched = sorted(
        {(spot["channel"], spot["broadcast_day"]) for spot in campaign_spots},
        key=lambda key: (config.WEEKDAYS.index(key[1]), key[0]),
    )
    schedules = []
    for channel, broadcast_day in touched:
        rows = programme_schedule[
            (programme_schedule[config.CHANNEL_COLUMN] == channel)
            & (programme_schedule[config.TRANSMISSION_DAY_COLUMN] == broadcast_day)
        ]
        if rows.empty:
            raise ValueError(f"The schedule does not carry {channel} on {broadcast_day}")
        programmes = sorted(
            (_read_programme(row) for row in rows.to_dict("records")),
            key=lambda programme: programme["start_minute"],
        )
        for earlier, later in zip(programmes, programmes[1:]):
            if later["start_minute"] < earlier["end_minute"]:
                raise ValueError(
                    f"{earlier['programme_title']} and {later['programme_title']} "
                    f"overlap on {channel} on {broadcast_day}"
                )
        schedules.append(
            {"channel": channel, "broadcast_day": broadcast_day, "programmes": programmes}
        )
    return schedules


def fetch_campaign_notes(open_campaign_notes: pd.DataFrame, campaign_reference: str) -> list:
    """One campaign's notes, oldest first by their sequence number read as a number."""
    notes = _rows_of(open_campaign_notes, campaign_reference)
    if notes.empty:
        raise ValueError(f"Campaign {campaign_reference} has no notes")
    notes = convert_whole_numbers(notes, config.NOTE_WHOLE_NUMBER_PATTERNS)
    sequences = notes[config.NOTE_SEQUENCE_COLUMN]
    if sequences.isna().any() or sequences.duplicated().any():
        raise ValueError(
            f"The notes of {campaign_reference} do not each carry their own sequence number"
        )
    blank = notes[[config.NOTE_SOURCE_COLUMN, config.NOTE_TEXT_COLUMN]].map(_is_blank)
    if blank.any(axis=None):
        raise ValueError(f"A note of {campaign_reference} has its source or text blank")
    ordered = notes.sort_values(config.NOTE_SEQUENCE_COLUMN)
    return [
        {
            "note_sequence": int(row[config.NOTE_SEQUENCE_COLUMN]),
            "note_source": str(row[config.NOTE_SOURCE_COLUMN]).strip(),
            "note_text": str(row[config.NOTE_TEXT_COLUMN]).strip(),
        }
        for row in ordered.to_dict("records")
    ]
