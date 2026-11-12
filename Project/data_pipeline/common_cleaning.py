"""Repairs shared by the cleaning runs, the quality profiler and the live-booking tools.

Every function takes a table and returns a repaired copy, leaving its input untouched. None of
them reads or writes a file or removes a column. A value written in a notation the configuration
does not know is never guessed at: it is refused with a ValueError naming the column and values.
"""

import html
import re
from collections.abc import Callable

import numpy as np
import pandas as pd

from config import config

REFUSED_VALUE_SAMPLE_SIZE = 5


def _as_text(series: pd.Series) -> str:
    """Return the series as stripped text, keeping missing values missing."""
    return series.map(lambda value: str(value).strip(), na_action="ignore")


def refuse_values(column_name: str, values: list, reason: str) -> None:
    """Raise the ValueError every repair uses, naming the column and a sample of the values."""
    sample = sorted({str(value) for value in values})[:REFUSED_VALUE_SAMPLE_SIZE]
    raise ValueError(f"Column '{column_name}' {reason}: {sample}")


def _parse_column(
    dataframe: pd.DataFrame, column_name: str, parse_value: Callable, reason: str, result_dtype: str
):
    """Run parse_value over one column; it returns a number, or None for a value it cannot read.

    Gaps stay gaps. Any value the parser cannot read is refused, all of them named together.
    """
    text = _as_text(dataframe[column_name])
    parsed = text.map(parse_value, na_action="ignore")
    unreadable = text.notna() & parsed.isna()
    if unreadable.any():
        refuse_values(column_name, text[unreadable], reason)
    return parsed.astype(result_dtype)


def _first_group_as_int(pattern: str):
    """Build a parser that reads the first group of a pattern as a whole number."""

    def parse(value):
        found = re.fullmatch(pattern, value)
        return int(found.group(1)) if found else None

    return parse


def remove_duplicate_records(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Keep the first of every set of rows that are identical in every column."""
    return dataframe.drop_duplicates(keep="first").reset_index(drop=True)


def normalize_category_values(
    dataframe: pd.DataFrame, value_mappings: dict[str, dict]
) -> pd.DataFrame:
    """Fold every spelling, case and code of a category value onto its one canonical form."""
    folded = dataframe.copy()
    for column_name, mapping in value_mappings.items():
        if column_name not in folded.columns:
            continue
        lowered = _as_text(folded[column_name]).map(str.lower, na_action="ignore")
        canonical = lowered.map(mapping)
        unknown = lowered.notna() & canonical.isna()
        if unknown.any():
            refuse_values(
                column_name,
                folded.loc[unknown, column_name],
                "holds values its mapping does not recognise",
            )
        folded[column_name] = canonical.astype(object)
    return folded


def remove_rows_missing_target(dataframe: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Drop exactly the rows that carry no answer, renumbering what is left from nought."""
    return dataframe[dataframe[target_column].notna()].reset_index(drop=True)


def convert_whole_numbers(dataframe: pd.DataFrame, number_patterns: dict[str, str]) -> pd.DataFrame:
    """Read counts and whole percentages written with a unit or a sign as whole numbers."""
    converted = dataframe.copy()
    for column_name, pattern in number_patterns.items():
        if column_name in converted.columns:
            converted[column_name] = _parse_column(
                converted,
                column_name,
                _first_group_as_int(pattern),
                "holds values that are not whole numbers in a known notation",
                "Int64",
            )
    return converted


def convert_decimal_numbers(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Read the submissions' decimal percentages, such as '41.1 %', as numbers."""
    converted = dataframe.copy()
    for column_name, pattern in config.SUBMISSION_DECIMAL_NUMBER_PATTERNS.items():
        if column_name not in converted.columns:
            continue

        def parse(value, pattern: str = pattern):
            found = re.fullmatch(pattern, value)
            return float(found.group(1)) if found else None

        converted[column_name] = _parse_column(
            converted,
            column_name,
            parse,
            "holds values that are not decimal numbers in a known notation",
            "float",
        )
    return converted


def _length_in_seconds(value):
    """Read a length written as seconds, minutes and seconds, or hours, minutes and seconds."""
    found = re.fullmatch(config.SECONDS_NOTATION_PATTERN, value)
    if found:
        return int(found.group(1))
    found = re.fullmatch(config.MINUTES_AND_SECONDS_NOTATION_PATTERN, value)
    if found:
        return int(found.group(1)) * config.SECONDS_PER_MINUTE + int(found.group(2))
    found = re.fullmatch(config.HOURS_MINUTES_AND_SECONDS_NOTATION_PATTERN, value)
    if found:
        hours, minutes, seconds = (int(part) for part in found.groups())
        return hours * config.SECONDS_PER_HOUR + minutes * config.SECONDS_PER_MINUTE + seconds
    return None


def convert_spot_lengths_to_seconds(
    dataframe: pd.DataFrame, length_columns: list[str]
) -> pd.DataFrame:
    """Turn every length notation the schedulers use into whole seconds."""
    converted = dataframe.copy()
    for column_name in length_columns:
        converted[column_name] = _parse_column(
            converted,
            column_name,
            _length_in_seconds,
            "holds lengths in no known notation",
            "Int64",
        )
    return converted


def _audience_in_thousands(value):
    """Read an audience written in millions, in full viewers or in thousands, as thousands."""
    found = re.fullmatch(config.MILLIONS_NOTATION_PATTERN, value)
    if found:
        thousands = float(found.group(1)) * config.THOUSANDS_PER_MILLION
    elif re.fullmatch(config.VIEWERS_NOTATION_PATTERN, value):
        viewers = float(value.replace(config.DIGIT_GROUPING_SEPARATOR, ""))
        thousands = viewers / config.VIEWERS_PER_THOUSAND
    else:
        found = re.fullmatch(config.THOUSANDS_NOTATION_PATTERN, value)
        if not found:
            return None
        thousands = float(found.group(1))
    return round(thousands, config.AUDIENCE_DECIMALS)


def convert_audiences_to_thousands(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Put every audience column on one scale, thousands of viewers."""
    converted = dataframe.copy()
    for column_name in config.AUDIENCE_COLUMNS_IN_THOUSANDS:
        converted[column_name] = _parse_column(
            converted,
            column_name,
            _audience_in_thousands,
            "holds audiences in no known notation",
            "float",
        )
    return converted


def clock_time_to_minutes(value) -> int | None:
    """Read one HH:MM clock time as minutes after midnight, hours past 23 included; else None."""
    found = re.fullmatch(config.CLOCK_TIME_PATTERN, str(value).strip())
    if not found:
        return None
    return int(found.group(1)) * config.MINUTES_PER_HOUR + int(found.group(2))


def convert_clock_times_to_minutes(
    dataframe: pd.DataFrame, time_columns: list[str]
) -> pd.DataFrame:
    """Read each named HH:MM column as minutes after midnight, so 25:10 becomes 1510."""
    converted = dataframe.copy()
    for column_name in time_columns:
        converted[column_name] = _parse_column(
            converted,
            column_name,
            clock_time_to_minutes,
            "holds times that are not written HH:MM",
            "Int64",
        )
    return converted


def _refuse_rows(
    dataframe: pd.DataFrame, refused_rows: pd.Series, columns: list[str], reason: str
) -> None:
    """Refuse a set of rows, naming the columns involved and the values they hold."""
    if refused_rows.any():
        values = dataframe.loc[refused_rows, columns].map(str).agg(" ".join, axis=1)
        refuse_values("/".join(columns), values, reason)


def convert_to_broadcast_clock(
    dataframe: pd.DataFrame, start_column: str, end_columns: list[str]
) -> pd.DataFrame:
    """Place every time on its broadcast day and the broadcast clock, whichever clock wrote it.

    Broadcast-clock systems already write the broadcast day (06:00 to 29:59). Calendar-clock
    systems write the ordinary clock on the calendar day: a start before 06:00 belongs to the
    previous weekday's broadcast day, 24 hours later on its clock, and an end at or before 06:00
    is 24 hours later too. Anything that cannot be placed for certain is refused.
    """
    day_column = config.TRANSMISSION_DAY_COLUMN
    system_column = config.SCHEDULING_SYSTEM_COLUMN
    placed = dataframe.copy()

    systems = placed[system_column]
    known_systems = config.BROADCAST_CLOCK_SYSTEMS + config.CALENDAR_CLOCK_SYSTEMS
    _refuse_rows(
        placed,
        ~systems.isin(known_systems),
        [system_column],
        "names a scheduling system that is missing or unknown",
    )
    _refuse_rows(placed, placed[start_column].isna(), [start_column], "has no start time")
    _refuse_rows(
        placed,
        ~placed[day_column].isin(config.WEEKDAYS),
        [day_column],
        "holds a day that is not a weekday",
    )

    placed = convert_clock_times_to_minutes(placed, [start_column] + end_columns)
    on_calendar_clock = systems.isin(config.CALENDAR_CLOCK_SYSTEMS)
    on_broadcast_clock = ~on_calendar_clock
    time_columns = [start_column] + end_columns

    too_late_for_calendar = pd.Series(False, index=placed.index)
    for column_name in time_columns:
        too_late_for_calendar |= on_calendar_clock & (placed[column_name] >= config.MINUTES_PER_DAY)
    _refuse_rows(
        placed,
        too_late_for_calendar.fillna(False).astype(bool),
        time_columns,
        "holds a calendar-clock time of 24:00 or later",
    )

    starts = placed[start_column]
    outside_broadcast_day = on_broadcast_clock & (
        (starts < config.BROADCAST_DAY_START_MINUTE) | (starts >= config.BROADCAST_DAY_END_MINUTE)
    )
    _refuse_rows(
        placed,
        outside_broadcast_day.astype(bool),
        [start_column],
        "holds a broadcast-clock start outside the broadcast day",
    )

    small_hours_start = on_calendar_clock & (starts < config.BROADCAST_DAY_START_MINUTE)
    small_hours_start = small_hours_start.astype(bool)
    placed.loc[small_hours_start, day_column] = placed.loc[small_hours_start, day_column].map(
        config.PREVIOUS_WEEKDAY
    )
    placed.loc[small_hours_start, start_column] += config.MINUTES_PER_DAY

    for end_column in end_columns:
        ends = placed[end_column]
        early_end = (on_calendar_clock & (ends <= config.BROADCAST_DAY_START_MINUTE)).fillna(False)
        placed.loc[early_end.astype(bool), end_column] += config.MINUTES_PER_DAY
        not_after_start = (placed[end_column] <= placed[start_column]).fillna(False)
        _refuse_rows(
            placed,
            not_after_start.astype(bool),
            [start_column, end_column],
            "holds an end that is not after its start",
        )
    return placed


def format_broadcast_clock(dataframe: pd.DataFrame, time_columns: list[str]) -> pd.DataFrame:
    """Write minutes on the broadcast clock back as HH:MM, so 1510 is 25:10 and 1800 is 30:00."""
    formatted = dataframe.copy()
    digits = config.CLOCK_TIME_DIGITS

    def as_clock(minutes):
        hours, remainder = divmod(int(minutes), config.MINUTES_PER_HOUR)
        return f"{str(hours).zfill(digits)}:{str(remainder).zfill(digits)}"

    for column_name in time_columns:
        formatted[column_name] = formatted[column_name].map(as_clock, na_action="ignore")
    return formatted


def plain_text(text: str) -> str:
    """One note as plain text: no tags, no escapes, single spaces, nothing leading or trailing."""
    without_markup = re.sub(config.MARKUP_PATTERN, " ", text)
    unescaped = html.unescape(without_markup).replace(config.NO_BREAK_SPACE, " ")
    return re.sub(config.WHITESPACE_RUN_PATTERN, " ", unescaped).strip()


def convert_markup_to_plain_text(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Turn every note's text into plain text; a tag between two words leaves them apart."""
    converted = dataframe.copy()
    column_name = config.NOTE_TEXT_COLUMN
    converted[column_name] = converted[column_name].map(
        lambda text: plain_text(str(text)), na_action="ignore"
    )
    return converted


def treat_out_of_range_as_missing(
    dataframe: pd.DataFrame, range_limits: dict[str, tuple]
) -> pd.DataFrame:
    """Blank every reading outside its column's limits, both limits included; keep every row."""
    limited = dataframe.copy()
    for column_name, (lowest, highest) in range_limits.items():
        if column_name not in limited.columns:
            continue
        numbers = pd.to_numeric(limited[column_name])
        impossible = ((numbers < lowest) | (numbers > highest)).fillna(False).astype(bool)
        limited[column_name] = limited[column_name].mask(impossible, np.nan)
    return limited
