"""Placing each spot's break in the schedule, testing the five rules and ranking the queue.

Every decision about a spot is made here, by deterministic code, from the rules read out of the
category's own code. No language model ever tests a spot or moves it from one list to the other.

Transmission order runs through the booking week, Monday first, then by the break's minute on
the broadcast clock; two spots in one break go out in the order of their spot references.
"""

from config import config
from data_pipeline.common_cleaning import clock_time_to_minutes


def transmission_order(spot: dict) -> tuple:
    """The key that sorts spots into the order they are transmitted."""
    return (
        config.WEEKDAYS.index(spot["broadcast_day"]),
        spot["break_minute"],
        spot["spot_reference"],
    )


def place_spot(campaign_spot: dict, campaign_schedules: list[dict]) -> dict:
    """Say which programme a spot's break is in (a centre break), or which two it is between."""
    matching = [
        schedule
        for schedule in campaign_schedules
        if schedule["channel"] == campaign_spot["channel"]
        and schedule["broadcast_day"] == campaign_spot["broadcast_day"]
    ]
    if len(matching) != 1:
        raise ValueError(
            f"{campaign_spot['spot_reference']}: {len(matching)} schedules carry "
            f"{campaign_spot['channel']} on {campaign_spot['broadcast_day']}, not one"
        )
    minute = campaign_spot["break_minute"]
    programmes = matching[0]["programmes"]

    around = [
        programme
        for programme in programmes
        if programme["start_minute"] < minute < programme["end_minute"]
    ]
    if around:
        return dict(campaign_spot, break_position=config.BREAK_IN, programmes=[dict(around[0])])

    ending = [programme for programme in programmes if programme["end_minute"] == minute]
    starting = [programme for programme in programmes if programme["start_minute"] == minute]
    if ending and starting:
        return dict(
            campaign_spot,
            break_position=config.BREAK_ADJACENT,
            programmes=[dict(ending[0]), dict(starting[0])],
        )
    raise ValueError(
        f"{campaign_spot['spot_reference']} at {campaign_spot['break_time']} sits at the "
        "edge of the broadcast day or in a gap in the schedule"
    )


def _inside_restricted_hours(break_minute: int, restricted_hours: dict) -> bool:
    """True when a break falls inside hours written on the ordinary clock, start in and end out.

    The broadcast clock is folded back onto the ordinary clock (25:10 is 01:10), and hours whose
    start is later than their end run through midnight, so 22:00 to 02:00 covers 22:00 to 26:00.
    """
    clock_minute = break_minute % config.MINUTES_PER_DAY
    start = clock_time_to_minutes(restricted_hours["start"])
    end = clock_time_to_minutes(restricted_hours["end"])
    if start < end:
        return start <= clock_minute < end
    return clock_minute >= start or clock_minute < end


def _live_sport_excepts(placement: dict, exception: dict | None) -> bool:
    """True when the code's exception lifts the restricted hours for this break.

    Only a centre break inside live sport coverage qualifies, and only when it is more than the
    code's minutes after the coverage starts and more than its minutes before it ends.
    """
    if exception is None or placement["break_position"] != config.BREAK_IN:
        return False
    programme = placement["programmes"][0]
    minute = placement["break_minute"]
    return (
        programme["live_sport"]
        and minute - programme["start_minute"] > exception["minutes_after_start"]
        and programme["end_minute"] - minute > exception["minutes_before_end"]
    )


def _breaks_childrens_rule(placement: dict, children_rule: str) -> bool:
    """True when a programme made for children is one the rule reaches."""
    if children_rule == config.CHILDREN_IN_ONLY and placement["break_position"] != config.BREAK_IN:
        return False
    return any(programme["made_for_children"] for programme in placement["programmes"])


def check_campaign_spots(spot_placements: list[dict], code_rules: dict) -> list:
    """Test every spot of one campaign against all five rules of its category's code."""
    if code_rules["children_rule"] not in config.CHILDREN_RULE_PATTERNS:
        raise ValueError(
            f"The children's rule {code_rules['children_rule']!r} is not one config knows"
        )
    spots_counted, previous_minute = {}, {}
    results = []
    for placement in sorted(spot_placements, key=transmission_order):
        channel_day = (placement["channel"], placement["broadcast_day"])
        minute = placement["break_minute"]
        breaches = set()

        if (
            code_rules["restricted_hours"]
            and _inside_restricted_hours(minute, code_rules["restricted_hours"])
            and not _live_sport_excepts(placement, code_rules["live_sport_exception"])
        ):
            breaches.add(config.BREACH_WINDOW)
        if any(
            programme["forecast_under18_index"] >= code_rules["index_threshold"]
            for programme in placement["programmes"]
        ):
            breaches.add(config.BREACH_INDEX)
        if _breaks_childrens_rule(placement, code_rules["children_rule"]):
            breaches.add(config.BREACH_CHILDREN)

        spots_counted[channel_day] = spots_counted.get(channel_day, 0) + 1
        if spots_counted[channel_day] > code_rules["daily_cap"]:
            breaches.add(config.BREACH_FREQUENCY)
        if (
            channel_day in previous_minute
            and minute - previous_minute[channel_day] < code_rules["separation_minutes"]
        ):
            breaches.add(config.BREACH_SEPARATION)
        previous_minute[channel_day] = minute

        breach_codes = [code for code in config.BREACH_CODES if code in breaches]
        results.append(dict(placement, breach_codes=breach_codes))
    return results


def decisions_open_for(spots_breaching: int, spots_compliant: int) -> list[str]:
    """The decisions the spots leave open, in the order of config.DECISIONS."""
    is_open = {
        config.DECISION_CLEAR: spots_breaching == 0,
        config.DECISION_CLEAR_WITH_RESTRICTIONS: spots_breaching > 0 and spots_compliant > 0,
        config.DECISION_REJECT: True,
    }
    return [decision for decision in config.DECISIONS if is_open[decision]]


def summarise_spot_checks(spot_results: list[dict]) -> dict:
    """Count a campaign's spots and breaches, find its first transmission and the decisions open."""
    if not spot_results:
        raise ValueError("There are no spot results to summarise")
    spots_breaching = sum(1 for result in spot_results if result["breach_codes"])
    spots_compliant = len(spot_results) - spots_breaching
    first = min(spot_results, key=transmission_order)
    return {
        "spots_booked": len(spot_results),
        "spots_compliant": spots_compliant,
        "spots_breaching": spots_breaching,
        "breaches_by_rule": {
            code: sum(1 for result in spot_results if code in result["breach_codes"])
            for code in config.BREACH_CODES
        },
        "first_transmission": {
            "broadcast_day": first["broadcast_day"],
            "break_time": first["break_time"],
            "break_minute": first["break_minute"],
        },
        "decisions_open": decisions_open_for(spots_breaching, spots_compliant),
    }


def compute_clearance_outcome(decision: str, spot_results: list[dict]) -> dict:
    """Work out which spots a decision clears and which it withdraws, keeping their order."""
    if decision not in config.DECISIONS:
        raise ValueError(f"{decision!r} is not one of {config.DECISIONS}")
    summary = summarise_spot_checks(spot_results)
    if decision not in summary["decisions_open"]:
        raise ValueError(
            config.DECISION_NOT_OPEN_MESSAGE.format(
                decision=decision,
                spots_breaching=summary["spots_breaching"],
                spots_booked=summary["spots_booked"],
                spots_compliant=summary["spots_compliant"],
            )
        )
    clears = {
        config.DECISION_CLEAR: lambda result: True,
        config.DECISION_CLEAR_WITH_RESTRICTIONS: lambda result: not result["breach_codes"],
        config.DECISION_REJECT: lambda result: False,
    }[decision]
    spots_cleared = [result["spot_reference"] for result in spot_results if clears(result)]
    spots_withdrawn = [result["spot_reference"] for result in spot_results if not clears(result)]
    return {
        "decision": decision,
        "spots_cleared": spots_cleared,
        "spots_withdrawn": spots_withdrawn,
        "spots_cleared_count": len(spots_cleared),
        "spots_withdrawn_count": len(spots_withdrawn),
    }


def rank_clearance_queue(queue_entries: list[dict]) -> list:
    """Largest exposure first, then the earlier first transmission, then the reference A to Z."""
    references = [entry["campaign_reference"] for entry in queue_entries]
    if len(references) != len(set(references)):
        raise ValueError("A campaign reference appears more than once in the queue")

    def queue_order(entry):
        first = entry["first_transmission"]
        return (
            -entry["exposure"],
            config.WEEKDAYS.index(first["broadcast_day"]),
            first["break_minute"],
            entry["campaign_reference"],
        )

    return sorted(queue_entries, key=queue_order)
