"""Checks for a Roblox trending forecast before it is saved."""


def _number_between_zero_and_one(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return 0 <= value <= 1


def validate_forecast(out):
    """Raise ValueError when a forecast dict is not safe to write.

    out must contain "probabilities" and "top10". Each probability row needs
    an integer universe id, a place from 1 to 10, and a probability from 0 to
    1. A single place may list at most 100 rows, and those probabilities may
    not add up to more than 1. top10 must be exactly ten different ids.
    """
    if not isinstance(out, dict):
        raise ValueError("forecast must be a dict")
    if "probabilities" not in out or "top10" not in out:
        raise ValueError("forecast must include probabilities and top10")

    rows = out["probabilities"]
    if isinstance(rows, (str, bytes)) or not isinstance(rows, (list, tuple)):
        raise ValueError("probabilities must be a list of rows")

    counts = {place: 0 for place in range(1, 11)}
    totals = {place: 0.0 for place in range(1, 11)}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each probability row must be a dict")
        universe_id = row.get("universe_id")
        place = row.get("place")
        probability = row.get("p")
        if type(universe_id) is not int:
            raise ValueError("universe_id must be an int")
        if type(place) is not int or place < 1 or place > 10:
            raise ValueError("place must be an int from 1 to 10")
        if not _number_between_zero_and_one(probability):
            raise ValueError("p must be a number from 0 to 1")
        counts[place] += 1
        totals[place] += probability
        if counts[place] > 100:
            raise ValueError("place %s has more than 100 rows" % place)

    for place, total in totals.items():
        if total > 1:
            raise ValueError("place %s probabilities sum to more than 1" % place)

    top10 = out["top10"]
    if isinstance(top10, (str, bytes)) or not isinstance(top10, (list, tuple)):
        raise ValueError("top10 must be a list of ids")
    try:
        distinct = set(top10)
    except TypeError:
        raise ValueError("top10 ids must be hashable")
    if len(top10) != 10 or len(distinct) != 10:
        raise ValueError("top10 must contain exactly 10 distinct ids")
