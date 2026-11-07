"""Exact token checks for model-generated statistical narration."""
import re
from datetime import date

NUMBER = re.compile(r"-?\d+(?:\.\d+)?%?")
UNITS = re.compile(r"\b(?:[1-9]|1[0-2]) (?=(?:short )?sentences?\b)", re.I)
ISO_DATE = re.compile(r"\d{4}-\d{2}(?:-\d{2})?")


def check_numbers(text, facts) -> bool:
    allowed = {str(value) for value in facts.values() if value is not None}
    dates = {value for value in allowed if ISO_DATE.fullmatch(value)}
    for stamp in dates:
        text = re.sub(rf"(?<!\d){stamp}(?![-\d])", "", text)
        allowed.add(stamp[:4])
    text = UNITS.sub("", text)
    # Exponents, comma grouping, plus signs and decimals without a leading zero
    # could otherwise be misread as multiple independent allowed tokens.
    if re.search(r"(?<=\d)[eE][+-]?\d|\d,\d|\+\d|(?<!\d)\.\d|−\d", text):
        return False
    return all(token in allowed or (token + "%") in allowed for token in NUMBER.findall(text))


def display_date(value: date):
    return value.isoformat()
