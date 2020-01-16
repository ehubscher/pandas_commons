import datetime
import regex
from typing import List

"""String utility module for common string operations."""


def number_position_suffix(number: int):
    """Returns the appropriate suffix associated with the given integer."""

    if number % 10 is 1:
        return 'st'

    if number % 10 is 2:
        return 'nd'

    if number % 10 is 3:
        return 'rd'

    return 'th'


def snake_case(string: str) -> str:
    """
    Formats a string to it's snake_cased form.

    :param string: The un-formatted string.
    :return: The snake-cased version of the input string.
    """

    sub = regex.sub(pattern=r'(.)([A-Z][a-z]+)', repl=r'\1_\2', string=string)

    return regex.sub(pattern=r'([a-z0-9])([A-Z])', repl=r'\1_\2', string=sub).lower()
