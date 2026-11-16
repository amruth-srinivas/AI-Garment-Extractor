import re

from app.errors import AppError

_HEX = re.compile(r"^[a-f0-9]{32}$")


def require_id(value: str) -> str:
    if not value or not _HEX.fullmatch(value):
        raise AppError("That record could not be found.", 404)
    return value
