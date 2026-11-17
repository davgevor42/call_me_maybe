"""Vocabulary: mapping from token ids to their real text."""

import json
from pathlib import Path

from pydantic import BaseModel


def _build_char_to_byte() -> dict[str, int]:
    safe = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    unsafe = [b for b in range(256) if b not in safe]

    byte_to_char: dict[int, str] = {}
    for b in safe:
        byte_to_char[b] = chr(b)
    for n, b in enumerate(unsafe):
        byte_to_char[b] = chr(256 + n)

    char_to_byte: dict[str, int] = {}
    for b, c in byte_to_char.items():
        char_to_byte[c] = b

    return char_to_byte


_CHAR_TO_BYTE: dict[str, int] = _build_char_to_byte()


def token_to_text(token: str) -> str:
    """Convert a raw vocab string (with 'Ġ', 'Ċ'...) into real text."""
    try:
        raw = bytes(_CHAR_TO_BYTE[c] for c in token)
    except KeyError:
        return token
    return raw.decode("utf-8", errors="replace")


class Vocab(BaseModel):
    """Maps every token id to its decoded text."""

    id_to_text: dict[int, str]

    @classmethod
    def from_file(cls, path: Path) -> "Vocab":
        """Load vocab.json and decode every token.

        Raises:
            ValueError: if the file is missing or invalid.
        """
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            return cls(
                id_to_text={
                    int(i): token_to_text(tok) for tok, i in raw.items()
                }
            )
        except (OSError, json.JSONDecodeError, AttributeError) as e:
            raise ValueError(f"Cannot load vocab {path}: {e}") from None
