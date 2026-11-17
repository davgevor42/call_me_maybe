"""Constrained decoding helpers."""

from .vocab import Vocab
from llm_sdk import Small_LLM_Model
import json

NUMBER_CHARS = "0123456789-."


def allowed_name_tokens(
        generated: str, names: list[str], vocab: Vocab
        ) -> list[int]:
    """Return ids of tokens that keep `generated` a prefix of a name."""
    allowed: list[int] = []
    for token_id, text in vocab.id_to_text.items():
        if text == "":
            continue
        candidate = generated + text
        if any(name.startswith(candidate) for name in names):
            allowed.append(token_id)
    return allowed


def choose_function_name(
        prompt: str, names: list[str], vocab: Vocab, model: Small_LLM_Model
        ) -> str:
    """Let the model pick one function name, token by token."""
    input_ids: list[int] = model.encode(prompt)[0].tolist()
    generated = ""

    while generated not in names:
        allowed = allowed_name_tokens(generated, names, vocab)
        if not allowed:
            break
        logits = model.get_logits_from_input_ids(input_ids)
        best_id = max(allowed, key=lambda i: logits[i])
        input_ids.append(best_id)
        generated += vocab.id_to_text[best_id]
    return generated


def number_token_ids(vocab: Vocab) -> list[int]:
    """Return (number ids, stop ids) for number generation.

    A number token is made only of digits, '-' and '.'.
    A stop token is a single character that ends a number.
    """
    number_ids: list[int] = []
    for token_id, text in vocab.id_to_text.items():
        if text == "":
            continue
        if len(text) == 1 and text in NUMBER_CHARS:
            number_ids.append(token_id)
    return number_ids


def generate_number(
        prompt: str, vocab: Vocab, model: Small_LLM_Model,
        max_tokens: int = 20, allow_decimal: bool = True) -> float:
    """Let the model write one number, then return it as a float."""
    number_ids = number_token_ids(vocab)
    input_ids: list[int] = model.encode(prompt)[0].tolist()
    text = ""
    for _ in range(max_tokens):
        has_digit = any(c.isdigit() for c in text)
        allowed: list[int] = []
        for i in number_ids:
            ch = vocab.id_to_text[i]
            if ch == "-" and text != "":
                continue
            if ch == "." and (
                not allow_decimal or not has_digit or "." in text
            ):
                continue
            allowed.append(i)

        logits = model.get_logits_from_input_ids(input_ids)
        best_id = max(range(len(logits)), key=lambda i: logits[i])
        if best_id not in allowed:
            if has_digit:
                break
            best_id = max(allowed, key=lambda i: logits[i])
        input_ids.append(best_id)
        text += vocab.id_to_text[best_id]
    try:
        return float(text)
    except ValueError:
        return 0.0


def unescape_json_string(text: str) -> str:
    """Decode JSON escapes (like \\\\d -> \\d) written by the model."""
    try:
        decoded = json.loads(f'"{text}"')
    except json.JSONDecodeError:
        return text
    return str(decoded)


def generate_string(
        prompt: str, vocab: Vocab, model: Small_LLM_Model, max_tokens: int = 40
        ) -> str:
    """Let the model write one string value, stopping at a closing quote."""
    input_ids: list[int] = model.encode(prompt)[0].tolist()
    text = ""

    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(input_ids)
        ranked = sorted(
            vocab.id_to_text, key=lambda i: logits[i], reverse=True
        )
        for token_id in ranked:
            piece = vocab.id_to_text[token_id]
            if "\n" in piece or piece == "":
                continue
            break
        if '"' in piece:
            text += piece.split('"')[0]
            return unescape_json_string(text)
        input_ids.append(token_id)
        text += piece
    return unescape_json_string(text)
