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

        if not allowed:
            break

        logits = model.get_logits_from_input_ids(input_ids)
        valid_ids = [i for i in vocab.id_to_text if 0 <= i < len(logits)]
        best_id = max(valid_ids, key=lambda i: logits[i])
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
        prompt: str, vocab: Vocab, model: Small_LLM_Model, max_tokens: int = 80
        ) -> str:
    """Let the model write one string value, stopping at a closing quote."""
    input_ids: list[int] = model.encode(prompt)[0].tolist()
    raw_chars: list[str] = []
    escaped = False

    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(input_ids)
        ranked = sorted(
            vocab.id_to_text, key=lambda i: logits[i], reverse=True
        )
        chosen_id = None
        piece = ""
        for token_id in ranked:
            candidate = vocab.id_to_text[token_id]
            if not candidate or "\n" in candidate:
                continue
            chosen_id = token_id
            piece = candidate
            break

        if chosen_id is None:
            break

        for ch in piece:
            if escaped:
                raw_chars.append(ch)
                escaped = False
            elif ch == "\\":
                raw_chars.append(ch)
                escaped = True
            elif ch == '"':
                return unescape_json_string("".join(raw_chars))
            else:
                raw_chars.append(ch)

        input_ids.append(chosen_id)

    return unescape_json_string("".join(raw_chars))
