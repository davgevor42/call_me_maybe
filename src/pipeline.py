"""Turning one user prompt into one function call."""

from typing import Any
import json
from llm_sdk import Small_LLM_Model
from .constrained import (
    choose_function_name,
    generate_number,
    generate_string
)
from .models import FunctionCall, FunctionDefinition
from .prompts import build_name_prompt, build_param_prompt
from .vocab import Vocab


def process_prompt(
        user_prompt: str,
        functions: list[FunctionDefinition],
        vocab: Vocab,
        model: Small_LLM_Model,
        ) -> FunctionCall:
    """Choose a function and fill its parameters for one request."""
    names = [fn.name for fn in functions]
    name_prompt = build_name_prompt(user_prompt, functions)
    chosen = choose_function_name(name_prompt, names, vocab, model)
    fn = next((f for f in functions if f.name == chosen), None)
    if fn is None:
        raise ValueError(f"Model returned unknown function: {chosen!r}")
    values: dict[str, Any] = {}
    filled = "{"
    for index, (param, spec) in enumerate(fn.parameters.items()):
        if index > 0:
            filled += ", "
        filled += f'"{param}":'
        if spec.type == "string":
            prompt = build_param_prompt(user_prompt, fn, filled + '"')
            text = generate_string(prompt, vocab, model)
            values[param] = text
            filled += json.dumps(text)
        elif spec.type == "integer":
            prompt = build_param_prompt(user_prompt, fn, filled)
            number = generate_number(
                prompt, vocab, model, allow_decimal=False
            )
            values[param] = int(number)
            filled += str(int(number))
        elif spec.type == "number":
            prompt = build_param_prompt(user_prompt, fn, filled)
            number = generate_number(
                prompt, vocab, model, allow_decimal=True
            )
            values[param] = number
            filled += str(number)
        elif spec.type == "boolean":
            prompt = build_param_prompt(user_prompt, fn, filled)
            word = choose_function_name(
                prompt, ["true", "false"], vocab, model
            )
            values[param] = word == "true"
            filled += word
        else:
            raise ValueError(
                f"Unsupported parameter type {spec.type!r} in {fn.name}"
            )
    return FunctionCall(prompt=user_prompt, name=fn.name, parameters=values)
