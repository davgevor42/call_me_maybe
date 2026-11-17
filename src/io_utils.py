"""Loading input files and writing the output file."""

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .models import FunctionCall, FunctionDefinition, PromptItem


def _read_json_list(path: Path) -> list[Any]:
    """Read a file that must contain a JSON array.

    Raises:
        ValueError: if the file is missing, unreadable, invalid JSON,
            or not an array.
    """
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"File not found: {path}") from None
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from None
    except OSError as e:
        raise ValueError(f"Cannot read {path}: {e}") from None
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON array")
    return raw


def load_functions(path: Path) -> list[FunctionDefinition]:
    """Load and validate the function definitions file."""
    try:
        return [FunctionDefinition(**item) for item in _read_json_list(path)]
    except (ValidationError, TypeError) as e:
        raise ValueError(f"Invalid content in {path}: {e}") from None


def load_prompts(path: Path) -> list[PromptItem]:
    """Load and validate the prompts file."""
    try:
        return [PromptItem(**item) for item in _read_json_list(path)]
    except (ValidationError, TypeError) as e:
        raise ValueError(f"Invalid content in {path}: {e}") from None


def save_results(path: Path, results: list[FunctionCall]) -> None:
    """Write the results as a JSON array, creating folders if needed."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("[\n")
            for index, r in enumerate(results):
                f.write("    {\n")
                f.write(f'        "prompt": {json.dumps(r.prompt)},\n')
                f.write(f'        "name": {json.dumps(r.name)},\n')
                f.write(f'        "parameters": {json.dumps(r.parameters)}\n')
                f.write("    }")
                f.write(",\n" if index < len(results) - 1 else "\n")
            f.write("]\n")
    except OSError as e:
        raise ValueError(f"Cannot write {path}: {e}") from None
