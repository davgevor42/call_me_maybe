"""Building the text prompts that are sent to the model."""

from .models import FunctionDefinition


def build_name_prompt(
        user_prompt: str, functions: list[FunctionDefinition]
        ) -> str:
    """Build the prompt used to choose the function name."""
    lines = ["Available functions:"]
    for fn in functions:
        params = ", ".join(f"{name}: {spec.type}" for name, spec
                           in fn.parameters.items())
        lines.append(f"- {fn.name}({params}): {fn.description}")
    lines.append("")
    lines.append(f"Request: {user_prompt}")
    lines.append("Function to call: ")
    return "\n".join(lines)


def build_param_prompt(
        user_prompt: str, fn: FunctionDefinition, filled: str
        ) -> str:
    """Build the prompt used to generate one parameter value.

    `filled` is the JSON written so far, ending right where the
    next value must go, for example: {"a":
    """
    params = ", ".join(
        f"{name} ({spec.type})" for name, spec in fn.parameters.items()
    )
    lines = [
        f"Function: {fn.name}",
        f"Description: {fn.description}",
        f"Parameters: {params}",
        (
          "Rule: Extract exact values from the request; do not append _db or"
          " extra words, and keep inner quotes."
        ),
        (
            'Example: Request: Use the analytics database -> {"database":'
            ' "analytics"}'
        ),
        'Example: Request: Print Say "hi"! -> {"text": "Say \\"hi\\"!"}',
        f"Request: {user_prompt}",
        "Arguments as JSON:",
        filled,
    ]
    return "\n".join(lines)
