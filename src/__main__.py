"""Entry point: python -m src."""

import argparse
import sys
from pathlib import Path

from llm_sdk import Small_LLM_Model

from .io_utils import load_functions, load_prompts, save_results
from .models import FunctionCall
from .pipeline import process_prompt
from .vocab import Vocab


def main() -> int:
    """Parse arguments, load inputs and print what was loaded."""
    parser = argparse.ArgumentParser(description="Function calling tool")
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json"),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calls.json"),
    )
    args = parser.parse_args()

    try:
        functions = load_functions(args.functions_definition)
        prompts = load_prompts(args.input)
        model = Small_LLM_Model()
        vocab = Vocab.from_file(Path(model.get_path_to_vocab_file()))
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    results: list[FunctionCall] = []
    for item in prompts:
        try:
            results.append(
                process_prompt(item.prompt, functions, vocab, model)
            )
        except Exception as e:
            print(f"Skipped {item.prompt!r}: {e}", file=sys.stderr)

    try:
        save_results(args.output, results)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Wrote {len(results)}/{len(prompts)} results to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
