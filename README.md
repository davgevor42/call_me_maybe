*This project has been created as part of the 42 curriculum by davgevor.*

## Description

**call me maybe** translates natural-language requests into structured function calls, using constrained decoding with a small model (Qwen/Qwen3-0.6B).

Given a prompt such as `"What is the sum of 2 and 3?"`, the program does not answer `5`. It outputs the function to call and its typed arguments:

```json
{
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
}
```

Small models are unreliable when simply asked to write JSON. This project does not rely on prompting alone: generation is controlled token by token, so the output is always valid and matches the schema in `functions_definition.json`.

Overview of the pipeline for each prompt:

1. The LLM chooses the function name, restricted to the names in the definitions file.
2. The program writes the JSON skeleton itself (keys, quotes, braces).
3. The LLM fills each parameter value, restricted by the parameter type (`number`, `integer`, `string`, `boolean`).
4. The values are stored in a pydantic model and serialized with `json`, so the output file is always valid JSON.

## Instructions

**Requirements:** `git`, [`uv`](https://docs.astral.sh/uv/), and internet access on the first run (torch and the model, about 1.2 GB, are downloaded and cached).

**Installation:**

```bash
git clone <repository-url>
cd call_me_maybe
uv sync          # or: make install
```

**Execution:**

```bash
uv run python -m src [--functions_definition <file>] [--input <file>] [--output <file>]
```

Defaults, relative to the project root: `data/input/functions_definition.json`, `data/input/function_calling_tests.json` and `data/output/function_calls.json`. The output directory is created automatically.

**Makefile rules:**

- `make install`: `uv sync`
- `make run`: run the program with default paths
- `make debug`: run the program under `pdb`
- `make clean`: remove `__pycache__`, `.mypy_cache`, `.pytest_cache`
- `make lint`: `flake8 .` and `mypy .` with the mandatory flags
- `make lint-strict`: `flake8 .` and `mypy . --strict`

**Project layout:**

```
.
├── Makefile
├── pyproject.toml / uv.lock
├── .flake8 / mypy.ini
├── llm_sdk/               provided SDK (not modified)
├── data/input/            input files
└── src/
    ├── __main__.py        CLI entry point and main loop
    ├── models.py          pydantic models
    ├── io_utils.py        reading inputs, writing the output file
    ├── vocab.py           token id -> real text table
    ├── prompts.py         prompt builders
    ├── constrained.py     constrained decoding
    └── pipeline.py        one prompt -> one function call
```

## Resources

- [uv documentation](https://docs.astral.sh/uv/)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [Python argparse](https://docs.python.org/3/library/argparse.html) and [json](https://docs.python.org/3/library/json.html)
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- GPT-2 byte-level BPE (`bytes_to_unicode`): *Language Models are Unsupervised Multitask Learners*, OpenAI, 2019
- Articles and talks on constrained / structured decoding for LLMs

**Use of AI:** an AI assistant was used as a tutor throughout the project:

- explaining concepts (tokens, logits, constrained decoding, byte-level BPE, `uv`, argparse, pydantic),
- helping me understand the code line by line (what each function does, why it is written that way, and how the pieces fit together),
- diagnosing problems from my test output (the `" -"` token, regex escaping, flake8 and mypy configuration),
- helping to draft this README.

## Algorithm explanation

**Generation loop.** The SDK returns one list of logits (one score per token id) for the next token. At every step the program calls `get_logits_from_input_ids` with the prompt ids plus the tokens chosen so far, decides which token ids are valid, picks the valid token with the highest logit (invalid tokens can never win, which is equivalent to setting their logits to `-inf`), appends it, and repeats until the value is complete.

**Vocabulary map (`vocab.py`).** `vocab.json` maps token strings to ids, but the strings use GPT-2 byte-level encoding (for example `Ġ` stands for a space). The program rebuilds the standard byte-to-character table, reverses it, and decodes each token string into real text (characters to bytes to UTF-8, with `errors="replace"` for partial UTF-8 tokens). The result is a table `id -> text`, built once per run. Ids without text (special tokens) are never allowed.

**Function name selection.** The prompt lists every function with its description, then the request, then the unfinished line `Function to call: `. Let `generated` be the text produced so far. A token with text `t` is allowed if some function name starts with `generated + t`. The loop stops when `generated` equals a full name. The model's own logits choose among the allowed tokens, so the model, not a heuristic, selects the function.

**Parameter extraction.** The program writes the JSON skeleton and the key names. The model only produces values, with a prompt that ends exactly where the value goes (for example `{"a": `). Rules per type:

- `number`: only single-character tokens `0-9`, `-`, `.`; `-` only first, `.` once and only after a digit. Stored as `float`.
- `integer`: same, but `.` is never allowed. Stored as `int`.
- `string`: tokens with newlines or empty text are blocked; a token containing `"` ends the string (only the part before the quote is kept). Stored as `str`.
- `boolean`: prefix-constrained to `true` / `false`, with the same mechanism as function names. Stored as `bool`.

For numbers, once at least one digit exists, the model's unrestricted favorite token decides when the number ends: if it is not a valid number piece, the number is finished. Before the first digit, the choice is restricted to the allowed number tokens. Unsupported types (for example `array`, `object`) raise a clear error for that prompt.

**Output.** Values are kept in Python objects (pydantic `FunctionCall`) and serialized with `json.dumps`, so quotes, backslashes and special characters are always escaped correctly.

## Design decisions

- **Python writes the structure, the model writes values.** The model never generates braces, quotes or keys, so structural errors are impossible.
- **Nothing is hardcoded.** Function names, parameter names and types always come from the definitions file, and prompts are built from the data.
- **Prefix constraint reused** for function names and booleans.
- **Vocabulary decoded once** per run, not per token.
- **Per-prompt error isolation.** Each prompt runs in its own `try`; a failing prompt is reported on stderr and skipped, and the run continues. Input file errors give one clean `Error: ...` line.
- **Safety limits.** Every generation loop has a maximum token count.
- **Prompt example.** The parameter prompt contains a short generic example (a lowercase name not used in the tests) so the model copies values faithfully.
- **Config files.** `.flake8` and `mypy.ini` exclude `.venv` and the provided `llm_sdk` package; the `lint` rule still uses the flags required by the subject.
- **Only public SDK methods** are used (`encode`, `get_logits_from_input_ids`, `get_path_to_vocab_file`).

## Performance analysis

- **Reliability:** the output file is always valid JSON, because only Python serializes it and the model never generates the structure. Function names are always one of the defined names, and number, integer and boolean values always have the right type.
- **Accuracy:** on the 11 example prompts, all function names and arguments were correct after the fixes described below. <fill in: result on your final test set>
- **Speed:** the SDK has no KV cache, so every step re-processes the whole sequence. Measured on a MacBook (`mps`): about 0.4 to 0.8 s per function-name selection and 0.15 to 0.45 s per string value in early tests. <fill in: total time for the full run, from `time uv run python -m src`>
- **Limits:** constrained decoding guarantees valid output, not correct content. The 0.6B model can still pick a wrong function or copy a value imperfectly. Skipped prompts do not appear in the output file.

## Challenges faced

- **Byte-level vocabulary.** Token strings such as `ĠWhat` are not real text. Solved by rebuilding the GPT-2 byte table and decoding every token.
- **Number stop signal.** A model forced to write only digits can never end a number. First solved with stop tokens (`,`, `}`, space, newline), then with the rule that any non-number favorite token ends the number once a digit exists. A substring-versus-set bug (`" \n"` matching inside a string constant) was fixed by using a set.
- **Junk number tokens.** Tokens such as `----` or `....` passed a naive filter; only single-character number tokens are accepted.
- **Leading-space tokens.** The model prefers `" -"` (space plus minus), which is not in the allowed set, so negative numbers were first read wrongly. Removing the trailing space of the value prompt made the bare `-` win, verified on several negative and decimal prompts.
- **Regex strings.** The model wrote `\\d+` (JSON-escaped), producing a double backslash in the output. Solved by decoding the model's text once with `json.loads`.
- **Backslash blocking.** Blocking backslashes made regex values impossible; the block was removed because `json.dumps` escapes them safely.
- **Capitalization.** The model wrote `Shrek` for `shrek`. An instruction line did not help; a short worked example in the prompt did.
- **Tooling.** flake8 crashed on `.venv` and mypy could not see `llm_sdk`; both were fixed with configuration files.

## Testing strategy

- Step-by-step checks with scratch scripts for each component: vocabulary decoding (token counts, known ids such as `17 -> '2'`), allowed-token functions, number, string and boolean generation.
- Debug prints of the top-5 logits to understand wrong choices (for example the `" -"` token).
- End-to-end runs on the provided examples, checked by hand for function name and argument values.
- Extra function files covering `integer` and `boolean` parameters and mixed functions.
- Negative and decimal numbers, large numbers, strings with quotes and regex patterns.
- Error handling: missing file, invalid JSON, wrong JSON shape, unwritable output path.
- `make lint` (flake8 and mypy) and a clean `git clone` followed by `uv sync` and a run.

## Example usage

```bash
# default paths
uv run python -m src

# custom paths
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

Example output (`data/output/function_calls.json`):

```json
[
    {
        "prompt": "Greet john",
        "name": "fn_greet",
        "parameters": {"name": "john"}
    },
    {
        "prompt": "Reverse the string 'hello'",
        "name": "fn_reverse_string",
        "parameters": {"s": "hello"}
    }
]
```

Error handling example (a clean message, no traceback):

```bash
uv run python -m src --input missing.json
# Error: File not found: missing.json
```
