from pathlib import Path

from llm_sdk import Small_LLM_Model

from src.constrained import generate_string
from src.io_utils import load_functions
from src.prompts import build_param_prompt
from src.vocab import Vocab

model = Small_LLM_Model()
vocab = Vocab.from_file(Path(model.get_path_to_vocab_file()))
functions = load_functions(Path("data/input/functions_definition.json"))
fn = next(f for f in functions if f.name == "fn_greet")

prompt = build_param_prompt("Greet shrek", fn, '{"name": "')
print("result:", repr(generate_string(prompt, vocab, model)))
