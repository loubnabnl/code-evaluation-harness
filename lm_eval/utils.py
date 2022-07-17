import json
import multiprocessing
import os
import re

from datasets import load_dataset
from tqdm import tqdm

import torch
import transformers
from arguments import HumanEvalArguments
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    StoppingCriteria,
    StoppingCriteriaList,
    set_seed,
)


EOF_STRINGS = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"]


class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""

    def __init__(self, start_length, eof_strings, tokenizer):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(input_ids[:, self.start_length :])
        done = []
        for decoded_generation in decoded_generations:
            done.append(any([stop_string in decoded_generation for stop_string in self.eof_strings]))
        return all(done)


def first_block(string):
    """Split off first block of code by scanning for class, def etc. on newlines."""
    return re.split("|".join(EOF_STRINGS), string)[0].rstrip()


def complete_code(model, tokenizer, prompt, num_completions=1, **gen_kwargs):
    """Complete prompt with text generation pipeline and return num_completions."""
    prompt = tokenizer.eos_token + prompt

    tokenized_prompt = tokenizer(prompt, return_tensors="pt")
    tokenized_prompt = {k: v.to("cuda") for k, v in tokenized_prompt.items()}

    outputs = model.generate(**tokenized_prompt, num_return_sequences=num_completions, **gen_kwargs)
    code_gens = [tokenizer.decode(outputs[i]) for i in range(outputs.shape[0])]

    return [first_block(code_gen[len(prompt) :]) for code_gen in code_gens]


