import re
import json
from tqdm import tqdm
from collections import defaultdict

import torch
from torch.utils.data import IterableDataset

# to do add mbpp config

EOF_STRINGS = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"]


def generate_prompt(sample):
    """Generate prompts for APP, they include a question along with some starter code and function name if they exist
    We also specify the type of the prompt, i.e. whether it is call-based or standard input"""

    starter_code = None if len(sample["starter_code"]) == 0 else sample["starter_code"] 
    try:
        input_outpout = json.loads(sample["input_output"])
        fn_name = None if not input_outpout.get("fn_name") else input_outpout["fn_name"]
    except ValueError:
        fn_name = None
    _input = "\nQUESTION:\n"
    _input += sample["question"]
    if starter_code:
        _input += starter_code
    if fn_name:
        _input += "\nUse Standard Input format"
    else:
        _input += "\nUse Call-Based format"
    
    _input += "\nANSWER:\n"
    return _input


def remove_last_block(string):
    """Remove the last block of the code containing EOF_STRINGS for HumanEval"""
    
    string_list = re.split("(%s)" % "|".join(EOF_STRINGS), string)
    # last string should be ""
    return "".join(string_list[:-2])


class TokenizedDataset(IterableDataset):
    """Tokenize and preprocess the dataset
    Multiple copies of the same prompt are sent sequentially.
    See compute_code for more details.
    """

    def __init__(self, tokenizer, dataset, mode="humaneval", n_tasks=None, n_copies=1):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.mode = mode
        self.n_tasks = len(dataset) if n_tasks is None else n_tasks
        self.n_copies = n_copies

    def __iter__(self):
        prompts = []
        for task in range(self.n_tasks):
            if self.mode == "apps":
                prompt = generate_prompt(self.dataset[task]).strip()
            else:
                prompt = self.dataset[task]["prompt"].strip()
            prompt = self.tokenizer.eos_token + prompt
            #if len(self.tokenizer.tokenize(prompt)) > self.model.config.n_ctx:
            #    prompt= prompt[:self.model.config.n_ctx]
            prompts.append(prompt)
        outputs = self.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
        for task in range(self.n_tasks):
            for _ in range(self.n_copies):
                yield {
                    "ids": outputs.input_ids[task],
                    "task_id": task,
                    "input_len": outputs.attention_mask[task].sum(),
                }


def complete_code(accelerator, model, tokenizer, dataloader, n_tasks, batch_size=20, mode="humaneval", **gen_kwargs):
    """Generate multiple codes for each task in the dataset using multiple GPUs with accelerate.
    dataloader sends all the prompts from the evalution dataset to the model as the following:
    [p_0_0, p_0_1, ..., p_0_nc-1, p_1_0, ..., p_nt-1_nc-1] where nc is the number of copies of the prompt,
    and nt is the number of tasks. nc is such that num_samples(for each task)= nc * batch_size
    """

    gen_token_dict = defaultdict(list)  # dict of list of generated tokens
    for step, batch in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            if mode != "apps":
                gen_kwargs["stopping_criteria"][0].start_length = batch["ids"].shape[-1]
            generated_tokens = accelerator.unwrap_model(model).generate(
                input_ids=batch["ids"][:, : batch["input_len"]], num_return_sequences=batch_size, **gen_kwargs
            )
            # each task is generated batch_size times
            generated_tasks = batch["task_id"].repeat(batch_size)
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens, generated_tasks = accelerator.gather((generated_tokens, generated_tasks))
            generated_tokens = generated_tokens.cpu().numpy()
            generated_tasks = generated_tasks.cpu().numpy()

            for task, generated_tokens in zip(generated_tasks, generated_tokens):
                gen_token_dict[task].append(generated_tokens)

    code_gens = [[] for _ in range(n_tasks)]
    for task, generated_tokens in gen_token_dict.items():
        for s in generated_tokens:
            gen_code = tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            if mode == "humaneval":
                code_gens[task].append(remove_last_block(gen_code))
            elif mode == "apps":
                code_gens[task].append(gen_code.replace(tokenizer.eos_token, ""))
    return code_gens