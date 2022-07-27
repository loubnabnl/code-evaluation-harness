import re
from tqdm import tqdm

from transformers import pipeline, StoppingCriteria, StoppingCriteriaList
from accelerate import Accelerator

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


def remove_last_block(string):
    """Remove the last block of the code containing EOF_STRINGS"""
    string_list = re.split("(%s)" % "|".join(EOF_STRINGS), string)
    # last string should be ""
    return "".join(string_list[:-2])


class TokenizedDataset(IterableDataset):
    """Tokenize and preprocess the dataset
    Multiple copies of the same prompt are sent sequentially.
    See compute_code for more details.
    """

    def __init__(self, tokenizer, dataset, n_tasks=None, n_copies=1):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.n_tasks = len(dataset) if n_tasks is None else n_tasks
        self.n_copies = n_copies

    def __iter__(self):
        prompts = []
        for task in range(self.n_tasks):
            # without strip, the model generate commented codes ...
            prompts.append(self.tokenizer.eos_token + self.dataset[task]["prompt"].strip())
        outputs = self.tokenizer(prompts, padding=True, return_tensors="pt")
        for task in range(self.n_tasks):
            for _ in range(self.n_copies):
                yield {
                    "ids": outputs.input_ids[task],
                    "task_id": task,
                    "input_len": outputs.attention_mask[task].sum(),
                }


def complete_code(accelerator, model, tokenizer, dataloader, n_tasks, batch_size=20, **gen_kwargs):
    """Generate multiple codes for each task in the dataset. This function leverage accelerator to distribute
    the processing to multiple GPUs.
    dataloader, a wrapper around a TokenizeDataset objectm is supposed to send all the prompts from
    the evalution dataset to the modelm as the following:
    [p_0_0, p_0_1, ..., p_0_nc-1, p_1_0, ..., p_nt-1_nc-1]
    where nc is the number of copies of the prompt, and nt is the number of tasks.
    nc is such that num_sample = nc * batch_size

    Parameters
    ----------
    accelerator: Accelerator

    model: transformers.PreTrainedModel
        Code generation model. AutoTokenizer.from_pretrained(model_ckpt), ex model_ckpt = "lvwerra/codeparrot"

    tokenizer: transformers.AutoTokenizer
        The tokenizer used to train model

    dataloader: DataLoader
        The dataloader is a wrapper around a TokenizeDataset object. It is designed to be used with multiple GPUs.

    n_tasks: int
        The number of tasks in the dataset. It is used to determine the length of the output.
        Should be aligned with the number of tasks in the TokenizeDataset.

    batch_size: int
        num_return_sequences per copy of the prompt such that num_sample = batch_size * n_copies

    gen_kwargs: dict
        Keyword arguments for the generation function of the model.

    Returns
    -------
    code_gens: list of list of str, of length n_tasks
        List of generated codes for each task.
        Each element is a list of generated codes for each task, with length num_samples
    """
    gen_token_dict = defaultdict(list)  # dict of list of generated tokens
    for step, batch in tqdm(enumerate(dataloader)):
        with torch.no_grad():
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
            code_gens[task].append(remove_last_block(gen_code))
    return code_gens

def make_parallel_generations(model, tokenizer, dataset, args, num_tasks=None):
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=args.device)

    # Generation settings
    gen_kwargs = {
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "stopping_criteria": StoppingCriteriaList([EndOfFunctionCriteria(0, EOF_STRINGS, tokenizer)]),
    }

    n_tasks = num_tasks if num_tasks is not None else len(dataset)
    n_copies = args.n_samples // args.batch_size

    ds_tokenized = TokenizedDataset(tokenizer, dataset, n_copies=n_copies, n_tasks=n_tasks)
    # do not confuse args.batch_size, which is actually the num_return_sequences
    ds_loader = DataLoader(ds_tokenized, batch_size=1)

    model, human_eval_loader = accelerator.prepare(model, ds_loader)

    generations = complete_code(
        accelerator,
        model,
        tokenizer,
        ds_loader,
        n_tasks=n_tasks,
        batch_size=args.batch_size,
        **gen_kwargs,
    )
    if accelerator.is_main_process:
        references = []

        for task in tqdm(range(n_tasks)):
            test_func = human_eval["test"][task]["test"]
            entry_point = f"check({human_eval['test'][task]['entry_point']})"
            references.append("\n" + test_func + "\n" + entry_point)
    return generations, references