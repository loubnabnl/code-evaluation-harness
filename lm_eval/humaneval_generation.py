from tqdm import tqdm

from torch.utils.data.dataloader import DataLoader
from transformers import StoppingCriteria, StoppingCriteriaList
from accelerate import Accelerator
from accelerate.utils import set_seed

from lm_eval.utils import complete_code, TokenizedDataset


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


def make_parallel_generations(model, tokenizer, dataset, args, num_tasks=None):
    accelerator = Accelerator()
    set_seed(args.seed, device_specific=True)

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

    ds_tokenized = TokenizedDataset(tokenizer, dataset, mode="humaneval", n_copies=n_copies, n_tasks=n_tasks)
    # do not confuse args.batch_size, which is actually the num_return_sequences
    ds_loader = DataLoader(ds_tokenized, batch_size=1)

    model, ds_loader = accelerator.prepare(model, ds_loader)

    generations = complete_code(
        accelerator,
        model,
        tokenizer,
        ds_loader,
        n_tasks=n_tasks,
        batch_size=args.batch_size,
        mode="humaneval",
        **gen_kwargs,
    )
    if accelerator.is_main_process:
        references = []

        for task in tqdm(range(n_tasks)):
            test_func = dataset[task]["test"]
            entry_point = f"check({dataset[task]['entry_point']})"
            references.append("\n" + test_func + "\n" + entry_point)
    return generations, references