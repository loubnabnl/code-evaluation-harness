from torch.utils.data.dataloader import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed

from lm_eval.utils import complete_code, TokenizedDataset

def make_parallel_generations(model, tokenizer, dataset, args, num_tasks=None):
    accelerator = Accelerator()
    set_seed(args.seed, device_specific=True)

    # Generation settings
    gen_kwargs = {
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k
    }

    n_tasks = num_tasks if num_tasks is not None else len(dataset)
    n_copies = args.n_samples // args.batch_size

    ds_tokenized = TokenizedDataset(tokenizer, dataset, mode="apps", n_copies=n_copies, n_tasks=n_tasks)
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
        mode = "apps",
        **gen_kwargs,
    )
    return generations