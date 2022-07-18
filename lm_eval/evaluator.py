import os

from transformers import set_seed

from datasets import load_dataset
from evaluate import load

from lm_eval import apps_utils
from lm_eval import humaneval_mbpp_utils

_WARNING = """
################################################################################
                                  !!!WARNING!!!
################################################################################
The "code_eval" and "apps_metric" metrics you are about to use, execute untrusted 
model-generated code in Python.
Although it is highly unlikely that model-generated code will do something
overtly malicious in response to this test suite, model-generated code may act
destructively due to a lack of model capability or alignment.
Users are strongly encouraged to sandbox this evaluation suite so that it
does not perform destructive actions on their host or network. For more
information on how OpenAI sandboxes its code, see the paper "Evaluating Large
Language Models Trained on Code" (https://arxiv.org/abs/2107.03374).
Once you have read this disclaimer and taken appropriate precautions, set the argument 
"allow_code_execution" to True.
################################################################################\
"""

class Evaluator():
    def __init__(self, model, tokenizer, args):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

        # setup arguments
        self.output_path = args.output_path
        self.seed = args.seed

        # code evaluation permission
        self.allow_code_execution = args.allow_code_execution

        # evaluation dataset arguments
        self.level_apps = args.level_apps
        
    def generate_text(self, task):

        set_seed(self.seed)

        if task == "apps":
            dataset = load_dataset("codeparrot/apps", split="test", difficulties=[self.level_apps])
            generations = apps_utils.make_generations(self.model, self.tokenizer, dataset, self.args, self.args.num_tasks_apps)
            return generations, None

        elif task == "humaneval":
            dataset = load_dataset("openai_humaneval", split="test")
            generations, references = humaneval_mbpp_utils.make_generations(self.model, self.tokenizer, dataset, self.args, self.args.num_tasks_humaneval)
            return generations, references

        elif task == "mbpp":
            dataset = load_dataset("mbpp", split="test")
            generations, references = humaneval_mbpp_utils.make_generations(self.model, self.tokenizer, dataset, self.args, self.args.num_tasks_mbpp)
            return generations, references

        else:
            raise ValueError(f"Task {task} is not supported, please choose from apps, humaneval, or mbpp")

    def evaluate(self, task):

        if not self.allow_code_execution:
            print(_WARNING)
            raise ValueError("Code evaluation is not enabled. Read the warning above carefully and then use `--allow_code_execution=True` flag to enable code evaluation.")
        generations, references = self.generate_text(task)

        if task == "apps":
            code_metric = load("loubnabnl/apps_metric2")
            results = code_metric.compute(predictions=generations)

        else:
            os.environ["HF_ALLOW_CODE_EVAL"] = "1"
            # make sure tokenizer plays nice with multiprocessing
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            code_metric = load("code_eval")

            results, _ = code_metric.compute(
                references=references, predictions=generations, num_workers=self.args.num_workers
            )

        return results
