# Code Generation LM Evaluation Harness [WIP]

A framework for the evaluation of autoregressive code generation language models. 

## Overview

This project provides a unified framework to test autoregressive code generation language models.

Features:
- Any autoregressive model available on [Hugging Face hub](https://huggingface.co/) can be used, but we recommend using a code generation models trained specifically on Code such as [CodeParrot](https://huggingface.co/codeparrot/codeparrot), [InCoder](https://huggingface.co/facebook/incoder-6B) and [CodeGen](https://huggingface.co/Salesforce/codegen-16B-mono).
- 3 tasks implemented: [HumanEval](https://huggingface.co/datasets/openai_humaneval), [APPS](https://huggingface.co/datasets/codeparrot/apps) and [MBPP](https://huggingface.co/datasets/mbpp).


## Setup

```bash
git clone https://github.com/loubnabnl/code-evaluation-harness.git
cd code-evaluation-harness
pip install -r requirements.txt
```

## Basic Usage

To evaluate a model, (e.g. CodeParrot) on HumanEval and APPS benchmarks, you can run the following command:

```bash
python main.py \
	--model codeparrot/codeparrot \
	--device 0 \
	--tasks humaneval,apps
```

## Acknowledgements
This repository is inspired from [EleutherAI's LM evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness).
