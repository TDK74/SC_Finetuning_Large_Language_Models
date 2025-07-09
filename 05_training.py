import os
import datasets
import tempfile
import logging
import random
import config
import yaml
import time
import torch
import transformers
import pandas as pd
import jsonlines
import lamini

from utilities import *
from transformers import AutoTokenizer
from transformers import AutoModelForCasualLM
from transformers import TrainingArguments


lamini.api_url = os.getenv("POWERML__PRODUCTION__URL")
lamini.api_key = os.getenv("POWERML__PRODUCTION__KEY")

logger = logging.getLogger(__name__)
global_config = None

## ------------------------------------------------------##
dataset_name = "lamini_docs.jsonl"
dataset_path = f"/content/{dataset_name}"
use_hf = False

## ------------------------------------------------------##
dataset_path = "lamini/lamini_docs"
use_hf = True

## ------------------------------------------------------##
model_name = "EleutherAI/pythia-70m"

## ------------------------------------------------------##
training_config = {
                    "model" : {
                            "pretrained_name" : model_name,
                            "max_length" : 2048
                            },
                    "datasets" : {
                                "use_hf" : use_hf,
                                "path" : dataset_path
                                },
                    "verbose" : True
                }

## ------------------------------------------------------##
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
train_dataset, test_dataset = tokenize_and_split_data(training_config, tokenizer)

print(train_dataset)
print(test_dataset)

## ------------------------------------------------------##
base_model = AutoModelForCasualLM.from_pretrained(model_name)

## ------------------------------------------------------##
device_count = torch.cuda.device_count()

if device_count > 0:
    logger.debug("Select GPU device.")
    device = torch.device("cuda")

else:
    logger.debug("Select CPU device.")
    device = torch.device("cpu")

## ------------------------------------------------------##
base_model.to(device)

## ------------------------------------------------------##
def inference(text, model, tokenizer, max_input_tokens = 1000, max_output_tokens = 100):
    input_ids = tokenizer.encode(text,
                                 return_tensors = "pt",
                                 truncation = True,
                                 max_length = max_input_tokens)

    device = model.device
    generated_tokens_with_prompt = model.generate(input_ids = input_ids.to(device),
                                                  max_length = max_output_tokens)

    generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt,
                                                        skip_special_tokens = True)

    generated_text_answer = generated_text_with_prompt[0][len(text) : ]

    return generated_text_answer

## ------------------------------------------------------##
test_text = test_dataset[0]['question']
print("Question input (test): ", test_text)
print(f"Correct answer from Lamini docs: {test_dataset[0]['answer']}")
print("Model's answer: ")
print(inference(test_text, base_model, tokenizer))

## ------------------------------------------------------##
max_steps = 3

## ------------------------------------------------------##
trained_model_name = f"lamini_docs{max_steps}_steps"
output_dir = trained_model_name

## ------------------------------------------------------##
training_args = TrainingArguments(
                                learning_rate = 1.0e-5,
                                num_train_epochs = 1,
                                max_steps = max_steps,
                                per_device_train_batch_size = 1,
                                output_dir = output_dir,
                                overwrite_output_dir = False,
                                disable_tqdm = False,
                                eval_steps = 120,
                                save_steps = 120,
                                warmup_steps = 1,
                                per_device_eval_batch_size = 1,
                                evaluation_strategy = "steps",
                                logging_strategy = "steps",
                                logging_steps = 1,
                                optim = "adafactor",
                                gradient_accumulation_steps = 1,
                                gradient_checkpoint = False,
                                load_best_model_at_end = True,
                                save_total_limit = 1,
                                metric_for_best_model = "eval_loss",
                                greater_is_better = False
                                )

## ------------------------------------------------------##
model_flops = (
                base_model.floating_point_ops(
                                                {
                                                "input_ids" : torch.zeros(
                                                                        (1,
                                                                         training_config["model"]["max_length"])
                                                                        )
                                                }
                                            )
                * training_args.gradient_accumulation_steps
                )

print(base_model)
print("Memory footprint", base_model.get_memory_footprint() / 1e9, "GB")
print("Flops", model_flops / 1e9, "GFLOPs")

## ------------------------------------------------------##
trainer = Trainer(
                model = base_model,
                model_flops = model_flops,
                total_steps = max_steps,
                args = training_args,
                train_dataset = train_dataset,
                eval_dataset = test_dataset,
                )

## ------------------------------------------------------##
training_output = trainer.train()

## ------------------------------------------------------##
save_dir = f'{output_dir}/final'

trainer.save_model(save_dir)
print("Saved model to: ", save_dir)

## ------------------------------------------------------##
finetuned_slightly_model = AutoModelForCasualLM.fomr_pretrained(save_dir, local_files_only = True)

## ------------------------------------------------------##
finetuned_slightly_model.to(device)

## ------------------------------------------------------##
test_question = test_dataset[0]['question']
print("Question input(test): ", test_question)

print("Finetuned slightly model's answer: ")
print(inference(test_question, finetuned_slightly_model, tokenizer))

## ------------------------------------------------------##
test_answer = test_dataset[0]['answer']
print("Target answer output (test): ", test_answer)

## ------------------------------------------------------##
finetuned_longer_model = AutoModelForCasualLM.from_pretrained("lamini/lamini_docs_finetuned")
tokenizer = AutoTokenizer.from_pretrained("lamini/llamini_docs_finetuned")

finetuned_longer_model.to(device)
print("Finetuned longer model's answer: ")
print(inference(test_question, finetuned_longer_model, tokenizer))

## ------------------------------------------------------##
bigger_finetuned_model = BasicModelRunner(model_name_to_id["bigger_model_name"])
bigger_finetuned_output = bigger_finetuned_model(test_question)
print("Bigger (2.8B) finetuned model (test): ", bigger_finetuned_output)

## ------------------------------------------------------##
count = 0

for i in range(len(train_dataset)):
    if "keep the discussion relevant to Lamini" in train_dataset[i]["answer"]:
        print(i, train_dataset[i]["question"], train_dataset[i]["answer"])
        count += 1

print(count)

## ------------------------------------------------------##
base_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
base_model = AutoModelForCasualLM.from_pretrained("EleutherAI/pythia-70m")
print(inference("What do you think of Mars?", base_model, base_tokenizer))

## ------------------------------------------------------##
print(inference("What do you think of Mars?", finetuned_longer_model, tokenizer))

## ------------------------------------------------------##
model = BasicModelRunner("EleutherAI/pythia-410m")
model.load_data_from_jsonlines("lamini_docs.jsonl",
                               input_key = "question",
                               output_key = "answer")
model.train(is_public = True)

## ------------------------------------------------------##
out = model.evaluate()

## ------------------------------------------------------##
lofd = []

for e in out['eval_results']:
    q = f"{e['input']}"
    at = f"{e['outputs'][0]['output']}"
    ab= f"{e['outputs'][1]['output']}"
    di = {'question' : q, 'trained_model' : at, 'Base Model' : ab}
    lofd.append(di)

df = pd.DataFrame.from_dict(lofd)
style_df = df.style.set_properties(**{'text-aligh' : 'left'})
style_df = style_df.set_properties(**{"vertical-align" : "text-top"})
style_df
