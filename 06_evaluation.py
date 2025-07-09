import datasets
import tempfile
import logging
import random
import config
import os
import yaml
import difflib
import panda as pd

from tqdm import tqdm
from utilities import *
from transformers import AutoTokenizer, AutoModelForCasualLM


logger = logging.getLogger(__name__)
global_config = None

## ------------------------------------------------------##
dataset = datasets.load_dataset("lamini/lamini_docs")

test_dataset = dataset["test"]

## ------------------------------------------------------##
print(test_dataset[0]["question"])
print(test_dataset[0]["answer"])

## ------------------------------------------------------##
model_name = "lamini/lamini_docs_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCasualLM.from_pretrained(model_name)

## ------------------------------------------------------##
def is_exact_match(a, b):
    return a.strip() == b.strip()

## ------------------------------------------------------##
model.eval()

## ------------------------------------------------------##
def inference(text, model, tokenizer, max_inpit_tokens = 1000, max_output_tokens = 100):
    tokenizer.pad_token = tokenizer.eos_token
    input_ids = tokenizer.encode(text,
                                 return_tensors = "pt",
                                 truncation = True,
                                 max_length = max_inpit_tokens)

    device = model.device
    generated_tokens_with_prompt = model.generated(input_ids = input_ids.to(device),
                                               max_length = max_output_tokens)

    generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt,
                                                    skip_sppecial_tokens = True)

    generated_text_answer = generated_text_with_prompt[0][len(text) : ]

    return generated_text_answer

## ------------------------------------------------------##
test_question = test_dataset[0]["question"]
generated_answer = inference(test_question, model, tokenizer)
print(test_question)
print(generated_answer)

## ------------------------------------------------------##
answer = test_dataset[0]["answer"]
print(answer)

## ------------------------------------------------------##
exact_match = is_exact_match(generated_answer, answer)
print(exact_match)

## ------------------------------------------------------##
n = 10
metrics = {'exact_matches' : []}
predictions = []

for i, item in tqdm(enumerate(test_dataset)):
    print("i Evaluating: " + str(item))
    question = item['question']
    answer = item['answer']

    try:
        predicted_answer = inference(question, model, tokenizer)

    except:
        continue

    exact_match = is_exact_match(predicted_answer, answer)
    metrics['exact_matches'].append(exact_match)

    if i > n and n != -1:
        break

    print('Number of exact matches: ', sum(metrics['exact_matches']))

## ------------------------------------------------------##
df = pd.DataFrame(predictions, columns = ["predictec_answer", "target_answer"])
print(df)

## ------------------------------------------------------##
evaluation_dataset_path = "lamini/lamini_docs_evaluation"
evaluation_dataset = datasets.load_dataset(evaluation_dataset_path)

## ------------------------------------------------------##
pd.DataFrame(evaluation_dataset)
