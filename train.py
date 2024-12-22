# Constant variables
MAX_TOKEN_LENGTH = 1024
PHOMT_PREFIX = "translate English to Vietnamese: {text}"
DATA_SOURCE = "Darejkal/phomt_partial"
CACHE_DIR = "cache"
OUT_FOLDER = "out"
IGNORE_INDEX = -100
RESPONSE_OUTFILE = "test_answers.txt"

# Download dataset
from datasets import load_dataset

data = load_dataset(
    path=DATA_SOURCE,
    num_proc=2,
    verification_mode="no_checks",
    cache_dir=CACHE_DIR
)

finetune_data = data['train']
eval_data = data['test']

finetune_data = finetune_data.select(range(500))
eval_data = eval_data.select(range(10))

print(finetune_data)
print(eval_data)

# Print a sample
print(finetune_data[0]['en'])
print(finetune_data[0]['vi'])

# Preprocess data
from transformers import AutoTokenizer
from functools import partial

tokenizer = AutoTokenizer.from_pretrained("google/mt5-small", cache_dir="cache", legacy=False, use_fast=False)
custom_tokenize = partial(tokenizer, max_length=MAX_TOKEN_LENGTH, padding="max_length", truncation=True)

def preprocess_function(examples):
    inputs, answers=list(zip(*[(PHOMT_PREFIX.format(text=examples["en"][i]), examples["vi"][i]) for i in range(len(examples["en"]))]))
    model_inputs = custom_tokenize(text=inputs, text_target=answers)
    return model_inputs

tokenized_train_datasets = finetune_data.map(preprocess_function, batched=True, num_proc=2)
tokenized_eval_datasets = eval_data.map(preprocess_function, batched=True, num_proc=2)

print(tokenized_train_datasets)
print(tokenized_eval_datasets)

# Print a sample
print(tokenized_train_datasets[0]['en'])
print(tokenized_train_datasets[0]['vi'])
print(tokenized_train_datasets[0]['input_ids'])
print(tokenized_train_datasets[0]['attention_mask'])
print(tokenized_train_datasets[0]['labels'])

# Define computing metrics function
import evaluate
import numpy as np
from transformers.trainer_utils import EvalPrediction

eval_metrics = evaluate.combine(["sacrebleu"])

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    preds = np.where(preds != IGNORE_INDEX, preds, tokenizer.pad_token_id)
    labels = np.where(labels != IGNORE_INDEX, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = eval_metrics.compute(predictions=decoded_preds,
                                  references=decoded_labels)
    result = {"bleu": result["score"]}
    return result

# Download pre-trained model
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small", cache_dir=CACHE_DIR, device_map="auto")

# Setting training arguments
from transformers import Seq2SeqTrainingArguments
training_args = Seq2SeqTrainingArguments(
    report_to="none",
    output_dir="/content/working",
    save_steps=100,
    eval_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
    evaluation_strategy="steps",
    seed=117,
    learning_rate=2e-4,
    per_device_eval_batch_size=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    lr_scheduler_type="cosine",
    weight_decay=1e-4,
    max_grad_norm=1.0,
    warmup_ratio=0.01,
    group_by_length=False,
    num_train_epochs=1,
    predict_with_generate=True,
    dataloader_num_workers=4,
    dataloader_prefetch_factor=2,
    generation_num_beams=5,
    generation_max_length=MAX_TOKEN_LENGTH,
)

#
from transformers import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    pad_to_multiple_of=8,
    label_pad_token_id=IGNORE_INDEX
)

# Init trainer and start training
from transformers import Seq2SeqTrainer
trainer =  Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_datasets,
    eval_dataset=tokenized_eval_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()

# Define generating answer function
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_answer(texts):
    input_ids = tokenizer(texts, return_tensors="pt", max_length=1024, padding=True, truncation=True, return_token_type_ids=False).to(device)
    output_ids = model.generate(
        **input_ids,
        num_return_sequences=1,
        num_beams=5,
        early_stopping=True,
        max_length=MAX_TOKEN_LENGTH,
    )
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)

# Generate answer for samples in eval dataset
import datasets
random_eval_ds = eval_data
index = 0

writer = open(RESPONSE_OUTFILE, "w", encoding="utf-8")

while index < len(random_eval_ds):
    next = min(index + 4, len(random_eval_ds))
    texts = [PHOMT_PREFIX.format(text=random_eval_ds[i]["en"]) for i in range(index, next)]
    outputs = generate_answer(texts)
    for output in outputs:
        print(output.strip().replace("\n", " "))
        writer.write(output.strip().replace("\n", " ") + "\n")
    index = next

writer.close()

# Calculate eval metrics
# import evaluate
import json

references = random_eval_ds['vi']
predictions = [
    line.strip() for line in open(RESPONSE_OUTFILE, "r", encoding="utf-8").readlines()
]

sacrebleu = evaluate.load("sacrebleu")
results = sacrebleu.compute(predictions=predictions, references=references)
print(results)

with open(RESPONSE_OUTFILE+"_blue","w") as f:
    f.write(json.dumps(results))

rougue = evaluate.load("rouge")
results = rougue.compute(predictions=predictions, references=references)
print(results)

with open(RESPONSE_OUTFILE+"_rougue","w") as f:
    f.write(json.dumps(results))

# Save model locally
trainer.save_model("/content/saved_model/")

# Push model to HuggingFace
access_token = "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"
model.push_to_hub("h9art/MLOps-toy-mt5", token=access_token)
