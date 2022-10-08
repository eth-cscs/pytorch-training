import argparse
import deepspeed
import datasets
import torch
from datasets import load_dataset
from transformers import (AdamW, AutoModelForSeq2SeqLM,
                          AutoTokenizer, DataCollatorForSeq2Seq)
from datasets.utils import disable_progress_bar
from datasets import disable_caching


disable_progress_bar()
disable_caching()

# Benchmark settings
parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark')
parser.add_argument('--model', type=str, default='t5-3b',
                    help='Name of the model from HuggingFace')
parser.add_argument('--download-only', action='store_true',
                    help='Download model, tokenizer, etc and exit')
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

model_name = args.model
tokenizer = AutoTokenizer.from_pretrained(
    model_name, use_fast=True,
    cache_dir=f'./cache/{model_name}_tokenizer'
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    cache_dir=f'./cache/{model_name}_model'
)

if args.download_only:
    exit()

hf_dataset = load_dataset('xsum')


def preprocess_function(examples):    
    inputs = examples['document']
    targets = examples['summary']
    inputs = [f'summarize: {inp}' for inp in inputs]
    model_inputs = tokenizer(inputs, max_length=1024,
                             padding=False, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128,
                           padding=False, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


processed_datasets = hf_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=hf_dataset["train"].column_names,
    num_proc=12
)

data_collator = DataCollatorForSeq2Seq(tokenizer)

train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["validation"]

weight_decay = 0.0
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters()
                   if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters()
                   if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]

model_engine, optimizer, trainloader, __ = deepspeed.initialize(
    args=args, model=model, model_parameters=optimizer_grouped_parameters,
    training_data=train_dataset, collate_fn=data_collator
)

# model.train()
for epoch in range(1):
    for step, batch in enumerate(trainloader):
        # outputs = model(input_ids=batch['input_ids'].to(model_engine.device),
        #                 attention_mask=batch['attention_mask'].to(model_engine.device),
        #                 labels=batch['labels'].to(model_engine.device),
        #                 # decoder_input_ids=batch['decoder_input_ids'].to(model_engine.device)
        #                 )
        outputs = model(**batch.to(model_engine.device))
        model_engine.backward(outputs.loss)
        model_engine.step()
