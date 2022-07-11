# This script is based on
# https://keras.io/examples/nlp/text_extraction_with_bert/

import argparse
import deepspeed
import os
import dataset_utils as du
import eval_utils as eu
import torch
from datasets import load_dataset, load_metric
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import DataLoader
from torch.nn import functional as F
from datetime import datetime


def print_peak_memory(prefix, device):
    if device == 0:
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")


# Benchmark settings
parser = argparse.ArgumentParser(description='BERT finetuning on SQuAD')
parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
parser.add_argument('--download-only', action='store_true',
                    help='Download model, tokenizer, etc and exit')
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

hf_model = 'bert-base-uncased'
bert_cache = os.path.join(os.getcwd(), 'cache')

slow_tokenizer = BertTokenizer.from_pretrained(
    hf_model,
    cache_dir=os.path.join(bert_cache, f'_{hf_model}-tokenizer')
)
save_path = os.path.join(bert_cache, f'{hf_model}-tokenizer')
if not os.path.exists(save_path):
    os.makedirs(save_path)
    slow_tokenizer.save_pretrained(save_path)

# Load the fast tokenizer from saved file
tokenizer = BertWordPieceTokenizer(os.path.join(save_path, 'vocab.txt'),
                                   lowercase=True)

model = BertForQuestionAnswering.from_pretrained(
    hf_model,
    cache_dir=os.path.join(bert_cache, f'{hf_model}_qa')
)

if args.download_only:
    exit()

model.train()

hf_dataset = load_dataset('squad')

max_len = 384

ds_filtered = hf_dataset.filter(
    lambda example: du.filter_squad_bad(example, max_len, tokenizer),
    num_proc=12
)
ds_processed = ds_filtered.map(
    lambda example: du.process_squad_item(example, max_len, tokenizer),
    remove_columns=hf_dataset["train"].column_names,
    num_proc=4
)

train_set = ds_processed["train"]
train_set.set_format(type='torch')

parameters = filter(lambda p: p.requires_grad, model.parameters())

model_engine, optimizer, trainloader, __ = deepspeed.initialize(
    args=args,
    model=model,
    model_parameters=parameters,
    training_data=train_set
)

rank = torch.distributed.get_rank()
print_peak_memory(f"Rank -{rank}: Max memory allocated after creating DDP", 0)


# training
for epoch in range(2):  # loop over the dataset multiple times
    for i, batch in enumerate(trainloader, 0):
        outputs = model(input_ids=batch['input_ids'].to(model_engine.device),
                        token_type_ids=batch['token_type_ids'].to(model_engine.device),
                        attention_mask=batch['attention_mask'].to(model_engine.device),
                        start_positions=batch['start_token_idx'].to(model_engine.device),
                        end_positions=batch['end_token_idx'].to(model_engine.device))
        # forward + backward + optimize
        loss = outputs[0]
        model_engine.backward(loss)
        model_engine.step()
        # print_peak_memory("Max memory allocated after optimizer step", 0)

        # if i > 10:
        #     break

if rank == 0:
    print('Finished Training')


    if os.environ['SLURM_NODEID'] == '0':
        model_hash = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        model_path_name = f'./cache/model_trained_deepspeed_{model_hash}'
    
        # save model's state_dict
        torch.save(model.state_dict(), model_path_name)
    
        # create the model again since the previous one is on the gpu
        model_cpu = BertForQuestionAnswering.from_pretrained(
            "bert-base-uncased",
            cache_dir=os.path.join(bert_cache, 'bert-base-uncased_qa')
        )
    
        # load the model on cpu
        model_cpu.load_state_dict(
            torch.load(model_path_name,
                       map_location=torch.device('cpu'))
        )
    
        # load the model on gpu
        # model.load_state_dict(torch.load(model_path_name))
        # model.eval()

        squad_example_objects = [du.create_squad_example(item, max_len, tokenizer)
                                 for item in ds_filtered['validation']]
        eval_set = ds_processed["validation"]
        eval_set.set_format(type='torch')
        batch_size = 1

        eval_dataloader = DataLoader(
            eval_set,
            shuffle=False,
            batch_size=batch_size
        )

        for i, eval_batch in enumerate(eval_dataloader):
            if not i%1000:
                eu.EvalUtility(eval_batch, squad_example_objects[i:i + batch_size], model_cpu).results()

            if i > 10000:
                break
