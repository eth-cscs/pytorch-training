# This script is based on
# https://keras.io/examples/nlp/text_extraction_with_bert/

import argparse
import deepspeed
import numpy as np
import os
import json
import dataset_utils as du
import eval_utils as eu
import torch
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
    cache_dir=os.path.join(bert_cache, f'{hf_model}-uncased_qa')
)
model.train()

train_path = os.path.join(bert_cache, 'data', 'train-v1.1.json')
eval_path = os.path.join(bert_cache, 'data', 'dev-v1.1.json')
with open(train_path) as f:
    raw_train_data = json.load(f)

with open(eval_path) as f:
    raw_eval_data = json.load(f)

batch_size = 8
max_len = 384

train_squad_examples = du.create_squad_examples(
    raw_train_data,
    max_len,
    tokenizer
)
x_train, y_train = du.create_inputs_targets(
    train_squad_examples,
    shuffle=True,
    seed=42
)
print(f"{len(train_squad_examples)} training points created.")

eval_squad_examples = du.create_squad_examples(
    raw_eval_data,
    max_len,
    tokenizer
)
x_eval, y_eval = du.create_inputs_targets(eval_squad_examples)
print(f"{len(eval_squad_examples)} evaluation points created.")


class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        return (torch.tensor(self.x[0][idx]),
                torch.tensor(self.x[1][idx]),
                torch.tensor(self.x[2][idx]),
                torch.tensor(self.y[0][idx]),
                torch.tensor(self.y[1][idx]))

    def __len__(self):
        return len(self.x[0])


train_set = SquadDataset(x_train, y_train)

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
        outputs = model(input_ids=batch[0].to(model_engine.device),
                        token_type_ids=batch[1].to(model_engine.device),
                        attention_mask=batch[2].to(model_engine.device),
                        start_positions=batch[3].to(model_engine.device),
                        end_positions=batch[4].to(model_engine.device))
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
        model_path_name = './cache/model_trained_deepspeed_{model_hash}'
    
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
    
        model.eval()
    
        samples = np.random.choice(len(x_eval[0]), 50, replace=False)
    
        eu.EvalUtility(
            (x_eval[0][samples], x_eval[1][samples], x_eval[2][samples]),
            model_cpu,
            eval_squad_examples[samples]
        ).results()
