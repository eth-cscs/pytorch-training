import os
import dataset_utils as du
import eval_utils as eu
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertForQuestionAnswering
from tokenizers import BertWordPieceTokenizer

from datasets.utils import disable_progress_bar
disable_progress_bar()


hf_model = 'bert-base-uncased'
bert_cache = os.path.join(os.getcwd(), 'cache')
save_path = os.path.join(bert_cache, f'{hf_model}-tokenizer')

# Load the fast tokenizer from saved file
tokenizer = BertWordPieceTokenizer(os.path.join(save_path, 'vocab.txt'),
                                   lowercase=True)

model = BertForQuestionAnswering.from_pretrained(
    hf_model,
    cache_dir=os.path.join(bert_cache, f'{hf_model}_qa')
)

raw_dataset = load_dataset('squad')

val_ds = raw_dataset['validation']

max_len = 384

ds_filtered = raw_dataset['validation'].filter(
    lambda example: du.filter_squad_bad(example, max_len, tokenizer),
    num_proc=12
)

ds_processed = ds_filtered.map(
    lambda example: du.process_squad_item(example, max_len, tokenizer),
    remove_columns=ds_filtered.column_names,
    num_proc=12
)

ds_processed.set_format(type='torch')

batch_size = 1

eval_dataloader = DataLoader(
    ds_processed,
    shuffle=False,
    batch_size=batch_size
)

model_hash = '2022-07-11-172004'
model_path_name = f'./cache/model_trained_deepspeed_{model_hash}'

# load the model on cpu
model.load_state_dict(
    torch.load(model_path_name,
               map_location=torch.device('cpu'))
)

squad_example_objects = [du.create_squad_example(item, max_len, tokenizer)
                         for item in ds_filtered]

for i, eval_batch in enumerate(eval_dataloader):
    if not i%1000:
        eu.EvalUtility(eval_batch, squad_example_objects[i:i + batch_size], model).results()

    if i > 10000:
        break
