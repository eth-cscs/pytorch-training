# Text summarization with T5 on XSum

We are going to fine-tune [T5 implemented by HuggingFace](https://huggingface.co/t5-small) for text summarization on the [Extreme Summarization (XSum)](https://huggingface.co/datasets/xsum) dataset.
The data if composed by news articles and the corresponding summaries.

This notebooks is based on the script [run_summarization_no_trainer.py](https://github.com/huggingface/transformers/blob/v4.12.5/examples/pytorch/summarization/run_summarization_no_trainer.py) from HuggingFace.

We will be using the following hugging sizes available from HuggingFace

| Variant                                     |   Parameters    |
|:-------------------------------------------:|----------------:|
| [T5-small](https://huggingface.co/t5-small) |    60,506,624   | 
| [T5-large](https://huggingface.co/t5-large) |   737,668,096   | 
| [T5-3b](https://huggingface.co/t5-3b)       | 2,851,598,336   | 

### Runnig the model on Piz Daint
```
srun python t5-text-summarization-deepspeed.py --deepspeed_config ds_config.json --model t5-3b
```

More info:
* [T5 on HuggingFace docs](https://huggingface.co/transformers/model_doc/t5.html)
