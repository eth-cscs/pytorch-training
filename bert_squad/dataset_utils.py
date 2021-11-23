import os
import numpy as np


class SquadExample:
    def __init__(self, question, context, start_char_idx, answer_text,
                 all_answers, max_len, tokenizer):
        self.question = question
        self.context = context
        self.start_char_idx = start_char_idx
        self.answer_text = answer_text
        self.all_answers = all_answers
        self.max_len = max_len
        self.skip = False
        self.tokenizer = tokenizer

    def preprocess(self):
        context = self.context
        question = self.question
        answer_text = self.answer_text
        start_char_idx = self.start_char_idx

        # Clean context, answer and question
        context = " ".join(str(context).split())
        question = " ".join(str(question).split())
        answer = " ".join(str(answer_text).split())

        # Find end character index of answer in context
        end_char_idx = start_char_idx + len(answer)
        if end_char_idx >= len(context):
            self.skip = True
            return

        # Mark the character indexes in context that are in answer
        is_char_in_ans = [0] * len(context)
        for idx in range(start_char_idx, end_char_idx):
            is_char_in_ans[idx] = 1

        # Tokenize context
        tokenized_context = self.tokenizer.encode(context)

        # Find tokens that were created from answer characters
        ans_token_idx = []
        for idx, (start, end) in enumerate(tokenized_context.offsets):
            if sum(is_char_in_ans[start:end]) > 0:
                ans_token_idx.append(idx)

        if len(ans_token_idx) == 0:
            self.skip = True
            return

        # Find start and end token index for tokens from answer
        start_token_idx = ans_token_idx[0]
        end_token_idx = ans_token_idx[-1]

        # Tokenize question
        tokenized_question = self.tokenizer.encode(question)

        # Create inputs
        input_ids = tokenized_context.ids + tokenized_question.ids[1:]
        token_type_ids = [0] * len(tokenized_context.ids) + [1] * len(
            tokenized_question.ids[1:]
        )
        attention_mask = [1] * len(input_ids)

        # Pad and create attention masks.
        # Skip if truncation is needed
        padding_length = self.max_len - len(input_ids)
        if padding_length > 0:  # pad
            input_ids = input_ids + ([0] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
        elif padding_length < 0:  # skip
            self.skip = True
            return

        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.start_token_idx = start_token_idx
        self.end_token_idx = end_token_idx
        self.context_token_to_char = tokenized_context.offsets


def process_squad_item(item, max_len, tokenizer):
    squad_examples = []
    for para in item["paragraphs"]:
        context = para["context"]
        for qa in para["qas"]:
            question = qa["question"]
            answer_text = qa["answers"][0]["text"]
            all_answers = [_["text"] for _ in qa["answers"]]
            start_char_idx = qa["answers"][0]["answer_start"]
            squad_eg = SquadExample(
                question, context, start_char_idx, answer_text,
                all_answers, max_len, tokenizer)
            squad_eg.preprocess()
            if not squad_eg.skip:
                squad_examples.append(squad_eg)
    return np.array(squad_examples)


def create_squad_examples(raw_data, max_len, tokenizer):
    squad_examples = []
    for item in raw_data["data"]:
        squad_examples.extend(process_squad_item(item, max_len, tokenizer))
    return np.array(squad_examples)


def create_inputs_targets(squad_examples, shuffle=False, seed=0):
    dataset_dict = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],
        "start_token_idx": [],
        "end_token_idx": [],
    }

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(squad_examples)

    for item in squad_examples:
        if not item.skip:
            for key in dataset_dict:
                dataset_dict[key].append(getattr(item, key))

    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = (
        dataset_dict["input_ids"],
        dataset_dict["token_type_ids"],
        dataset_dict["attention_mask"],
    )
    y = (dataset_dict["start_token_idx"], dataset_dict["end_token_idx"])
    return x, y
