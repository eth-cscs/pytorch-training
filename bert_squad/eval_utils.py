import re
import string
import torch
import numpy as np
from torch.nn import functional as F

from rich.console import Console
from rich.table import Table

def normalize_text(text):
    text = text.lower()

    # Remove punctuations
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)

    # Remove articles
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    text = re.sub(regex, " ", text)

    # Remove extra white space
    text = " ".join(text.split())
    return text


class EvalUtility():
    """
    Each `SquadExample` object contains the character level offsets for each
    token in its input paragraph. We use them to get back the span of text
    corresponding to the tokens between our predicted start and end tokens.
    All ground-truth answers are also present in each `SquadExample` object.
    We calculate the percentage of data points where the span of text obtained
    from model predictions matches one of the ground-truth answers.
    """

    def __init__(self, x_eval, model, squad_examples):
        self.model = model
        self.squad_examples = squad_examples
        self.input_ids = torch.tensor(x_eval[0])
        self.token_type_ids = torch.tensor(x_eval[1])
        self.attention_mask = torch.tensor(x_eval[2])
        
        self.set_rich_print()

    def results(self, logs=None):
        outputs_eval = self.model(input_ids=self.input_ids,
                                  token_type_ids=self.token_type_ids,
                                  attention_mask=self.attention_mask
                                  )
        pred_start = F.softmax(outputs_eval.start_logits,
                               dim=-1).cpu().detach().numpy()
        pred_end = F.softmax(outputs_eval.end_logits,
                             dim=-1).cpu().detach().numpy()
        count = 0
        eval_examples_no_skip = [_ for _ in self.squad_examples
                                 if _.skip is False]
        for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
            squad_eg = eval_examples_no_skip[idx]
            offsets = squad_eg.context_token_to_char
            start = np.argmax(start)
            end = np.argmax(end)
            if start >= len(offsets):
                continue

            pred_char_start = offsets[start][0]
            if end < len(offsets):
                pred_char_end = offsets[end][1]
                pred_ans = squad_eg.context[pred_char_start:pred_char_end]
            else:
                pred_ans = squad_eg.context[pred_char_start:]

            normalized_pred_ans = normalize_text(pred_ans)
            normalized_true_ans = [normalize_text(_)
                                   for _ in squad_eg.all_answers]
            if normalized_pred_ans in normalized_true_ans:
                count += 1

            # print(f'  - {normalized_pred_ans:30.30s} |'
            #       f' ref: {squad_eg.answer_text:30s} |'
            #       f' {squad_eg.question}')

            self.table.add_row(f' {squad_eg.question}',
                               f' {normalized_pred_ans:30.30s}',
                               f' {squad_eg.answer_text:30s}')
        self.show_table()

    def set_rich_print(self):

        self.table = Table(title="Evaluation", show_lines=True)
        self.table.add_column("Question", justify="right", style="green")
        self.table.add_column("Model's answer", justify="right", style="cyan", no_wrap=True)
        self.table.add_column("Reference", style="magenta")

    def show_table(self):
        console = Console()
        console.print(self.table)
