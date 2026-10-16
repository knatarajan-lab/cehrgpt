"""
Using a trained patient language model for the two use cases.

* binary classification: the probability that the answer is "Yes." (a score for AUROC etc.)
* time to event: the number of days the model answers, or that the event was not expected
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

from cehrgpt.models.patient_language_model import PatientLanguageModel
from cehrgpt.qa.collator import PatientQACollator
from cehrgpt.qa.question_templates import NO_ANSWER, YES_ANSWER

_DAYS = re.compile(r"about\s+(\d+)\s+days?", re.IGNORECASE)
_NOT_OBSERVED = re.compile(r"not observed", re.IGNORECASE)


@dataclass
class TimeToEventPrediction:
    text: str
    # Days until the event, None when the model says it will not be observed or the text
    # could not be understood
    days: Optional[float]
    # True when the model answered that the event is not expected during the follow-up
    event_not_observed: bool


def parse_time_to_event_answer(text: str) -> TimeToEventPrediction:
    """Read "About 40 days." or "It was not observed in the 200 days of follow-up." """
    if _NOT_OBSERVED.search(text):
        return TimeToEventPrediction(text, None, True)
    match = _DAYS.search(text)
    if match:
        return TimeToEventPrediction(text, float(match.group(1)), False)
    return TimeToEventPrediction(text, None, False)


class PatientQAPredictor:
    """Wraps a model and its text tokenizer to answer questions for many patients."""

    def __init__(
        self,
        model: PatientLanguageModel,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
        batch_size: int = 32,
    ):
        self.model = model.eval()
        self.collator = PatientQACollator(tokenizer, max_length)
        self.batch_size = batch_size
        self.yes_id = self._first_answer_token(YES_ANSWER)
        self.no_id = self._first_answer_token(NO_ANSWER)
        if self.yes_id == self.no_id:
            raise ValueError(
                "The tokenizer gives 'Yes.' and 'No.' the same first token, so the "
                "probability of yes cannot be read from the next-token logits"
            )

    def _first_answer_token(self, answer: str) -> int:
        # The collator writes " " + answer after "Answer:"
        return self.collator.tokenizer(" " + answer, add_special_tokens=False)["input_ids"][0]

    def _batches(self, n: int):
        for start in range(0, n, self.batch_size):
            yield slice(start, min(start + self.batch_size, n))

    @torch.no_grad()
    def probability_of_yes(
        self, patient_features: Sequence[Sequence[float]], questions: Sequence[str]
    ) -> np.ndarray:
        """P(yes) for binary questions, as the share of yes among the yes/no tokens."""
        device = next(self.model.parameters()).device
        probabilities = []
        for chunk in self._batches(len(questions)):
            batch = self.collator.collate_prompts(
                list(questions[chunk]), [list(f) for f in patient_features[chunk]]
            )
            logits = self.model.next_token_logits(
                **{k: v.to(device) for k, v in batch.items()}
            )
            pair = logits[:, [self.yes_id, self.no_id]].float()
            probabilities.append(torch.softmax(pair, dim=-1)[:, 0].cpu().numpy())
        return np.concatenate(probabilities) if probabilities else np.zeros(0)

    @torch.no_grad()
    def predict_time_to_event(
        self,
        patient_features: Sequence[Sequence[float]],
        questions: Sequence[str],
        max_new_tokens: int = 24,
    ) -> List[TimeToEventPrediction]:
        device = next(self.model.parameters()).device
        predictions: List[TimeToEventPrediction] = []
        for chunk in self._batches(len(questions)):
            batch = self.collator.collate_prompts(
                list(questions[chunk]), [list(f) for f in patient_features[chunk]]
            )
            generated = self.model.generate(
                **{k: v.to(device) for k, v in batch.items()},
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            texts = self.collator.tokenizer.batch_decode(generated, skip_special_tokens=True)
            predictions.extend(parse_time_to_event_answer(t.strip()) for t in texts)
        return predictions
