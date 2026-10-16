"""
The labels of the open vocabulary head: which questions happen after each prediction point of a
patient, and when.

An event of the patient counts for its own question and for every question above it in the concept
hierarchy, the same roll-up as the MOTOR labels: a stay at an Inpatient Hospital (a visit record) is
also an Inpatient Visit, an Angina is also a Coronary disease and a Heart disease, a Metformin tablet
is also Metformin. QuestionHierarchy holds that roll-up, built from the universes of
question_universe and the concept_ancestor table, and build_question_labels turns the events of one
patient into the arguments of open_vocab_nll_loss.

A question that does not happen after a prediction point is censored at the end of the record, so it
is simply left out of the events; open_vocab_nll_loss treats a question that is not listed for a row as
censored. Only the first occurrence of a question matters.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import polars as pl


class QuestionHierarchy:
    """For every concept of the tokenizer, the questions it counts for: itself and its ancestors."""

    def __init__(self, questions_by_concept: Mapping[int, Sequence[int]]):
        self._questions = {int(c): tuple(sorted({int(q) for q in qs})) for c, qs in questions_by_concept.items()}

    @classmethod
    def from_universes(cls, universes: Mapping[str, pl.DataFrame], ancestor: pl.DataFrame) -> "QuestionHierarchy":
        """
        Args:
            universes: domain -> the universe of question_universe.build_universe for that domain
            ancestor: concept_ancestor with ancestor_concept_id, descendant_concept_id
        A concept rolls up only to the questions of its own domain, so a drug is never counted for a
        condition because the hierarchy happens to link them.
        """
        questions: Dict[int, set] = {}
        for universe in universes.values():
            tokens = universe.filter(pl.col("is_token"))["concept_id"].to_list()
            question_ids = universe["concept_id"].to_list()
            pairs = ancestor.filter(
                pl.col("descendant_concept_id").is_in(tokens) & pl.col("ancestor_concept_id").is_in(question_ids)
            ).select("descendant_concept_id", "ancestor_concept_id")
            for token, question in pairs.iter_rows():
                questions.setdefault(token, set()).add(question)
            for token in tokens:
                # a token is always a question of its own, also when the ancestor table does not list it
                questions.setdefault(token, set()).add(token)
        return cls(questions)

    def questions_of(self, concept_id: int) -> Tuple[int, ...]:
        """The questions the concept counts for; empty for a concept that is no token of the universe."""
        return self._questions.get(int(concept_id), ())

    def __contains__(self, concept_id: int) -> bool:
        return int(concept_id) in self._questions

    def __len__(self) -> int:
        return len(self._questions)


@dataclass
class QuestionLabels:
    """The arguments of open_vocab_nll_loss for the prediction points of one or more patients."""

    event_rows: np.ndarray  # the prediction point (row) each listed question happens after
    event_questions: np.ndarray  # the slot of the question in the list that was asked
    event_times: np.ndarray  # days from the prediction point to the first time it happens
    censor_times: np.ndarray  # (rows,) days from each prediction point to the end of the record

    @property
    def num_rows(self) -> int:
        return len(self.censor_times)

    @classmethod
    def concatenate(cls, parts: Sequence["QuestionLabels"]) -> "QuestionLabels":
        """Stack the labels of several patients; the rows of the later ones are shifted."""
        offsets = np.cumsum([0] + [p.num_rows for p in parts[:-1]])
        return cls(
            np.concatenate([p.event_rows + o for p, o in zip(parts, offsets)]).astype(np.int64),
            np.concatenate([p.event_questions for p in parts]).astype(np.int64),
            np.concatenate([p.event_times for p in parts]).astype(np.float64),
            np.concatenate([p.censor_times for p in parts]).astype(np.float64),
        )


def build_question_labels(
    hierarchy: QuestionHierarchy,
    event_concept_ids: Sequence[int],
    event_days: Sequence[float],
    prediction_days: Sequence[float],
    end_day: float,
    question_ids: Sequence[int],
) -> QuestionLabels:
    """
    The labels of one patient.

    Args:
        hierarchy: the roll-up from the concepts of the events to the questions
        event_concept_ids: the concept id of every event of the record, visit records included (the
            visit concept of a visit is an event at the start of the visit); anything that is not in
            the hierarchy (time tokens, [VS], unknown concepts) is ignored
        event_days: the day of every event, on any common scale (e.g. days since 1970)
        prediction_days: the day of each prediction point; each is a row
        end_day: the day the record ends, where the questions that did not happen are censored
        question_ids: the questions asked, in the order of their slot; the same set is asked at every
            prediction point
    A question happens after a prediction point if one of its events is strictly later than the point.
    """
    prediction_days = np.asarray(prediction_days, dtype=np.float64)
    rows = len(prediction_days)
    slot = {int(q): i for i, q in enumerate(question_ids)}
    censor_times = np.maximum(float(end_day) - prediction_days, 0.0)

    days_by_slot: Dict[int, List[float]] = {}
    for concept_id, day in zip(event_concept_ids, event_days):
        if int(concept_id) not in hierarchy:
            continue
        for question in hierarchy.questions_of(int(concept_id)):
            if question in slot:
                days_by_slot.setdefault(slot[question], []).append(float(day))

    rows_out, slots_out, times_out = [], [], []
    for question_slot, days in days_by_slot.items():
        days = np.unique(days)  # sorted
        first_after = np.searchsorted(days, prediction_days, side="right")
        happens = first_after < len(days)
        if not happens.any():
            continue
        rows_out.append(np.nonzero(happens)[0])
        slots_out.append(np.full(int(happens.sum()), question_slot))
        times_out.append(days[first_after[happens]] - prediction_days[happens])

    if rows_out:
        event_rows = np.concatenate(rows_out).astype(np.int64)
        event_questions = np.concatenate(slots_out).astype(np.int64)
        event_times = np.concatenate(times_out).astype(np.float64)
        order = np.lexsort((event_questions, event_rows))
        event_rows, event_questions, event_times = event_rows[order], event_questions[order], event_times[order]
    else:
        event_rows = event_questions = np.zeros(0, dtype=np.int64)
        event_times = np.zeros(0, dtype=np.float64)
    return QuestionLabels(event_rows, event_questions, event_times, censor_times)
