"""
The labels of the open vocabulary head: which questions happen after each prediction point of a
patient, and when.

An event of the patient counts for its own question and for every question above it in the concept
hierarchy, the same roll-up as the MOTOR labels: a stay at an Inpatient Hospital (a visit record) is
also an Inpatient Visit, an Angina is also a Coronary disease and a Heart disease, a Metformin tablet
is also Metformin. QuestionHierarchy holds that roll-up, built from the universes of
question_universe and the concept_ancestor table, and build_question_labels turns the events of one
patient into the arguments of open_vocab_nll_loss.

Only OMOP concept ids are used. The tokens of a sequence are strings, so a concept id is an int or a
string of digits ("9202"); anything else (time tokens, [VS], ETHOS/COMET tokens such as "Visit/IP",
"year:2020") is not a concept and is ignored.

A question that does not happen after a prediction point is censored at the end of the record, so it
is simply left out of the events; open_vocab_nll_loss treats a question that is not listed for a row as
censored. Only the first occurrence of a question matters.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import polars as pl


def to_concept_id(token) -> Optional[int]:
    """The concept id of a token: an int, or a string of digits; None for any other token."""
    if isinstance(token, (int, np.integer)) and not isinstance(token, bool):
        return int(token)
    if isinstance(token, str) and token.isascii() and token.isdigit():
        return int(token)
    return None


class QuestionHierarchy:
    """For every concept of the tokenizer, the questions it counts for: itself and its ancestors."""

    def __init__(self, questions_by_concept: Mapping[int, Sequence[int]]):
        self._questions = {int(c): tuple(sorted({int(q) for q in qs})) for c, qs in questions_by_concept.items()}
        # The same roll-up as arrays, for build_question_labels: concept number -> the slice
        # question_ids[offsets[n]:offsets[n + 1]]. A token is looked up as an int or as a string.
        concepts = list(self._questions)
        self._number = {}
        for n, concept in enumerate(concepts):
            self._number[concept] = n
            self._number[str(concept)] = n
        lengths = np.fromiter((len(self._questions[c]) for c in concepts), dtype=np.int64, count=len(concepts))
        self._offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        flat = [q for c in concepts for q in self._questions[c]]
        self._question_ids = np.asarray(flat, dtype=np.int64)

    def concept_numbers(self, tokens: Sequence) -> np.ndarray:
        """
        The number of each token in the hierarchy (the row of the lookup arrays), -1 for a token that
        is not a concept of it: time tokens, [VS], ETHOS/COMET tokens, unknown concepts.
        """
        get = self._number.get
        numbers = np.fromiter((get(t, -1) for t in tokens), dtype=np.int64, count=len(tokens))
        # a bool or a float must not match the int key it happens to equal, and "0510" is concept 510
        for i in np.nonzero(numbers >= 0)[0]:
            if isinstance(tokens[i], (bool, float)):
                numbers[i] = -1
        for i in np.nonzero(numbers < 0)[0]:
            concept = to_concept_id(tokens[i])
            if concept is not None:
                numbers[i] = self._number.get(concept, -1)
        return numbers

    def expand(self, numbers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        For concept numbers (all >= 0), the pairs (index into numbers, question id): one pair for each
        question each concept counts for.
        """
        starts = self._offsets[numbers]
        counts = self._offsets[numbers + 1] - starts
        owner = np.repeat(np.arange(len(numbers)), counts)
        first = np.cumsum(counts) - counts
        flat = starts[owner] + (np.arange(int(counts.sum())) - first[owner])
        return owner, self._question_ids[flat]

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

    def questions_of(self, concept_id) -> Tuple[int, ...]:
        """
        The questions the concept counts for; empty for a concept that is no token of the universe and
        for a token that is not a concept id at all.
        """
        concept_id = to_concept_id(concept_id)
        return self._questions.get(concept_id, ()) if concept_id is not None else ()

    def __contains__(self, concept_id) -> bool:
        concept_id = to_concept_id(concept_id)
        return concept_id is not None and concept_id in self._questions

    def vocab_numbers(self, vocab: Mapping[str, int]) -> np.ndarray:
        """
        For every id of a tokenizer vocabulary the concept number of its token (-1 if it is no concept
        of the hierarchy), so that the input ids of a batch map to concept numbers in one indexing.
        """
        tokens = list(vocab)
        numbers = np.full(max(vocab.values()) + 1, -1, dtype=np.int64)
        numbers[np.fromiter((vocab[t] for t in tokens), dtype=np.int64, count=len(tokens))] = self.concept_numbers(tokens)
        return numbers

    def save(self, path: str) -> None:
        """Write the roll-up as a parquet file of (concept_id, question_id) pairs."""
        concepts = [c for c, qs in self._questions.items() for _ in qs]
        questions = [q for qs in self._questions.values() for q in qs]
        pl.DataFrame({"concept_id": concepts, "question_id": questions}, schema={"concept_id": pl.Int64, "question_id": pl.Int64}).write_parquet(path)

    @classmethod
    def load(cls, path: str) -> "QuestionHierarchy":
        pairs = pl.read_parquet(path)
        questions: Dict[int, set] = {}
        for concept, question in pairs.iter_rows():
            questions.setdefault(concept, set()).add(question)
        return cls(questions)

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
        event_concept_ids: the concept id of every event of the record (ints or digit strings), visit
            records included (the visit concept of a visit is an event at the start of the visit);
            anything that is not in the hierarchy (time tokens, [VS], ETHOS/COMET tokens, unknown
            concepts) is ignored
        event_days: the day of every event, on any common scale (e.g. days since 1970)
        prediction_days: the day of each prediction point; each is a row
        end_day: the day the record ends, where the questions that did not happen are censored
        question_ids: the questions asked, in the order of their slot; the same set is asked at every
            prediction point
    A question happens after a prediction point if one of its events is strictly later than the point.
    """
    prediction_days = np.asarray(prediction_days, dtype=np.float64)
    censor_times = np.maximum(float(end_day) - prediction_days, 0.0)
    numbers = hierarchy.concept_numbers(event_concept_ids)
    event_rows, event_questions, event_times = first_events_after(
        hierarchy, numbers, np.asarray(event_days, dtype=np.float64), prediction_days, question_ids
    )
    return QuestionLabels(event_rows, event_questions, event_times, censor_times)


def first_events_after(
    hierarchy: QuestionHierarchy,
    numbers: np.ndarray,
    event_days: np.ndarray,
    prediction_days: np.ndarray,
    question_ids: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The core of build_question_labels, for events given as concept numbers of the hierarchy
    (QuestionHierarchy.concept_numbers, -1 for an event that is no concept).

    Returns event_rows, event_questions (the slot in question_ids) and event_times, ordered by row and slot.
    """
    rows = len(prediction_days)
    asked = np.asarray(question_ids, dtype=np.int64)  # the questions are concept ids
    if len(np.unique(asked)) != len(asked):
        raise ValueError("question_ids has to be unique, a question is asked once")

    def empty():
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)

    known = numbers >= 0
    if rows == 0 or len(asked) == 0 or not known.any():
        return empty()
    days = event_days[known]
    # every event counts for each question above it: one entry per (event, question)
    owner, questions = hierarchy.expand(numbers[known])
    # keep the questions that are asked, with their slot
    by_id = np.argsort(asked, kind="stable")
    sorted_ids = asked[by_id]
    where = np.minimum(np.searchsorted(sorted_ids, questions), len(asked) - 1)
    is_asked = sorted_ids[where] == questions
    if not is_asked.any():
        return empty()
    slots = by_id[where[is_asked]]
    event_days_q = days[owner[is_asked]]

    # Per question the days of its events in increasing order. The first event strictly after a row at
    # time t is the first one with day > t, so an event at day d answers the rows with
    # previous_day <= t < d, previous_day being the day of the previous event of the same question.
    order = np.lexsort((event_days_q, slots))
    slots, event_days_q = slots[order], event_days_q[order]
    same_question = np.zeros(len(slots), dtype=bool)
    same_question[1:] = slots[1:] == slots[:-1]
    previous_day = np.full(len(slots), -np.inf)
    previous_day[1:] = np.where(same_question[1:], event_days_q[:-1], -np.inf)

    # the rows in time order, so that the rows of an event are a contiguous range of them
    row_order = np.argsort(prediction_days, kind="stable")
    sorted_times = prediction_days[row_order]
    low = np.searchsorted(sorted_times, previous_day, side="left")  # first row with t >= previous_day
    high = np.searchsorted(sorted_times, event_days_q, side="left")  # rows below this have t < day
    counts = np.maximum(high - low, 0)
    total = int(counts.sum())
    if total == 0:
        return empty()
    owner = np.repeat(np.arange(len(slots)), counts)
    first_of_owner = np.cumsum(counts) - counts
    sorted_rows = low[owner] + (np.arange(total) - first_of_owner[owner])
    event_rows = row_order[sorted_rows].astype(np.int64)
    event_questions = slots[owner].astype(np.int64)
    event_times = event_days_q[owner] - prediction_days[event_rows]

    final = np.lexsort((event_questions, event_rows))
    return event_rows[final], event_questions[final], event_times[final]


@dataclass
class BatchQuestionLabels:
    """The questions asked of a batch and when they happen, rows counted over the whole batch."""

    question_rows: np.ndarray  # (K,) rows of the question embedding matrix, the slot is the position here
    event_rows: np.ndarray
    event_questions: np.ndarray  # slot in question_rows
    event_times: np.ndarray


def sample_questions(
    candidate_rows: np.ndarray,
    num_rows_total: int,
    num_questions: int,
    positive_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    The rows of the question embeddings asked of one batch.

    A share of the questions (positive_fraction) is drawn from the candidates, the questions that
    happen to some patient of the batch, and the rest uniformly from all questions, so the model sees both
    what happens and what does not. The result has no duplicates and at most num_questions entries.
    """
    num_questions = min(num_questions, num_rows_total)
    positives = min(int(round(num_questions * positive_fraction)), len(candidate_rows))
    chosen = rng.choice(candidate_rows, size=positives, replace=False) if positives else np.zeros(0, dtype=np.int64)
    missing = num_questions - len(chosen)
    if missing > 0:
        extra = rng.choice(num_rows_total, size=min(num_rows_total, missing + len(chosen)), replace=False)
        extra = extra[~np.isin(extra, chosen)][:missing]
        chosen = np.concatenate([chosen, extra])
    return chosen.astype(np.int64)


def build_batch_question_labels(
    hierarchy: QuestionHierarchy,
    question_concept_ids: np.ndarray,
    segments: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    num_questions: int,
    positive_fraction: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> BatchQuestionLabels:
    """
    The labels of a batch of patients, one question set for all of them.

    Args:
        question_concept_ids: the concept id of every row of the question embedding matrix
        segments: one per patient, (concept numbers of the events, day of the events, positions of the
            prediction points among the events); the prediction points are the rows of the batch in
            the order of the segments, then of the positions
        num_questions: how many questions to ask, see sample_questions
    """
    rng = rng if rng is not None else np.random.default_rng()
    question_concept_ids = np.asarray(question_concept_ids, dtype=np.int64)
    by_id = np.argsort(question_concept_ids, kind="stable")
    sorted_ids = question_concept_ids[by_id]

    prepared = []
    candidates = []
    for numbers, days, positions in segments:
        positions = np.asarray(positions, dtype=np.int64)
        days = np.maximum.accumulate(np.asarray(days, dtype=np.float64)) if len(days) else np.asarray(days, dtype=np.float64)
        prepared.append((np.asarray(numbers, dtype=np.int64), days, positions))
        known = np.asarray(numbers) >= 0
        if len(positions) == 0 or not known.any():
            continue
        owner, questions = hierarchy.expand(np.asarray(numbers, dtype=np.int64)[known])
        # a question can be asked about a patient if it happens after the first prediction point
        after = days[known][owner] > days[positions].min()
        candidates.append(questions[after])
    if candidates:
        candidate_ids = np.unique(np.concatenate(candidates))
        where = np.minimum(np.searchsorted(sorted_ids, candidate_ids), len(sorted_ids) - 1)
        candidate_rows = by_id[where[sorted_ids[where] == candidate_ids]]
    else:
        candidate_rows = np.zeros(0, dtype=np.int64)
    question_rows = sample_questions(candidate_rows, len(question_concept_ids), num_questions, positive_fraction, rng)
    asked = question_concept_ids[question_rows]

    all_rows, all_questions, all_times = [], [], []
    offset = 0
    for numbers, days, positions in prepared:
        if len(positions):
            rows, slots, times = first_events_after(hierarchy, numbers, days, days[positions], asked)
            all_rows.append(rows + offset)
            all_questions.append(slots)
            all_times.append(times)
        offset += len(positions)
    if all_rows:
        return BatchQuestionLabels(
            question_rows, np.concatenate(all_rows), np.concatenate(all_questions), np.concatenate(all_times)
        )
    return BatchQuestionLabels(question_rows, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64))
