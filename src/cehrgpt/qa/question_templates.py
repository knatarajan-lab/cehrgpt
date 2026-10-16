"""
Template based question answering pairs generated from a patient timeline.

Every question is about the future: whether and when an event happens after the index date.
There are two use cases, each with its own question type and answer format:

* binary classification (``will_occur``): "Will X happen within N days?" is answered with
  exactly "Yes." or "No.", so the probability of "Yes" is a usable score
* time to event (``time_to_event``): "How many days until X?" is answered with a number of
  days, or with a not-observed statement when the patient did not have it during the follow-up

The answers are read off the events that follow it, so a model that sees only the patient
history (through the linear probing state) has to predict them. Nothing asks about the
history itself. Everything here is plain python and deterministic given the random generator,
so no patient data leaves the machine.
"""

import random
from dataclasses import dataclass
from typing import Collection, Dict, List, Optional, Sequence, Tuple

SECONDS_PER_DAY = 24 * 3600

YES_ANSWER = "Yes."
NO_ANSWER = "No."

# Several wordings per question so the language model does not latch on to one phrasing
WILL_OCCUR_PHRASINGS = (
    "Will the patient have {concept} within {horizon} days?",
    "Is the patient going to be diagnosed with or treated for {concept} in the next {horizon} days?",
    "Does the patient develop {concept} within the next {horizon} days?",
    "Within {horizon} days from now, will {concept} occur for this patient?",
)
TIME_TO_EVENT_PHRASINGS = (
    "How many days until the patient's next {concept}?",
    "After the index date, how many days until {concept} is recorded for this patient?",
    "When will the patient next have {concept}?",
    "How soon will {concept} occur for this patient?",
)


@dataclass
class QAConfig:
    """
    Controls which questions are generated and how often.

    Attributes:
        horizons_in_days: the windows a will-occur question can ask about
        will_occur_weight: relative frequency of will-occur questions
        time_to_event_weight: relative frequency of time-to-event questions
        will_occur_positive_fraction: share of will-occur questions about an event that does
            happen. The real base rate of any single event is much lower, so a model trained
            at 0.5 will over-predict "yes"; lower it to match the prevalence you want.
        time_to_event_positive_fraction: same for time-to-event questions
        paraphrase: use all the wordings, otherwise only the first of each
        new_onset_only: only ask about concepts the patient has not had before the index
            date, so every question is about a new event and not a repeat of a recorded one
        out_of_scope_fraction: share of questions about concepts that are NOT MOTOR tasks,
            because the point of the language model is to answer beyond the fixed task set.
            None leaves the choice to the patient's events; it needs motor_codes to be given
            to generate_qa_pairs, and falls back to any concept when none of that kind exists
    """

    horizons_in_days: Tuple[int, ...] = (30, 90, 180, 365, 730)
    will_occur_weight: float = 1.0
    time_to_event_weight: float = 1.0
    will_occur_positive_fraction: float = 0.5
    time_to_event_positive_fraction: float = 0.7
    paraphrase: bool = True
    new_onset_only: bool = False
    out_of_scope_fraction: Optional[float] = None

    def __post_init__(self):
        for name in ("will_occur_positive_fraction", "time_to_event_positive_fraction"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.out_of_scope_fraction is not None and not (
            0.0 <= self.out_of_scope_fraction <= 1.0
        ):
            raise ValueError("out_of_scope_fraction must be between 0 and 1")
        if not self.horizons_in_days or min(self.horizons_in_days) < 1:
            raise ValueError("horizons_in_days must be positive")
        if self.will_occur_weight < 0 or self.time_to_event_weight < 0:
            raise ValueError("The question weights cannot be negative")
        if self.will_occur_weight + self.time_to_event_weight == 0:
            raise ValueError("At least one question type needs a positive weight")


@dataclass
class PatientTimeline:
    """
    A patient's tokenized sequence split at the index date.

    Attributes:
        concept_ids: the concept tokens of the whole sequence
        epoch_times: POSIX seconds of every token, same length as concept_ids
        index_time: POSIX seconds of the index date; the model only sees what is before it
        age_at_index: the patient's age at the index date, if known
    """

    concept_ids: List[str]
    epoch_times: List[float]
    index_time: float
    age_at_index: Optional[int] = None

    def __post_init__(self):
        if len(self.concept_ids) != len(self.epoch_times):
            raise ValueError("concept_ids and epoch_times must have the same length")


@dataclass
class QAPair:
    question: str
    answer: str
    qa_type: str
    # The concept the question is about, so questions can be grouped (e.g. in or out of
    # the MOTOR task set) and concepts can be held out for evaluation
    concept_id: str
    # Whether the event happened in the window asked about (the answer is not "No")
    event_occurred: bool = False


class ClinicalEvents:
    """The clinical concepts that follow the index date, and which ones came before it."""

    def __init__(self, timeline: PatientTimeline, concept_names: Dict[str, str]):
        # Concepts already recorded before the index date; used only to filter the questions
        self.prior_concepts = set()
        self.first_future_time: Dict[str, float] = {}
        self.follow_up_end = timeline.index_time
        for concept_id, epoch_time in zip(timeline.concept_ids, timeline.epoch_times):
            self.follow_up_end = max(self.follow_up_end, epoch_time)
            # Only concepts that can be spelled out; this drops [VS], time and demographic tokens
            if not concept_id.isdigit() or concept_id not in concept_names:
                continue
            if epoch_time <= timeline.index_time:
                self.prior_concepts.add(concept_id)
            elif concept_id not in self.first_future_time:
                self.first_future_time[concept_id] = epoch_time
        self.index_time = timeline.index_time

    def days_to_first(self, concept_id: str) -> Optional[float]:
        if concept_id not in self.first_future_time:
            return None
        return (self.first_future_time[concept_id] - self.index_time) / SECONDS_PER_DAY

    @property
    def follow_up_days(self) -> float:
        return (self.follow_up_end - self.index_time) / SECONDS_PER_DAY


def _plural(count: int, unit: str) -> str:
    return f"{count} {unit}" if count == 1 else f"{count} {unit}s"


class ConceptCoverage:
    """
    Keeps track of how often each concept was asked about and prefers the least asked.

    Without it the questions follow the data: frequent concepts are asked about all the time
    and rare ones almost never. Choosing the least asked among a few random candidates
    spreads the questions over the whole vocabulary, positives and negatives alike.
    """

    def __init__(self, num_candidates: int = 8):
        if num_candidates < 1:
            raise ValueError("num_candidates must be at least 1")
        self.num_candidates = num_candidates
        self.counts: Dict[str, int] = {}

    def choose(self, candidates: Sequence[str], rng: random.Random) -> str:
        if len(candidates) > self.num_candidates:
            candidates = rng.sample(list(candidates), self.num_candidates)
        return min(candidates, key=lambda c: self.counts.get(c, 0))

    def record(self, concept_id: str) -> None:
        self.counts[concept_id] = self.counts.get(concept_id, 0) + 1


def _pick_concept(
    events: ClinicalEvents,
    pool: Sequence[str],
    rng: random.Random,
    positive: bool,
    config: QAConfig,
    motor_codes: Collection[str],
    coverage: Optional[ConceptCoverage],
) -> Optional[str]:
    num_candidates = coverage.num_candidates if coverage else 1
    want_out_of_scope = None
    if config.out_of_scope_fraction is not None and motor_codes:
        want_out_of_scope = rng.random() < config.out_of_scope_fraction
    # Try the preferred kind of concept first, then fall back to any
    for respect_scope in (True, False) if want_out_of_scope is not None else (False,):

        def acceptable(concept_id: str) -> bool:
            if config.new_onset_only and concept_id in events.prior_concepts:
                return False
            if respect_scope and (concept_id not in motor_codes) != want_out_of_scope:
                return False
            return True

        if positive:
            candidates = [c for c in events.first_future_time if acceptable(c)]
        else:
            # Rejection sampling: the pool is the whole vocabulary, so filtering it for every
            # question would be far too slow, and nearly every draw is acceptable
            candidates = []
            for _ in range(10 * num_candidates if pool else 0):
                concept_id = rng.choice(pool)
                if (
                    concept_id not in events.first_future_time
                    and concept_id not in candidates
                    and acceptable(concept_id)
                ):
                    candidates.append(concept_id)
                    if len(candidates) >= num_candidates:
                        break
        if candidates:
            return coverage.choose(candidates, rng) if coverage else rng.choice(candidates)
    return None


def _phrasing(phrasings: Sequence[str], config: QAConfig, rng: random.Random) -> str:
    return rng.choice(phrasings) if config.paraphrase else phrasings[0]


def will_occur(
    events: ClinicalEvents,
    concept_names: Dict[str, str],
    negative_pool: Sequence[str],
    rng: random.Random,
    config: QAConfig,
    motor_codes: Collection[str] = (),
    coverage: Optional[ConceptCoverage] = None,
) -> Optional[QAPair]:
    """Binary classification: will <concept> happen within a horizon? Answered Yes. or No."""
    positive = rng.random() < config.will_occur_positive_fraction
    concept_id = _pick_concept(
        events, negative_pool, rng, positive, config, motor_codes, coverage
    )
    if concept_id is None:
        return None
    horizon = rng.choice(config.horizons_in_days)
    question = _phrasing(WILL_OCCUR_PHRASINGS, config, rng).format(
        concept=concept_names[concept_id], horizon=horizon
    )
    days = events.days_to_first(concept_id)
    if days is not None and days <= horizon:
        answer = YES_ANSWER
    elif events.follow_up_days >= horizon:
        answer = NO_ANSWER
    else:
        # The record ends before the horizon, so the absence of the event proves nothing
        return None
    return QAPair(
        question, answer, "will_occur", concept_id, event_occurred=answer == YES_ANSWER
    )


def time_to_event(
    events: ClinicalEvents,
    concept_names: Dict[str, str],
    negative_pool: Sequence[str],
    rng: random.Random,
    config: QAConfig,
    motor_codes: Collection[str] = (),
    coverage: Optional[ConceptCoverage] = None,
) -> Optional[QAPair]:
    """Time to event: how long until <concept>? Without the event, the answer gives the follow-up."""
    positive = rng.random() < config.time_to_event_positive_fraction
    concept_id = _pick_concept(
        events, negative_pool, rng, positive, config, motor_codes, coverage
    )
    if concept_id is None:
        return None
    question = _phrasing(TIME_TO_EVENT_PHRASINGS, config, rng).format(
        concept=concept_names[concept_id]
    )
    days = events.days_to_first(concept_id)
    if days is not None:
        answer = f"About {_plural(max(int(round(days)), 1), 'day')}."
    elif events.follow_up_days >= 1:
        answer = (
            f"It was not observed in the {_plural(int(events.follow_up_days), 'day')} "
            "of follow-up."
        )
    else:
        return None
    return QAPair(
        question, answer, "time_to_event", concept_id, event_occurred=days is not None
    )


def generate_qa_pairs(
    timeline: PatientTimeline,
    concept_names: Dict[str, str],
    negative_pool: Sequence[str],
    rng: random.Random,
    num_questions: int = 4,
    config: Optional[QAConfig] = None,
    motor_codes: Collection[str] = (),
    coverage: Optional[ConceptCoverage] = None,
) -> List[QAPair]:
    """
    Generate up to num_questions distinct question/answer pairs for one patient.

    Args:
        timeline: the patient's sequence and index date
        concept_names: concept token to readable name
        negative_pool: the concepts that can be asked about when they did not happen, as a
            list; normally every eligible concept of the vocabulary
        rng: source of randomness, seed it for reproducible datasets
        num_questions: how many pairs to try to produce
        config: question mix, horizons and base rates, see QAConfig
        motor_codes: the concepts that are MOTOR tasks; only used for out_of_scope_fraction
        coverage: if given, prefers concepts that were asked about the least so far and is
            updated with every accepted question
    """
    config = config or QAConfig()
    templates = [
        (will_occur, config.will_occur_weight),
        (time_to_event, config.time_to_event_weight),
    ]
    events = ClinicalEvents(timeline, concept_names)
    pairs: List[QAPair] = []
    seen = set()
    # A few extra attempts because a template can decline (e.g. censored follow-up)
    for _ in range(num_questions * 3):
        if len(pairs) >= num_questions:
            break
        template = rng.choices(
            [t for t, _ in templates], weights=[w for _, w in templates]
        )[0]
        pair = template(
            events, concept_names, negative_pool, rng, config, motor_codes, coverage
        )
        if pair is not None and pair.question not in seen:
            seen.add(pair.question)
            pairs.append(pair)
            if coverage is not None:
                coverage.record(pair.concept_id)
    return pairs
