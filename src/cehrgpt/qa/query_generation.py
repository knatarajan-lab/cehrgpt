"""
Binary queries about a patient's future.

A query asks: "does the patient have this concept, or any descendant of it, in this window?"
The window is given in days after the index date, (start, end]. The label is read off the
events that follow the index date; the model only sees the patient history.
"""

import bisect
import random
from dataclasses import dataclass
from typing import Collection, Dict, List, Optional, Sequence, Set, Tuple

from cehrgpt.qa.concept_hierarchy import ConceptHierarchy
from cehrgpt.qa.concept_filter import is_artificial_token

SECONDS_PER_DAY = 24 * 3600

# Cumulative windows (from the index date) and windows that start later
DEFAULT_WINDOWS = (
    (0, 30),
    (0, 90),
    (0, 180),
    (0, 365),
    (0, 730),
    (30, 90),
    (90, 180),
    (180, 365),
)


@dataclass
class PatientTimeline:
    """
    A patient's tokenized sequence split at the index date.

    Attributes:
        concept_ids: the concept tokens of the whole sequence
        epoch_times: POSIX seconds of every token, same length as concept_ids
        index_time: POSIX seconds of the index date; the model only sees what is before it
    """

    concept_ids: List[str]
    epoch_times: List[float]
    index_time: float

    def __post_init__(self):
        if len(self.concept_ids) != len(self.epoch_times):
            raise ValueError("concept_ids and epoch_times must have the same length")


@dataclass(frozen=True)
class Query:
    concept_id: str
    start_days: int
    end_days: int
    label: bool


@dataclass
class QueryConfig:
    """
    Controls which queries are generated.

    Attributes:
        windows: the (start, end) windows in days after the index date, start < end
        positive_fraction: share of queries whose answer is yes. The real base rate of any
            single concept is far lower, so lower it to match the prevalence you want
        new_onset_only: skip concepts the patient already had before the index date
            (including through a descendant), so every yes is a new event
        out_of_scope_fraction: share of queries about concepts that are not MOTOR tasks;
            None leaves the choice to the data. Needs motor_codes in generate_queries
    """

    windows: Tuple[Tuple[int, int], ...] = DEFAULT_WINDOWS
    positive_fraction: float = 0.5
    new_onset_only: bool = False
    out_of_scope_fraction: Optional[float] = None

    def __post_init__(self):
        if not self.windows:
            raise ValueError("windows cannot be empty")
        for start, end in self.windows:
            if start < 0 or end <= start:
                raise ValueError(f"A window needs 0 <= start < end, got {(start, end)}")
        if not 0.0 <= self.positive_fraction <= 1.0:
            raise ValueError("positive_fraction must be between 0 and 1")
        if self.out_of_scope_fraction is not None and not (
            0.0 <= self.out_of_scope_fraction <= 1.0
        ):
            raise ValueError("out_of_scope_fraction must be between 0 and 1")


class PatientFuture:
    """
    When each query concept happens for one patient.

    An event of a token counts for the token and for all its ancestors, which is what makes
    a query about a concept cover its descendants.
    """

    def __init__(
        self,
        timeline: PatientTimeline,
        hierarchy: ConceptHierarchy,
        eligible_tokens: Collection[str],
        query_concepts: Collection[str],
    ):
        self.index_time = timeline.index_time
        self.follow_up_end = timeline.index_time
        self.prior_concepts: Set[str] = set()
        offsets: Dict[str, List[float]] = {}
        for token, epoch_time in zip(timeline.concept_ids, timeline.epoch_times):
            self.follow_up_end = max(self.follow_up_end, epoch_time)
            if is_artificial_token(token) or token not in eligible_tokens:
                continue
            covered = [c for c in hierarchy.ancestors(token) if c in query_concepts]
            if epoch_time <= timeline.index_time:
                self.prior_concepts.update(covered)
            else:
                days = (epoch_time - timeline.index_time) / SECONDS_PER_DAY
                for concept_id in covered:
                    offsets.setdefault(concept_id, []).append(days)
        self.event_days = {c: sorted(times) for c, times in offsets.items()}

    @property
    def follow_up_days(self) -> float:
        return (self.follow_up_end - self.index_time) / SECONDS_PER_DAY

    def has_event(self, concept_id: str, start_days: float, end_days: float) -> bool:
        """Whether the concept, or a descendant, occurs in (start_days, end_days]."""
        times = self.event_days.get(concept_id)
        if not times:
            return False
        position = bisect.bisect_right(times, start_days)
        return position < len(times) and times[position] <= end_days


class QueryCoverage:
    """
    Counts how often each concept was queried and prefers the least queried.

    Without it the queries follow the data: frequent concepts are queried all the time and
    rare ones almost never. Taking the least queried among a few random candidates spreads
    the queries over the whole vocabulary.
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


def _pick(
    future: PatientFuture,
    pool: Sequence[str],
    rng: random.Random,
    window: Tuple[int, int],
    positive: bool,
    config: QueryConfig,
    motor_codes: Collection[str],
    coverage: Optional[QueryCoverage],
) -> Optional[str]:
    start, end = window
    num_candidates = coverage.num_candidates if coverage else 1
    want_out_of_scope = None
    if config.out_of_scope_fraction is not None and motor_codes:
        want_out_of_scope = rng.random() < config.out_of_scope_fraction
    for respect_scope in (True, False) if want_out_of_scope is not None else (False,):

        def acceptable(concept_id: str) -> bool:
            if config.new_onset_only and concept_id in future.prior_concepts:
                return False
            if respect_scope and (concept_id not in motor_codes) != want_out_of_scope:
                return False
            return True

        if positive:
            candidates = [
                c
                for c in future.event_days
                if future.has_event(c, start, end) and acceptable(c)
            ]
        else:
            # Rejection sampling: the pool is the whole vocabulary, so filtering it for every
            # query would be far too slow, and nearly every draw is acceptable
            candidates = []
            for _ in range(10 * num_candidates if pool else 0):
                concept_id = rng.choice(pool)
                if (
                    concept_id not in candidates
                    and not future.has_event(concept_id, start, end)
                    and acceptable(concept_id)
                ):
                    candidates.append(concept_id)
                    if len(candidates) >= num_candidates:
                        break
        if candidates:
            return coverage.choose(candidates, rng) if coverage else rng.choice(candidates)
    return None


def generate_queries(
    future: PatientFuture,
    pool: Sequence[str],
    rng: random.Random,
    num_queries: int = 4,
    config: Optional[QueryConfig] = None,
    motor_codes: Collection[str] = (),
    coverage: Optional[QueryCoverage] = None,
) -> List[Query]:
    """
    Generate up to num_queries distinct labelled queries for one patient.

    A no is only produced when the record covers the whole window: if the patient's follow-up
    ends earlier, the absence of an event proves nothing and the query is skipped.

    Args:
        future: the patient's events after the index date
        pool: the concepts that can be queried when they did not happen, as a list; normally
            every query concept of the vocabulary
        rng: source of randomness, seed it for reproducible datasets
        num_queries: how many queries to try to produce
        config: windows, base rate and filters, see QueryConfig
        motor_codes: the concepts that are MOTOR tasks; only used for out_of_scope_fraction
        coverage: if given, prefers concepts queried the least so far and is updated with
            every accepted query
    """
    config = config or QueryConfig()
    queries: List[Query] = []
    seen = set()
    # A few extra attempts because a query can be impossible (e.g. censored follow-up)
    for _ in range(num_queries * 3):
        if len(queries) >= num_queries:
            break
        window = rng.choice(config.windows)
        positive = rng.random() < config.positive_fraction
        if not positive and future.follow_up_days < window[1]:
            continue
        concept_id = _pick(
            future, pool, rng, window, positive, config, motor_codes, coverage
        )
        if concept_id is None or (concept_id, window) in seen:
            continue
        seen.add((concept_id, window))
        queries.append(Query(concept_id, window[0], window[1], label=positive))
        if coverage is not None:
            coverage.record(concept_id)
    return queries
