"""
The concept hierarchy behind a query such as "any diabetes in the next 90 days".

A query concept covers itself and all its descendants: an event of a descendant (say Type 2
diabetes) answers a query about the ancestor (Diabetes mellitus). The hierarchy comes from the
ontology saved with the tokenizer (ontology.pickle).
"""

import logging
import os
import pickle
from collections import defaultdict
from typing import Collection, Dict, FrozenSet, Iterable, List, Optional, Set

from cehrgpt.tokenization.tokenization_constants import ONTOLOGY_FILE_NAME

LOG = logging.getLogger(__name__)


class ConceptHierarchy:
    def __init__(
        self,
        parents: Optional[Dict[str, Set[str]]] = None,
        domains: Optional[Dict[str, str]] = None,
    ):
        """
        Args:
            parents: concept -> its direct parents; None or empty means no hierarchy, so a
                concept only covers itself
            domains: concept -> OMOP domain, if known
        """
        self._parents = {k: set(v) for k, v in (parents or {}).items()}
        self.domains = dict(domains) if domains else None
        self._ancestors: Dict[str, FrozenSet[str]] = {}

    @property
    def has_hierarchy(self) -> bool:
        return any(self._parents.values())

    @classmethod
    def from_ontology(cls, ontology) -> "ConceptHierarchy":
        return cls(
            dict(ontology.parents_map),
            {k: v for k, v in ontology.concept_domain_map.items() if v},
        )

    @classmethod
    def from_tokenizer_folder(cls, tokenizer_path: str) -> "ConceptHierarchy":
        path = os.path.join(tokenizer_path, ONTOLOGY_FILE_NAME)
        if not os.path.exists(path):
            LOG.warning(
                "%s not found: a query concept will only cover itself, not its descendants, "
                "and the domains of the concepts are unknown",
                path,
            )
            return cls()
        with open(path, "rb") as f:
            return cls.from_ontology(pickle.load(f))

    def ancestors(self, concept_id: str) -> FrozenSet[str]:
        """The concept itself and every ancestor above it."""
        if concept_id not in self._ancestors:
            seen: Set[str] = set()
            stack = [concept_id]
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                stack.extend(self._parents.get(current, ()))
            self._ancestors[concept_id] = frozenset(seen)
        return self._ancestors[concept_id]

    def domain(self, concept_id: str) -> Optional[str]:
        return self.domains.get(concept_id) if self.domains else None

    def subtree_sizes(self, tokens: Iterable[str]) -> Dict[str, int]:
        """For every ancestor of the tokens, how many of the tokens it covers."""
        sizes: Dict[str, int] = defaultdict(int)
        for token in tokens:
            for ancestor in self.ancestors(token):
                sizes[ancestor] += 1
        return dict(sizes)

    def descendants_among(self, concept_id: str, tokens: Iterable[str]) -> List[str]:
        """The tokens covered by a concept (itself included when it is one of them)."""
        return [t for t in tokens if concept_id in self.ancestors(t)]


def query_concepts(
    eligible_tokens: Collection[str],
    hierarchy: ConceptHierarchy,
    allowed_domains: Optional[Collection[str]] = None,
    max_covered_tokens: Optional[int] = None,
) -> Dict[str, int]:
    """
    The concepts a query may be about, with the number of eligible tokens each covers.

    Every eligible token and every ancestor of one is a candidate, so queries range from a
    single procedure up to a broad class of conditions.

    Args:
        eligible_tokens: tokens that are real clinical events (see concept_filter)
        hierarchy: provides the ancestors and, if known, the domains
        allowed_domains: keep only candidates in these domains when the domains are known
        max_covered_tokens: drop broad concepts that cover more tokens than this, because a
            query like "any clinical finding" is true for almost every patient
    """
    sizes = hierarchy.subtree_sizes(eligible_tokens)
    allowed = set(allowed_domains) if allowed_domains else None
    result = {}
    for concept_id, size in sizes.items():
        if max_covered_tokens is not None and size > max_covered_tokens:
            continue
        if allowed is not None and hierarchy.domains:
            if hierarchy.domain(concept_id) not in allowed:
                continue
        result[concept_id] = size
    return result
