import datetime
import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional

from cehrgpt.gpt_utils import is_inpatient_visit_type_token


@dataclass(frozen=True)
class CehrGptEvent:
    time: datetime.datetime
    code: str
    domain: str
    text_value: Optional[str] = None
    numeric_value: Optional[float] = None
    unit: Optional[str] = None
    code_label: Optional[str] = None
    visit_id: Optional[int] = None
    record_id: Optional[int] = None

    def get_code_label(self):
        if self.code_label:
            return self.code_label
        return self.code

    def __hash__(self):
        # Use a large prime number for hashing combination
        prime = 31
        # Start with a hash of 1
        hash_code = 1

        # Combine hash of the datetime, assuming current_datetime is part of the instance
        # Convert datetime to timestamp for consistent hashing
        hash_code = hash_code * prime + hash(self.time)
        hash_code = hash_code * prime + hash(self.code)
        hash_code = hash_code * prime + hash(self.domain)
        hash_code = hash_code * prime + hash(
            self.text_value if self.text_value is not None else 0
        )
        hash_code = hash_code * prime + hash(
            self.numeric_value if self.numeric_value is not None else 0
        )
        hash_code = hash_code * prime + hash(self.unit if self.unit is not None else 0)
        hash_code = hash_code * prime + hash(
            self.visit_id if self.visit_id is not None else 0
        )
        return hash_code


@dataclass(frozen=True)
class CehrGptVisit:
    visit_type: str
    visit_concept_id: int
    visit_start_datetime: datetime.datetime
    patient_id: Optional[int] = None
    visit_id: Optional[int] = None
    visit_end_datetime: Optional[datetime.datetime] = None
    discharge_facility: Optional[str] = None
    discharge_to_concept_id: Optional[int] = None
    events: List[CehrGptEvent] = field(default_factory=list)

    def get_narrative(self, birth_datetime: datetime.datetime) -> str:
        birth_year = birth_datetime.year
        age = self.visit_start_datetime.year - birth_year
        narrative = f"\n{self.visit_type} on {self.visit_start_datetime.date().strftime('%Y-%m-%d')} (Age: {age})\n"
        if is_inpatient_visit_type_token(self.visit_concept_id):
            group_by_date = defaultdict(lambda: defaultdict(list))
            for event in self.events:
                group_by_date[event.time.date()][event.domain].append(
                    event.get_code_label()
                )
            for date, domain_concepts in group_by_date.items():
                narrative += (
                    f"\tOn day {(date - self.visit_start_datetime.date()).days}:\n"
                )
                for domain in sorted(domain_concepts):
                    narrative += f"\t\t{domain}:\n"
                    for concept in domain_concepts[domain]:
                        narrative += f"\t\t   * {concept}\n"
        else:
            group_by_domain = defaultdict(list)
            for event in self.events:
                group_by_domain[event.domain].append(event.get_code_label())
            for domain in sorted(group_by_domain):
                narrative += f"\t{domain}:\n"
                for concept in group_by_domain[domain]:
                    narrative += f"\t   * {concept}\n"
        return narrative


@dataclass(frozen=True)
class CehrGptPatient:
    birth_datetime: datetime.datetime
    gender_concept_id: int
    gender: str
    race_concept_id: int
    race: str
    patient_id: Optional[int] = None
    visits: List[CehrGptVisit] = field(default_factory=list)

    def get_narrative(self) -> str:
        narrative = (
            f"Patient Demographics:\n\tGender: {self.gender}\n\tRace: {self.race}\n"
        )
        for visit in self.visits:
            narrative += visit.get_narrative(self.birth_datetime)
        return narrative

    def get_events(self) -> List[CehrGptEvent]:
        return itertools.chain.from_iterable(visit.events for visit in self.visits)


def parse_datetime(datetime_str: str) -> datetime.datetime:
    """Parse datetime string to datetime object, handling multiple formats."""
    try:
        # First try ISO format
        return datetime.datetime.fromisoformat(datetime_str)
    except ValueError:
        try:
            # Try RFC format (e.g., 'Tue, 01 Jan 2013 00:00:00 GMT')
            return parsedate_to_datetime(datetime_str)
        except Exception:
            # If all else fails, try a few common formats
            for fmt in [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%a, %d %b %Y %H:%M:%S %Z",
                "%Y-%m-%d",
            ]:
                try:
                    return datetime.datetime.strptime(datetime_str, fmt)
                except ValueError:
                    continue
            raise ValueError(f"Unable to parse datetime string: {datetime_str}")


def create_event(event_data: Dict[Any, Any]) -> CehrGptEvent:
    """Create a CehrGptEvent from dictionary data."""
    # Convert time string to datetime
    event_data["time"] = parse_datetime(event_data["time"])

    # Create CehrGptEvent with only the fields that are present in the data
    return CehrGptEvent(
        **{
            k: v
            for k, v in event_data.items()
            if k in CehrGptEvent.__dataclass_fields__
        }
    )


def create_visit(visit_data: Dict[Any, Any]) -> CehrGptVisit:
    """Create a CehrGptVisit from dictionary data."""
    # Convert datetime strings to datetime objects
    visit_data["visit_start_datetime"] = parse_datetime(
        visit_data["visit_start_datetime"]
    )
    if visit_data.get("visit_end_datetime"):
        visit_data["visit_end_datetime"] = parse_datetime(
            visit_data["visit_end_datetime"]
        )

    # Handle events list
    if "events" in visit_data:
        visit_data["events"] = [create_event(event) for event in visit_data["events"]]

    # Create CehrGptVisit with only the fields that are present in the data
    return CehrGptVisit(
        **{
            k: v
            for k, v in visit_data.items()
            if k in CehrGptVisit.__dataclass_fields__
        }
    )


def load_cehrgpt_patient_from_json(data: Dict[str, Any]) -> CehrGptPatient:
    # Convert birth_datetime string to datetime object
    data["birth_datetime"] = parse_datetime(data["birth_datetime"])
    # Handle visits list
    if "visits" in data:
        data["visits"] = [create_visit(visit) for visit in data["visits"]]
    # Create CehrGptPatient with only the fields that are present in the data
    return CehrGptPatient(
        **{k: v for k, v in data.items() if k in CehrGptPatient.__dataclass_fields__}
    )
