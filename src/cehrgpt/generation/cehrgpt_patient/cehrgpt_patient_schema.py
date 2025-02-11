import datetime
import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional

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

    def get_narrative(
        self, birth_datetime: datetime.datetime, html_output: bool = False
    ) -> str:
        birth_year = birth_datetime.year
        age = self.visit_start_datetime.year - birth_year
        if html_output:
            narrative = f"<div><strong>{self.visit_type}</strong> on {self.visit_start_datetime.date().strftime('%Y-%m-%d')} (Age: {age})<br></div>"
        else:
            narrative = f"\n{self.visit_type} on {self.visit_start_datetime.date().strftime('%Y-%m-%d')} (Age: {age})\n"

        if is_inpatient_visit_type_token(self.visit_concept_id):
            group_by_date = defaultdict(lambda: defaultdict(list))
            for event in self.events:
                group_by_date[event.time.date()][event.domain].append(
                    event.get_code_label()
                )
            for date, domain_concepts in group_by_date.items():
                if html_output:
                    narrative += f"<div style='padding-left:20px;'>On day {(date - self.visit_start_datetime.date()).days}:<br></div>"
                else:
                    narrative += (
                        f"\tOn day {(date - self.visit_start_datetime.date()).days}:\n"
                    )
                for domain in sorted(domain_concepts):
                    if html_output:
                        narrative += f"<div style='padding-left:40px;'><strong>{domain}:</strong><br></div>"
                    else:
                        narrative += f"\t\t{domain}:\n"
                    for concept in domain_concepts[domain]:
                        if html_output:
                            narrative += (
                                f"<div style='padding-left:60px;'>* {concept}<br></div>"
                            )
                        else:
                            narrative += f"\t\t   * {concept}\n"
        else:
            group_by_domain = defaultdict(list)
            for event in self.events:
                group_by_domain[event.domain].append(event.get_code_label())
            for domain in sorted(group_by_domain):
                if html_output:
                    narrative += f"<div style='padding-left:20px;'><strong>{domain}:</strong><br></div>"
                else:
                    narrative += f"\t{domain}:\n"
                for concept in group_by_domain[domain]:
                    if html_output:
                        narrative += (
                            f"<div style='padding-left:40px;'>* {concept}<br></div>"
                        )
                    else:
                        narrative += f"\t   * {concept}\n"

        if html_output:
            narrative += "</div>"  # Closing div for the narrative

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

    def get_narrative(self, html_output: bool = False) -> str:
        if html_output:
            narrative = (
                f"<div>Patient Demographics:<br>"
                f"<ul>"
                f"<li>Gender: {self.gender}</li>"
                f"<li>Race: {self.race}</li>"
                f"</ul>"
                f"</div>"
            )
        else:
            narrative = (
                f"Patient Demographics:\n\tGender: {self.gender}\n\tRace: {self.race}\n"
            )
        for visit in self.visits:
            narrative += visit.get_narrative(self.birth_datetime, html_output)
        return narrative

    def get_events(self) -> List[CehrGptEvent]:
        return itertools.chain.from_iterable(visit.events for visit in self.visits)
