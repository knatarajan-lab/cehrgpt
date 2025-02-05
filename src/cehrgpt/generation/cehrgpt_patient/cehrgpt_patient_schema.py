import datetime
from typing import List, NotRequired, TypedDict

from cehrbert.med_extension.schema_extension import Event


class CehrGptEvent(Event):
    visit_id: int
    domain: NotRequired[str]
    record_id: NotRequired[int]


class CehrGptVisit(TypedDict):
    patient_id: int
    visit_id: int
    visit_type: str
    visit_concept_id: int
    visit_start_datetime: datetime.datetime
    visit_end_datetime: NotRequired[datetime.datetime]
    discharge_facility: NotRequired[str]
    discharge_to_concept_id: NotRequired[int]
    events: List[CehrGptEvent]


class CehrGptPatient(TypedDict):
    patient_id: int
    birth_datetime: datetime.datetime
    gender_concept_id: int
    gender: str
    race_concept_id: int
    race: str
    visits: List[CehrGptVisit]
