import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional

from cehrgpt.generation.cehrgpt_patient.cehrgpt_patient_schema import (
    CehrGptEvent,
    CehrGptPatient,
    CehrGptVisit,
)
from cehrgpt.generation.cehrgpt_patient.conversion_validation_rules import (
    clinical_token_types,
    get_validation_rules,
)
from cehrgpt.generation.cehrgpt_patient.typed_tokens import (
    CEHRGPTToken,
    TokenType,
    translate_to_cehrgpt_tokens,
)
from cehrgpt.gpt_utils import (
    DEMOGRAPHIC_PROMPT_SIZE,
    extract_hours_from_hour_token,
    extract_time_interval_in_days,
)


class IdGenerator:
    """
    A class to generate unique identifiers for different types of entities.

    This class manages the generation of unique, sequential identifiers for various entity types including
    persons, visits, conditions, procedures, drug exposures, measurements, and deaths. Each entity type
    has its own counter, starting from a specified initial value.

    Attributes:
        id_values (dict): A dictionary holding the current identifier value for each entity type.

    Args:
        start_value (int): The initial value from which to start the ID counters for all entity types.
    """

    def __init__(self, start_value=1):
        """
        Initialize the IdGenerator with the same starting value for all entity types.

        Args:
            start_value (int): The initial value for all ID counters. Default is 1.
        """
        self.id_values = {
            "person_id": start_value,
            "visit_occurrence_id": start_value,
            "condition_occurrence_id": start_value,
            "procedure_occurrence_id": start_value,
            "drug_exposure_id": start_value,
            "measurement_id": start_value,
            "death_id": start_value,
        }

    def get_next_id_by_domain(self, domain) -> Optional[int]:
        domain = domain.lower()
        if domain == "person":
            return self.get_next_person_id()
        elif domain.startswith("visit"):
            return self.get_next_visit_occurrence_id()
        elif domain.startswith("condition"):
            return self.get_next_condition_occurrence_id()
        elif domain.startswith("procedure"):
            return self.get_next_procedure_occurrence_id()
        elif domain.startswith("drug"):
            return self.get_next_drug_exposure_id()
        elif domain.startswith("death"):
            return self.get_next_death_id()
        elif domain.startswith("measurement"):
            return self.get_next_measurement_id()
        return None

    def _get_next_id(self, key):
        """
        Private method to increment and return the next ID for a given entity type.

        Args:
            key (str): The key corresponding to the entity type.

        Returns:
            int: The next sequential ID for the specified entity type.
        """
        self.id_values[key] += 1
        return self.id_values[key]

    def get_next_person_id(self):
        """
        Get the next unique identifier for a person.

        Returns:
            int: The next sequential person ID.
        """
        return self._get_next_id("person_id")

    def get_next_visit_occurrence_id(self):
        """
        Get the next unique identifier for a visit occurrence.

        Returns:
            int: The next sequential visit occurrence ID.
        """
        return self._get_next_id("visit_occurrence_id")

    def get_next_condition_occurrence_id(self):
        """
        Get the next unique identifier for a condition occurrence.

        Returns:
            int: The next sequential condition occurrence ID.
        """
        return self._get_next_id("condition_occurrence_id")

    def get_next_procedure_occurrence_id(self):
        """
        Get the next unique identifier for a procedure occurrence.

        Returns:
            int: The next sequential procedure occurrence ID.
        """
        return self._get_next_id("procedure_occurrence_id")

    def get_next_drug_exposure_id(self):
        """
        Get the next unique identifier for a drug exposure.

        Returns:
            int: The next sequential drug exposure ID.
        """
        return self._get_next_id("drug_exposure_id")

    def get_next_measurement_id(self):
        """
        Get the next unique identifier for a measurement.

        Returns:
            int: The next sequential measurement ID.
        """
        return self._get_next_id("measurement_id")

    def get_next_death_id(self):
        """
        Get the next unique identifier for a death occurrence.

        Returns:
            int: The next sequential death ID.
        """
        return self._get_next_id("death_id")


@dataclass
class DateTimeCursor:
    """
    A data class for handling a mutable datetime object that can be passed by reference.

    This class provides a simple structure to manage a datetime object that can be updated
    throughout different parts of an application, reflecting changes universally.

    Attributes:
        current_datetime (datetime.datetime): The current datetime object being tracked.
    """

    current_datetime: datetime.datetime

    def update_datetime(self, new_datetime: datetime.datetime):
        """
        Update the current datetime with a new datetime value.

        Args:
            new_datetime (datetime.datetime): The new datetime to replace the current one.
        """
        self.current_datetime = new_datetime

    def add_hours(self, hours: int):
        # Handle hour tokens differently than the day tokens
        # The way we construct the inpatient hour tokens is that the sum of the consecutive
        # hour tokens cannot exceed the current day, so the data_cursor is bounded by a
        # theoretical upper limit
        datetime_upper_bound = self.current_datetime.replace(
            hour=0, minute=0, second=0
        ) + datetime.timedelta(hours=23, minutes=59, seconds=59)
        new_datetime_cursor = self.current_datetime + datetime.timedelta(hours=hours)
        if new_datetime_cursor > datetime_upper_bound:
            new_datetime_cursor = datetime_upper_bound
        self.current_datetime = new_datetime_cursor

    def add_days(self, days: int):
        """
        Add a specified number of days to the current datetime.

        Args:
            days (int): Number of days to add to the current datetime.
        """
        current_date = self.current_datetime.replace(hour=0, minute=0, second=0)
        self.current_datetime = current_date + datetime.timedelta(days=days)

    def __str__(self):
        """
        Return a string representation of the current datetime.

        Returns:
            str: The string representation of the current datetime.
        """
        return self.current_datetime.strftime("%Y-%m-%d %H:%M:%S")


class PatientSequenceConverter:
    def __init__(
        self,
        tokens: List[CEHRGPTToken],
        id_generator: Optional[IdGenerator] = None,
        original_person_id: Optional[int] = None,
    ):
        self.tokens = tokens
        self.error_messages = []
        self.id_generator = id_generator
        self.validation_rules = get_validation_rules()
        self.validate()
        self.person_id = (
            original_person_id
            if original_person_id
            else id_generator.get_next_person_id() if id_generator else None
        )

    @property
    def is_validation_passed(self) -> bool:
        return len(self.error_messages) == 0

    def get_patient(
        self,
        domain_map: Dict[str, str],
        concept_map: Dict[str, str],
    ) -> Optional[CehrGptPatient]:
        if self.is_validation_passed:
            year_token, age_token, gender_token, race_token = self.tokens[
                :DEMOGRAPHIC_PROMPT_SIZE
            ]
            datetime_cursor = DateTimeCursor(
                current_datetime=datetime.datetime(
                    year=int(year_token.get_name()), month=1, day=1
                )
            )
            vs_index = 0
            clinical_tokens = self.tokens[DEMOGRAPHIC_PROMPT_SIZE:]
            visits = []
            for token_index, token in enumerate(clinical_tokens):
                if token.type == TokenType.VS:
                    vs_index = token_index
                elif token.type == TokenType.VE:
                    visits.append(
                        self.process_visit_block(
                            visit_tokens=clinical_tokens[vs_index : token_index + 1],
                            datetime_cursor=datetime_cursor,
                            domain_map=domain_map,
                            concept_map=concept_map,
                        )
                    )
                elif token.type == TokenType.ATT:
                    datetime_cursor.add_days(extract_time_interval_in_days(token.name))

            birth_datetime = datetime.datetime(
                int(year_token.get_name()) - int(age_token.get_name()), 1, 1
            )
            return CehrGptPatient(
                patient_id=self.person_id,
                birth_datetime=birth_datetime,
                gender_concept_id=int(gender_token.name),
                gender=concept_map.get(gender_token.name, None),
                race_concept_id=int(race_token.get_name()),
                race=concept_map.get(race_token.name, None),
                visits=visits,
            )
        return None

    def get_error_messages(self) -> List[str]:
        return self.error_messages

    def validate(self) -> None:
        current_visit_token = None
        for i, token in enumerate(self.tokens):
            if token.type in [TokenType.INPATIENT_VISIT, TokenType.OUTPATIENT_VISIT]:
                current_visit_token = token
            if token.type == TokenType.VE:
                current_visit_token = None
            pre_token = self.tokens[i - 1] if i > 0 else None
            next_token = self.tokens[i + 1] if i < len(self.tokens) - 1 else None
            self._validate_token(token, pre_token, next_token, current_visit_token)

    def _validate_token(
        self,
        token: CEHRGPTToken,
        pre_token: Optional[CEHRGPTToken] = None,
        next_token: Optional[CEHRGPTToken] = None,
        current_visit_token: Optional[CEHRGPTToken] = None,
    ) -> None:
        is_validated = False
        for validation_rule in self.validation_rules:
            if validation_rule.is_required(token, current_visit_token):
                is_validated = True
                if not validation_rule.validate(token, pre_token, next_token):
                    self.error_messages.append(
                        validation_rule.get_validation_error_message(
                            token, pre_token, next_token
                        )
                    )
            if is_validated:
                break

    def process_visit_block(
        self,
        visit_tokens: List[CEHRGPTToken],
        datetime_cursor: DateTimeCursor,
        domain_map: Dict[str, str],
        concept_map: Dict[str, str],
    ) -> CehrGptVisit:
        visit_concept_id = 0
        discharge_to_concept_id = None
        visit_id = (
            self.id_generator.get_next_visit_occurrence_id()
            if self.id_generator
            else None
        )
        visit_start_datetime: Optional[datetime.datetime] = None
        visit_end_datetime: Optional[datetime.datetime] = None
        events = []
        for token in visit_tokens:
            if token.type == TokenType.VS:
                visit_start_datetime = datetime_cursor.current_datetime
            elif token.type == TokenType.VE:
                visit_end_datetime = datetime_cursor.current_datetime
            elif token.type in [TokenType.OUTPATIENT_VISIT, TokenType.INPATIENT_VISIT]:
                visit_concept_id = int(token.name)
            elif token.type == TokenType.VISIT_DISCHARGE:
                discharge_to_concept_id = int(token.name)
            elif token.type == TokenType.INPATIENT_ATT:
                datetime_cursor.add_days(extract_time_interval_in_days(token.name))
            elif token.type == TokenType.INPATIENT_HOUR:
                datetime_cursor.add_hours(extract_hours_from_hour_token(token.name))
            elif token.type in clinical_token_types or token.type == TokenType.DEATH:
                domain = domain_map.get(
                    token.name, "Death" if token.type == TokenType.DEATH else "Unknown"
                )
                record_id = (
                    self.id_generator.get_next_id_by_domain(domain)
                    if self.id_generator
                    else None
                )

                event = CehrGptEvent(
                    time=datetime_cursor.current_datetime,
                    code=token.name,
                    code_label=concept_map.get(token.name, None),
                    text_value=token.text_value,
                    numeric_value=token.numeric_value,
                    unit=token.unit,
                    visit_id=visit_id,
                    domain=domain,
                    record_id=record_id,
                )
                if event not in events:
                    events.append(event)

        if visit_start_datetime is None:
            visit_start_datetime = events[0].get("time")
        if visit_end_datetime is None:
            visit_start_datetime = events[-1].get("time")

        visit_type = concept_map.get(str(visit_concept_id), None)
        discharge_facility = concept_map.get(
            str(discharge_to_concept_id) if discharge_to_concept_id else None, None
        )
        return CehrGptVisit(
            patient_id=self.person_id,
            visit_id=visit_id,
            visit_type=visit_type,
            visit_concept_id=visit_concept_id,
            visit_start_datetime=visit_start_datetime,
            visit_end_datetime=visit_end_datetime,
            discharge_facility=discharge_facility,
            discharge_to_concept_id=discharge_to_concept_id,
            events=events,
        )


def get_cehrgpt_patient_converter(
    concept_ids: List[str],
    concept_domain_mapping: Dict[str, str],
    numeric_values: Optional[List[float]] = None,
    text_values: Optional[List[str]] = None,
    units: Optional[List[str]] = None,
) -> PatientSequenceConverter:
    cehrgpt_tokens = translate_to_cehrgpt_tokens(
        concept_ids=concept_ids,
        concept_domain_mapping=concept_domain_mapping,
        numeric_values=numeric_values,
        text_values=text_values,
        units=units,
    )
    return PatientSequenceConverter(tokens=cehrgpt_tokens)
