from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Union

from cehrgpt.gpt_utils import (
    is_age_token,
    is_death_token,
    is_discharge_type_token,
    is_gender_token,
    is_inpatient_att_token,
    is_inpatient_hour_token,
    is_inpatient_visit_type_token,
    is_outpatient_visit_type_token,
    is_race_token,
    is_seq_end,
    is_seq_start,
    is_visit_att_tokens,
    is_visit_end,
    is_visit_start,
    is_year_token,
)
from cehrgpt.models.special_tokens import OOV_CONCEPT_MAP


class TokenType(Enum):
    START = auto()
    END = auto()
    VS = auto()
    VE = auto()
    ATT = auto()
    INPATIENT_ATT = auto()
    INPATIENT_HOUR = auto()
    YEAR = auto()
    AGE = auto()
    GENDER = auto()
    RACE = auto()
    OUTPATIENT_VISIT = auto()
    DRUG = auto()
    CONDITION = auto()
    PROCEDURE = auto()
    MEASUREMENT = auto()
    OBSERVATION = auto()
    DEVICE = auto()
    DEATH = auto()
    INPATIENT_VISIT = auto()
    VISIT_DISCHARGE = auto()
    UNKNOWN = auto()

    @classmethod
    def from_value(cls, value):
        """Factory method to return an enum member based on the given value or UNKNOWN if not found."""
        for name, member in cls.__members__.items():
            if name == value:
                return member
        return cls.UNKNOWN  # Default case if value not found


@dataclass
class CEHRGPTToken:
    name: str
    type: TokenType
    index: int
    numeric_value: Optional[float] = None
    text_value: Optional[str] = None
    unit: Optional[str] = None

    def get_name(self) -> Union[int, str]:
        if self.type in [TokenType.YEAR, TokenType.AGE]:
            return int(self.name.split(":")[1])
        return self.name


def make_cehrgpt_token(
    token: str,
    token_index: int,
    domain_map: Dict[str, str],
    numeric_value: Optional[float] = None,
    text_value: Optional[str] = None,
    unit: Optional[str] = None,
    prev_token: Optional[str] = None,
) -> CEHRGPTToken:
    """
    Creates a CEHRGPTToken instance with specific attributes based on the given input token and its properties.

    The function determines the token's type using predefined criteria and mappings, including visit markers,
    attribute time tokens, sequence markers, and domain-specific mappings. Additional attributes like numeric and
    text values, and units can be optionally included.

    Parameters:
        token (str): The token string that is to be analyzed and classified.
        token_index (int): The index of the token in the sequence where it appears.
        domain_map (Dict[str, str]): A dictionary mapping tokens to their domain-specific types.
        numeric_value (Optional[float]): An optional numeric value associated with the token, default is None.
        text_value (Optional[str]): An optional text value associated with the token, default is None.
        unit (Optional[str]): An optional unit of measure associated with the token, default is None.
        prev_token (Optional[str]):

    Returns:
        CEHRGPTToken: An instance of CEHRGPTToken that contains the token's name, type, index, and optionally,
                      its numeric value, text value, and unit.

    Raises:
        ValueError: If the token does not match any known type and is not in the provided domain_map.

    Example:
        >>> make_cehrgpt_token("blood_pressure", 1, {"blood_pressure": "measurement"}, 120.0, "High", "mmHg")
        CEHRGPTToken(name='blood_pressure', type='measurement', index=1, numeric_value=120.0, text_value='High', unit='mmHg')
    """
    if is_seq_start(token):
        token_type = TokenType.START
    elif is_seq_end(token):
        token_type = TokenType.END
    elif is_year_token(token):
        token_type = TokenType.YEAR
    elif is_age_token(token):
        token_type = TokenType.AGE
    elif is_race_token(token) and token_index == 3:
        token_type = TokenType.RACE
    elif is_gender_token(token) and token_index == 2:
        token_type = TokenType.GENDER
    elif is_visit_start(token):
        token_type = TokenType.VS
    elif is_visit_end(token):
        token_type = TokenType.VE
    # TODO: it's important to put discharge before inpatient_visit because they share concept ids
    elif is_inpatient_visit_type_token(token) and is_visit_start(prev_token):
        token_type = TokenType.INPATIENT_VISIT
    elif is_discharge_type_token(token):
        token_type = TokenType.VISIT_DISCHARGE
    elif is_outpatient_visit_type_token(token):
        token_type = TokenType.OUTPATIENT_VISIT
    elif is_visit_att_tokens(token):
        token_type = TokenType.ATT
    elif is_inpatient_att_token(token):
        token_type = TokenType.INPATIENT_ATT
    elif is_inpatient_hour_token(token):
        token_type = TokenType.INPATIENT_HOUR
    elif is_death_token(token):
        token_type = TokenType.DEATH
    elif token in domain_map:
        token_type = TokenType.from_value(domain_map[token].upper())
    elif token in OOV_CONCEPT_MAP:
        token_type = TokenType.from_value(OOV_CONCEPT_MAP[token].upper())
    else:
        token_type = TokenType.UNKNOWN

    return CEHRGPTToken(
        name=token,
        type=token_type,
        index=token_index,
        numeric_value=numeric_value,
        text_value=text_value,
        unit=unit,
    )


def translate_to_cehrgpt_tokens(
    concept_ids: List[str],
    concept_domain_mapping: Dict[str, str],
    numeric_values: Optional[List[float]] = None,
    text_values: Optional[List[float]] = None,
    units: Optional[List[str]] = None,
) -> List[CEHRGPTToken]:
    cehrgpt_tokens: List[CEHRGPTToken] = []
    for event_index, event in enumerate(concept_ids):
        prev_event = concept_ids[event_index - 1] if event_index > 0 else None
        numeric_value = (
            numeric_values[event_index] if numeric_values is not None else None
        )
        text_value = text_values[event_index] if text_values is not None else None
        unit = units[event_index] if units is not None else None
        cehrgpt_tokens.append(
            make_cehrgpt_token(
                event,
                event_index,
                concept_domain_mapping,
                numeric_value,
                text_value,
                unit,
                prev_event,
            )
        )
    return cehrgpt_tokens
