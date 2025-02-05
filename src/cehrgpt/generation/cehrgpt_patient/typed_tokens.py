from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Union

from cehrgpt.gpt_utils import (
    generate_artificial_time_tokens,
    is_inpatient_att_token,
    is_inpatient_hour_token,
    is_seq_end,
    is_seq_start,
    is_visit_end,
    is_visit_start,
)
from cehrgpt.models.special_tokens import DISCHARGE_CONCEPT_LIST, OOV_CONCEPT_MAP

ATT_TIME_TOKENS = generate_artificial_time_tokens()


class TokenType(Enum):
    START = "START"
    END = "END"
    VS = "VS"
    VE = "VE"
    ATT = "ATT"
    INPATIENT_ATT = "INPATIENT_ATT"
    INPATIENT_HOUR = "INPATIENT_HOUR"
    YEAR = "YEAR"
    AGE = "AGE"
    GENDER = "GENDER"
    RACE = "RACE"
    VISIT = "VISIT"
    DRUG = "DRUG"
    CONDITION = "CONDITION"
    PROCEDURE = "PROCEDURE"
    MEASUREMENT = "MEASUREMENT"
    DEATH = "DEATH"
    INPATIENT_VISIT = "INPATIENT_VISIT"
    VISIT_DISCHARGE = "VISIT_DISCHARGE"
    UNKNOWN = "UNKNOWN"


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


def create_cehrgpt_token(
    token: str,
    token_index: int,
    domain_map: Dict[str, str],
    numeric_value: Optional[float] = None,
    text_value: Optional[str] = None,
    unit: Optional[str] = None,
) -> CEHRGPTToken:
    token_type = TokenType.UNKNOWN
    token_name = token
    if is_visit_start(token):
        token_type = TokenType.VS
    elif is_visit_end(token):
        token_type = TokenType.VE
    elif token in ATT_TIME_TOKENS:
        token_type = TokenType.ATT
    elif is_inpatient_att_token(token):
        token_type = TokenType.INPATIENT_ATT
    elif is_seq_start(token):
        token_type = TokenType.START
    elif is_seq_end(token):
        token_type = TokenType.END
    elif is_inpatient_hour_token(token):
        token_type = TokenType.INPATIENT_HOUR
    elif token in OOV_CONCEPT_MAP.keys():
        token_type = domain_map[token].upper()
    elif token in domain_map:
        token_type = domain_map[token]
    elif token in DISCHARGE_CONCEPT_LIST:
        token_type = TokenType.VISIT_DISCHARGE

    return CEHRGPTToken(
        name=token_name,
        type=token_type,
        index=token_index,
        numeric_value=numeric_value,
        text_value=text_value,
        unit=unit,
    )


#
# def generate_omop_concept_domain(concept_parquet) -> Dict[int, str]:
#     """
#     Generate a dictionary of concept_id to domain_id
#     :param concept_parquet: concept dataframe read from parquet file
#     :return: dictionary of concept_id to domain_id
#     """
#     domain_dict = {}
#     for i in concept_parquet.itertuples():
#         domain_dict[i.concept_id] = i.domain_id
#     return domain_dict
#
#
# def create_demo_typed_token(seq):
#     year_typed_token = TypedToken(value=seq[0].split(':')[1], type=TokenType.YEAR, index=0)
#     age_typed_token = TypedToken(value=seq[1].split(':')[1], type=TokenType.AGE, index=1)
#     gender_typed_token = TypedToken(value=seq[2], type=TokenType.GENDER, index=2)
#     race_typed_token = TypedToken(value=seq[3], type=TokenType.RACE, index=3)
#     return [year_typed_token, age_typed_token, gender_typed_token, race_typed_token]
#
#

# def transform_to_typed_tokens(patient_sequence_path, concept_path):
#     concept_df = pd.read_parquet(os.path.join(concept_path))
#     domain_map = generate_omop_concept_domain(concept_df)
#     pt_seq_concept_ids = pd.read_parquet(
#         os.path.join(patient_sequence_path), columns=['person_id', 'concept_ids'])
#     # drop start token if it exists
#     pt_seq_concept_ids['concept_ids'] = pt_seq_concept_ids['concept_ids'].apply(
#         lambda x: x[1:] if 'start' in x[0].lower else x)
#     pt_seq_concept_ids['typed_tokens'] = pt_seq_concept_ids['concept_ids'].apply(
#         lambda x: create_demo_typed_token(x).extend([convert_to_typed_token(tk, domain_map) for tk in x[4:]]))
#     return pt_seq_concept_ids
