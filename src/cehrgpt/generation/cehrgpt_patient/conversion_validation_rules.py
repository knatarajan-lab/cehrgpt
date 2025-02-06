import datetime
from abc import ABC, abstractmethod
from typing import List, Optional

from .typed_tokens import CEHRGPTToken, TokenType

clinical_token_types = [
    TokenType.CONDITION,
    TokenType.PROCEDURE,
    TokenType.DRUG,
    TokenType.MEASUREMENT,
]


class ValidationRule(ABC):
    @abstractmethod
    def is_required(
        self,
        token: CEHRGPTToken,
        current_visit_type: Optional[CEHRGPTToken] = None,
    ) -> bool:
        # function of index or value (e.g. check VS validation rule if index = 0)
        # if this is true, meaning it's always required no matter which index this token has
        pass

    @abstractmethod
    def validate(
        self,
        token: CEHRGPTToken,
        pre_token: Optional[CEHRGPTToken] = None,
        next_token: Optional[CEHRGPTToken] = None,
    ) -> bool:
        pass

    @staticmethod
    def get_validation_error_message(
        token: CEHRGPTToken,
        pre_token: Optional[CEHRGPTToken] = None,
        next_token: Optional[CEHRGPTToken] = None,
    ):
        return (
            f"Token at index {token.index} is not valid {token.type.name} token: {token.name} between "
            f"previous token {pre_token.type.name if pre_token else None} {pre_token.name if pre_token else None} "
            f"at {pre_token.index if pre_token else None} "
            f"and next token {next_token.type.name if next_token else None} {next_token.name if next_token else None} "
            f"at {next_token.index if next_token else None}"
        )


class YearValidationRule(ValidationRule):
    def is_required(
        self,
        token: CEHRGPTToken,
        current_visit_type: Optional[CEHRGPTToken] = None,
    ):
        # The validation rule needs to be called only if it's the first token in the demographic block
        return token.index == 0

    def validate(
        self,
        token: CEHRGPTToken,
        pre_token: Optional[CEHRGPTToken] = None,
        next_token: Optional[CEHRGPTToken] = None,
    ):
        return (
            token.type == TokenType.YEAR
            and 1900 < int(token.get_name()) < datetime.date.today().year
        )


class AgeValidationRule(ValidationRule):
    def is_required(
        self,
        token: CEHRGPTToken,
        current_visit_type: Optional[CEHRGPTToken] = None,
    ):
        return token.index == 1

    def validate(
        self,
        token: CEHRGPTToken,
        pre_token: Optional[CEHRGPTToken] = None,
        next_token: Optional[CEHRGPTToken] = None,
    ):
        return token.type == TokenType.AGE and 0 <= int(token.get_name()) <= 100


class GenderValidationRule(ValidationRule):
    def is_required(
        self,
        token: CEHRGPTToken,
        current_visit_type: Optional[CEHRGPTToken] = None,
    ):
        return token.index == 2

    def validate(
        self,
        token: CEHRGPTToken,
        pre_token: Optional[CEHRGPTToken] = None,
        next_token: Optional[CEHRGPTToken] = None,
    ):
        return token.type == TokenType.GENDER


class RaceValidationRule(ValidationRule):
    def is_required(
        self,
        token: CEHRGPTToken,
        current_visit_type: Optional[CEHRGPTToken] = None,
    ):
        return token.index == 3

    def validate(
        self,
        token: CEHRGPTToken,
        pre_token: Optional[CEHRGPTToken] = None,
        next_token: Optional[CEHRGPTToken] = None,
    ):
        return token.type == TokenType.RACE


class VisitStartValidationRule(ValidationRule):
    def is_required(
        self,
        token: CEHRGPTToken,
        current_visit_type: Optional[CEHRGPTToken] = None,
    ):
        return token.type == TokenType.VS

    def validate(
        self,
        token: CEHRGPTToken,
        pre_token: Optional[CEHRGPTToken] = None,
        next_token: Optional[CEHRGPTToken] = None,
    ):
        if pre_token and pre_token.type not in [
            TokenType.ATT,
            TokenType.RACE,
            TokenType.GENDER,
        ]:
            return False
        if not next_token:
            return False
        # Enforce the patterns: [ATT or GENDER or RACE] [VS] [VT or DEATH]
        next_token_validation = next_token.type in [
            TokenType.OUTPATIENT_VISIT,
            TokenType.INPATIENT_VISIT,
            TokenType.DEATH,
        ]
        return token.type == TokenType.VS and next_token_validation


class VisitEndValidationRule(ValidationRule):
    def is_required(
        self,
        token: CEHRGPTToken,
        current_visit_type: Optional[CEHRGPTToken] = None,
    ):
        return token.type == TokenType.VE

    def validate(
        self,
        token: CEHRGPTToken,
        pre_token: Optional[CEHRGPTToken] = None,
        next_token: Optional[CEHRGPTToken] = None,
    ):
        if not pre_token:
            return False

        # If DEATH token occurs before [VE], there can not be any events after [VE]
        if pre_token.type == TokenType.DEATH and next_token:
            return False

        # Enforce the patterns: 1) [clinical_event or discharge_event] [VE] [ATT]; 2) [DEATH] [VE]
        if next_token and next_token.type != TokenType.ATT:
            return False

        pre_token_validation = (
            pre_token.type in clinical_token_types
            or pre_token.type in [TokenType.VISIT_DISCHARGE, TokenType.DEATH]
        )
        return pre_token_validation and token.type == TokenType.VE


class VisitTypeValidationRule(ValidationRule):
    def is_required(
        self,
        token: CEHRGPTToken,
        current_visit_type: Optional[CEHRGPTToken] = None,
    ):
        return token.type in [TokenType.OUTPATIENT_VISIT, TokenType.INPATIENT_VISIT]

    def validate(
        self,
        token: CEHRGPTToken,
        pre_token: Optional[CEHRGPTToken] = None,
        next_token: Optional[CEHRGPTToken] = None,
    ):
        if not pre_token or not next_token:
            return False

        # Patterns: [VS] [VT] [clinical_event or i-D1 or i-h1]
        token_validation = token.type in [
            TokenType.OUTPATIENT_VISIT,
            TokenType.INPATIENT_VISIT,
        ]
        next_token_validation = (
            next_token.type in clinical_token_types
            or next_token.type in [TokenType.INPATIENT_HOUR, TokenType.INPATIENT_ATT]
        )
        return (
            pre_token.type == TokenType.VS
            and token_validation
            and next_token_validation
        )


class AttValidationRule(ValidationRule):
    def is_required(
        self,
        token: CEHRGPTToken,
        current_visit_type: Optional[CEHRGPTToken] = None,
    ):
        return token.type == TokenType.ATT

    def validate(
        self,
        token: CEHRGPTToken,
        pre_token: Optional[CEHRGPTToken] = None,
        next_token: Optional[CEHRGPTToken] = None,
    ) -> bool:
        if not pre_token or not next_token:
            return False
        # Enforce the pattern [VE] [ATT] [VS]
        return (
            pre_token.type == TokenType.VE
            and token.type == TokenType.ATT
            and next_token.type == TokenType.VS
        )


class InpatientAttValidationRule(ValidationRule):
    def is_required(
        self,
        token: CEHRGPTToken,
        current_visit_type: Optional[CEHRGPTToken] = None,
    ):
        return token.type == TokenType.INPATIENT_ATT

    def validate(
        self,
        token: CEHRGPTToken,
        pre_token: Optional[CEHRGPTToken] = None,
        next_token: Optional[CEHRGPTToken] = None,
    ):
        if not pre_token or not next_token:
            return False

        # Enforce the patterns: [clinical_event or inpatient_visit] [i-D1] [i-H1 or clinical_event or discharge]
        pre_token_validation = (
            pre_token.type in clinical_token_types
            or pre_token.type == TokenType.INPATIENT_VISIT
        )
        next_token_validation = (
            next_token.type == TokenType.INPATIENT_HOUR
            or next_token.type in clinical_token_types
            or next_token.type in [TokenType.VISIT_DISCHARGE]
        )
        return (
            pre_token_validation
            and token.type == TokenType.INPATIENT_ATT
            and next_token_validation
        )


class InpatientConceptValidationRule(ValidationRule):
    def is_required(
        self,
        token: CEHRGPTToken,
        current_visit_type: Optional[CEHRGPTToken] = None,
    ):
        # We skip the other artificial tokens, all other tokens need to be validated
        return token.type not in [
            TokenType.VS,
            TokenType.VE,
            TokenType.OUTPATIENT_VISIT,
            TokenType.INPATIENT_VISIT,
            TokenType.INPATIENT_ATT,
            TokenType.INPATIENT_HOUR,
            TokenType.VISIT_DISCHARGE,
        ] and (
            current_visit_type and current_visit_type.type == TokenType.INPATIENT_VISIT
        )

    def validate(
        self,
        token: CEHRGPTToken,
        pre_token: Optional[CEHRGPTToken] = None,
        next_token: Optional[CEHRGPTToken] = None,
    ):
        if not pre_token or not next_token:
            return False

        # Patterns: [i-D1 or i-H1 or clinical_event] [clinical_event] [clinical_event or i-D1 or i-H1 or discharge]
        pre_token_validation = (
            pre_token.type in clinical_token_types
            or pre_token.type
            in [
                TokenType.INPATIENT_HOUR,
                TokenType.INPATIENT_ATT,
                TokenType.INPATIENT_VISIT,
            ]
        )
        token_validation = token.type in clinical_token_types
        next_token_validation = (
            next_token.type in clinical_token_types
            or next_token.type
            in [
                TokenType.INPATIENT_HOUR,
                TokenType.INPATIENT_ATT,
                TokenType.VISIT_DISCHARGE,
            ]
        )
        return pre_token_validation and token_validation and next_token_validation


class DischargeValidationRule(ValidationRule):
    def is_required(
        self,
        token: CEHRGPTToken,
        current_visit_type: Optional[CEHRGPTToken] = None,
    ):
        return token.type == TokenType.VISIT_DISCHARGE

    def validate(
        self,
        token: CEHRGPTToken,
        pre_token: Optional[CEHRGPTToken] = None,
        next_token: Optional[CEHRGPTToken] = None,
    ):
        if not pre_token or not next_token:
            return False
        # Patterns: [clinical_event or i-D1 or i-H1] [discharge] [VE]
        pre_token_validation = (
            pre_token.type in clinical_token_types
            or pre_token.type in [TokenType.INPATIENT_HOUR, TokenType.INPATIENT_ATT]
        )
        return (
            pre_token_validation
            and token.type == TokenType.VISIT_DISCHARGE
            and next_token.type == TokenType.VE
        )


class ConceptValidationRule(ValidationRule):
    def is_required(
        self,
        token: CEHRGPTToken,
        current_visit_type: Optional[CEHRGPTToken] = None,
    ):
        # We skip the other artificial tokens, all other tokens need to be validated
        return token.type not in [
            TokenType.VS,
            TokenType.VE,
            TokenType.OUTPATIENT_VISIT,
        ] and (
            current_visit_type and current_visit_type.type == TokenType.OUTPATIENT_VISIT
        )

    def validate(
        self,
        token: CEHRGPTToken,
        pre_token: Optional[CEHRGPTToken] = None,
        next_token: Optional[CEHRGPTToken] = None,
    ):
        if not pre_token or not next_token:
            return False
            # Patterns: [clinical_event or visit_type] [clinical_event] [clinical_event or VE]
        pre_token_validation = (
            pre_token.type in clinical_token_types
            or pre_token.type == TokenType.OUTPATIENT_VISIT
        )
        next_token_validation = (
            next_token.type in clinical_token_types or next_token.type == TokenType.VE
        )
        return (
            pre_token_validation
            and token.type in clinical_token_types
            and next_token_validation
        )


class DeathValidationRule(ValidationRule):
    def is_required(
        self,
        token: CEHRGPTToken,
        current_visit_type: Optional[CEHRGPTToken] = None,
    ):
        # We skip the other artificial tokens, all other tokens need to be validated
        return token.type == TokenType.DEATH

    def validate(
        self,
        token: CEHRGPTToken,
        pre_token: Optional[CEHRGPTToken] = None,
        next_token: Optional[CEHRGPTToken] = None,
    ):
        if not pre_token or not next_token:
            return False
        # Pattern: [VS][DEATH][VE]
        return (
            pre_token.type == TokenType.VS
            and token.type == TokenType.DEATH
            and next_token.type == TokenType.VE
        )


def get_validation_rules() -> List[ValidationRule]:
    """Recursively find all subclasses of a given class."""
    return [_() for _ in ValidationRule.__subclasses__()]
