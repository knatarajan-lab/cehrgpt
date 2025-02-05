import datetime
from abc import ABC, abstractmethod

from cehrgpt.gpt_utils import is_visit_end, is_visit_start

from .typed_tokens import CEHRGPTToken, TokenType


class ValidationRule(ABC):
    @abstractmethod
    def is_required(self, token: CEHRGPTToken) -> bool:
        # function of index or value (e.g. check VS validation rule if index = 0)
        # if this is true, meaning it's always required no matter which index this token is
        pass

    @abstractmethod
    def validate(self, token: CEHRGPTToken) -> bool:
        pass

    @abstractmethod
    def get_validation_error_message(self, start_index: int, token: CEHRGPTToken):
        pass


class AttValidationRule(ValidationRule):
    def is_required(self, token: CEHRGPTToken):
        return True

    def validate(self, token: CEHRGPTToken):
        return token.type == TokenType.ATT

    def get_validation_error_message(self, start_index: int, token: CEHRGPTToken):
        return f"Token at index {start_index + token.index} is not valid ATT token: {token.name}"


class YearValidationRule(ValidationRule):
    def is_required(self, token: CEHRGPTToken):
        # The validation rule needs to be called only if it's the first token in the demographic block
        return token.index == 0

    def validate(self, token: CEHRGPTToken):
        return (
            token.type == TokenType.YEAR
            and 1900 < int(token.name) < datetime.date.today().year
        )

    def get_validation_error_message(self, start_index: int, token: CEHRGPTToken):
        return f"Token at index {start_index + token.index} is not valid YEAR token: {token.name}"


class AgeValidationRule(ValidationRule):
    def is_required(self, token: CEHRGPTToken):
        return token.index == 1

    def validate(self, token: CEHRGPTToken):
        return token.type == TokenType.AGE and 0 <= int(token.name) <= 100

    def get_validation_error_message(self, start_index: int, token: CEHRGPTToken):
        return f"Token at index {start_index + token.index} is not valid AGE token: {token.name}"


class GenderValidationRule(ValidationRule):
    def is_required(self, token: CEHRGPTToken):
        return token.index == 2

    def validate(self, token: CEHRGPTToken):
        return token.type == TokenType.GENDER

    def get_validation_error_message(self, start_index: int, token: CEHRGPTToken):
        return f"Token at index {start_index + token.index} is not valid GENDER token: {token.name}"


class RaceValidationRule(ValidationRule):
    def is_required(self, token: CEHRGPTToken):
        return token.index == 3

    def validate(self, token: CEHRGPTToken):
        return token.type == TokenType.RACE

    def get_validation_error_message(self, start_index: int, token: CEHRGPTToken):
        return f"Token at index {start_index + token.index} is not valid RACE token: {token.name}"


class VisitStartValidationRule(ValidationRule):
    def is_required(self, token: CEHRGPTToken):
        return token.index == 0

    def validate(self, token: CEHRGPTToken):
        return token.type == TokenType.VS

    def get_validation_error_message(self, start_index: int, token: CEHRGPTToken):
        return f"Token at index {start_index + token.index} is not valid VS token: {token.name}"


class VisitEndValidationRule(ValidationRule):
    def __init__(self, visit_block_length: int):
        self.token_length = visit_block_length

    def is_required(self, token: CEHRGPTToken):
        return token.index == self.token_length - 1

    def validate(self, token: CEHRGPTToken):
        return token.type == TokenType.VE

    def get_validation_error_message(self, start_index: int, token: CEHRGPTToken):
        return f"Token at index {start_index + token.index} is not valid VE token: {token.name}"


class VisitTypeValidationRule(ValidationRule):
    def is_required(self, token: CEHRGPTToken):
        return token.index == 1

    def validate(self, token: CEHRGPTToken):
        return token.type == TokenType.VISIT

    def get_validation_error_message(self, start_index: int, token: CEHRGPTToken):
        return f"Token at index {start_index + token.index} is not valid VISIT token: {token.name}"


class InpatientConceptValidationRule(ValidationRule):

    def __init__(self, token_length: int):
        self.token_length = token_length

    def is_required(self, token: CEHRGPTToken):
        return 1 < token.index < self.token_length - 2

    def validate(self, token: CEHRGPTToken):
        return token.type in [
            TokenType.CONDITION,
            TokenType.DRUG,
            TokenType.PROCEDURE,
            TokenType.MEASUREMENT,
            TokenType.DEATH,
        ]

    def get_validation_error_message(self, start_index: int, token: CEHRGPTToken):
        return f"Token at index {start_index + token.index} is not valid INPATIENT token: {token.name}"


class DischargeValidationRule(ValidationRule):
    def __init__(self, token_length: int):
        self.token_length = token_length

    def is_required(self, token: CEHRGPTToken):
        return token.index == self.token_length - 2

    def validate(self, token: CEHRGPTToken):
        return token.type == TokenType.VISIT_DISCHARGE

    def get_validation_error_message(self, start_index: int, token: CEHRGPTToken):
        return f"Token at index {start_index + token.index} is not valid INPATIENT DISCHARGE token: {token.name}"


class ConceptValidationRule(ValidationRule):

    def __init__(self, token_length: int):
        self.token_length = token_length

    def is_required(self, token: CEHRGPTToken):
        return 1 < token.index < self.token_length - 1

    def validate(self, token: CEHRGPTToken):
        return token.type in [
            TokenType.CONDITION,
            TokenType.DRUG,
            TokenType.PROCEDURE,
            TokenType.MEASUREMENT,
        ]

    def get_validation_error_message(self, start_index: int, token: CEHRGPTToken):
        return f"Token at index {start_index + token.index} is not valid CONCEPT token: {token.name}"
