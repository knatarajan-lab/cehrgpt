import os
from textwrap import dedent
from typing import Optional, Tuple

from jinja2 import BaseLoader, Environment
from openai import OpenAI
from pydantic import BaseModel

from cehrgpt.generation.cehrgpt_patient.clinical_statement_generator import (
    create_clinical_statement,
)

MODEL = "gpt-4o-2024-08-06"
TEMPLATE = """
Instructions: Analyze the clinical statement provided and extract the following information for synthetic data generation: number of patients, race, gender, diagnosis,
age at diagnosis, and the medication prescribed. If any information is not available in the statement,
leave the corresponding field blank.

Additional requirements:
1. Here are the allowed values are for race
- White
- Asian
- Black or African American
- Middle Eastern or North African
- Native Hawaiian or Other Pacific Islander

2. Here are the allowed values are for gender
- Female
- Male

3. When you extract diagnoses, convert the diagnosis to the standard name
4. When you extract drugs, convert the drug to their standard ingredients

Clinical Statement: {{ clinical_statement }}

Example Responses:
Example 1:
Statement: A white male was diagnosed with essential hypertension at the age of 50 and he was being treated with ACE inhibitors.
Expected Answer:
Number of patients: 1
Race: White
Gender: Male
Age of diagnosis: 50
Diagnosis: Essential Hypertension
Drug: ACE Inhibitors

Example 2:
Statement: 10 female patients who were diagnosed with T2DM at the age of 40.
Expected Answer:
Number of patients: 10
Race:
Gender: Female
Age of diagnosis: 40
Diagnosis: T2DM
Drug:
"""

ENV = Environment(loader=BaseLoader(), autoescape=True)
QUERY_TEMPLATE = ENV.from_string(TEMPLATE)


class InstructCehrGptQueryTemplate(BaseModel):
    gender: str
    race: str
    age_of_diagnosis: int
    diagnosis: str
    drug: str
    num_of_patients: int = 1


def parse_question_to_cehrgpt_query(
    clinical_statement: str,
) -> Optional[Tuple[str, int]]:
    client = OpenAI(api_key=os.environ.get("OPEN_AI_KEY", None))
    prompt = QUERY_TEMPLATE.render(clinical_statement=clinical_statement)
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a medical professional."},
            {"role": "user", "content": dedent(prompt)},
        ],
        response_format=InstructCehrGptQueryTemplate,
    )
    parsed = completion.choices[0].message.parsed
    if parsed:
        race = parsed.race
        gender = parsed.gender
        age_of_diagnosis = parsed.age_of_diagnosis
        diagnosis = parsed.diagnosis
        drug = parsed.drug
        num_of_patients = parsed.num_of_patients
        if not race and not gender and not diagnosis:
            return None
        # Default to one patient if LLM fails to identify this
        if not num_of_patients:
            num_of_patients = 1

        clinical_statement = create_clinical_statement(
            gender=gender,
            race=race,
            age_condition_drug_tuples=[(age_of_diagnosis, diagnosis, drug)],
        )
        return clinical_statement, num_of_patients
    return None
