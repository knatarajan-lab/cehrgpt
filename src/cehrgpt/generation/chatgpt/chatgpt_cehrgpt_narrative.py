import json
import os
from datetime import datetime
from textwrap import dedent
from typing import List, Optional

from jinja2 import BaseLoader, Environment
from openai import OpenAI
from pydantic import BaseModel

from cehrgpt.generation.cehrgpt_patient.cehrgpt_patient_schema import (
    CehrGptEvent,
    CehrGptPatient,
    CehrGptVisit,
)

MODEL = "gpt-4o-2024-08-06"
TEMPLATE = """
Instruction for Generating a Clinically Realistic Patient Timeline

You are tasked with generating a high-quality synthetic patient timeline that accurately represents realistic {{ query }}.
The timeline must follow a structured format to ensure clinical plausibility.

Requirements
1. Demographics: Start the sequence with essential demographics including the patient's date of birth, gender, and race.
    - Date of birth should be in the format of %Y-%m-%dT%H:%M:%S
    - Race Type:
        * No matching concept
        * White
        * Black or African American
        * Other Race
        * Asian
        * American Indian or Alaska Native
        * Native Hawaiian or Other Pacific Islander
    - Gender Type:
        * FEMALE
        * MALE
        * No matching concept
2. Visits: Incorporate a series of medical visits characterized by:
    - Visit Type: Choose from the following options:
        * Inpatient Visit
        * Outpatient Visit
        * Emergency Room Visit
        * Combined Emergency Room and Inpatient Visit
        * Office Visit
        * Ambulatory Radiology Clinic/Center
        * Telehealth
    - Timing: Each visit should specify a start date and time (%Y-%m-%dT%H:%M:%S). For inpatient visits, include an end date and time.
3. Events Within Each Visit: Document clinical events such as conditions, medications, and procedures during each visit:
    - Conditions (Diagnoses)
        * Return condition date_time, concept name, concept_code, and the vocabulary name
        * Allowed Vocabulary:
            * SNOMED
            * ICD10
    - Drugs (Prescriptions/Administrations)
        * Return drug date_time, concept name, concept_code, and the vocabulary name
        * Allowed Vocabulary:
            * RXNORM
            * CVX
    - Procedures
        * Return procedure date_time, concept name, concept_code, and the vocabulary name
        * Allowed Vocabulary:
            * CPT4
            * ICD10
            * HCPCS
4. Inpatient Visit Specifics:
    - Organize clinical events by day of hospitalization, using a "Day 0" for admission and subsequent days numerically.

Example Output (Illustration Only):
Do not simply mimic this example. Generate unique, clinically realistic data.

Patient Demographics:
    Gender: MALE
    Race: White

Outpatient Visit on 2008-01-30 (Age: 47)
    Condition:
       * concept_name: Coronary arteriosclerosis; concept_code: ICD10//I25.1
       * concept_name: Tobacco dependence syndrome; concept_code: ICD10//F17.200
    Drug:
       * concept_name: Ibuprofen 400 MG Oral Tablet; concept_code: RXNORM//5640

Emergency Room and Inpatient Visit on 2008-02-08 (Age: 47)
    Visit End Date: 2008-02-10

    On Day 0 on 2008-02-08:
        Condition:
           * concept_name: Coronary arteriosclerosis; concept_code: ICD10//I25.1
        Drug:
           * concept_name: 10 ML potassium chloride 2 MEQ/ML Injection; concept_code: RXNORM//308057
        Procedure:
           * concept_name: Intravascular imaging of coronary vessels; concept_code: ICD10//B245ZZZ

    On Day 1 on 2008-02-09:
        Drug:
           * concept_name: Acetaminophen 325 MG Oral Tablet [Tylenol]; concept_code: RXNORM//161
        Procedure:
           * concept_name: Electrocardiogram, routine ECG with at least 12 leads; concept_code: ICD10//4A02114

    On Day 2 on 2008-02-10:
        Condition:
           * concept_name: Coronary arteriosclerosis; concept_code: ICD10//I25.1
        Drug:
           * concept_name: Nitroglycerin 0.3 MG Sublingual Tablet; concept_code: RXNORM//5640
        Procedure:
           * concept_name: Electrocardiogram, routine ECG with at least 12 leads; concept_code: ICD10//4A02114

Outpatient Visit on 2008-03-08 (Age: 47)
    Condition:
       * concept_name: Type 2 Diabetes Mellitus; concept_code: ICD10//E11.9
    Drug:
       * concept_name: Metformin 500 MG Oral Tablet; concept_code: RXNORM//860975
"""


class Event(BaseModel):
    concept_name: str
    concept_code: str
    vocabulary_name: str
    domain: str
    date_time: str


class Visit(BaseModel):
    visit_start_datetime: str
    visit_end_datetime: Optional[str]
    visit_type: str
    events: List[Event]


class Patient(BaseModel):
    gender: str
    race: str
    birth_datetime: str
    visits: List[Visit]


def parse_datetime(date_str):
    """Parses date strings, handling both date-only and full ISO 8601 formats."""
    if date_str.endswith("T24:00:00"):
        # Convert 'YYYY-MM-DDT24:00:00' → 'YYYY-MM-DD + 1 day T00:00:00'
        date_str = date_str.split("T")[0]
    try:
        # Handle full ISO 8601 format, including 'Z' (UTC)
        return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        try:
            # Handle ISO format without 'Z'
            return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            # Fallback for date-only format
            return datetime.strptime(date_str, "%Y-%m-%d")


def datetime_converter(obj):
    """Convert datetime objects to ISO 8601 string format."""
    if isinstance(obj, datetime):
        return obj.isoformat()  # Example: '2024-02-13T09:00:00'
    raise TypeError(f"Type {type(obj)} not serializable")


def map_gender_to_gender_concept_id(gender: str) -> int:
    gender_concept_id = 0
    gender_lowercase = gender.lower()
    if gender_lowercase == "female":
        gender_concept_id = 8532
    elif gender_lowercase == "male":
        gender_concept_id = 8507
    return gender_concept_id


def map_race_race_concept_id(race: str) -> int:
    race_concept_id = 0
    race_lowercase = race.lower()
    if race_lowercase == "white":
        race_concept_id = 8527
    elif race_lowercase == "black or african american":
        race_concept_id = 8516
    elif race_lowercase == "unknown":
        race_concept_id = 8552
    elif race_lowercase == "other race":
        race_concept_id = 8522
    elif race_lowercase == "Asian":
        race_concept_id = 8515
    elif race_lowercase == "american indian or alaska native":
        race_concept_id = 8657
    elif race_lowercase == "native hawaiian or other pacific islander":
        race_concept_id = 8557
    return race_concept_id


def map_visit_type_to_visit_concept_id(visit_type: str) -> int:
    visit_concept_id = 0
    visit_type_lowercase = visit_type.lower()
    if visit_type_lowercase == "inpatient visit":
        visit_concept_id = 9201
    elif visit_type_lowercase == "outpatient visit":
        visit_concept_id = 9202
    elif visit_type_lowercase == "emergency room visit":
        visit_concept_id = 9203
    elif visit_type_lowercase == "emergency room and inpatient Visit":
        visit_concept_id = 262
    elif visit_type_lowercase == "office visit":
        visit_concept_id = 581477
    elif visit_type_lowercase == "ambulatory radiology clinic / center":
        visit_concept_id = 38004250
    elif visit_type_lowercase == "telehealth":
        visit_concept_id = 5083
    return visit_concept_id


def translate_to_cehrgpt_patient(generated_patient: Patient) -> CehrGptPatient:
    cehrgpt_visits: List[CehrGptVisit] = []
    for visit in generated_patient.visits:
        visit_concept_id = map_visit_type_to_visit_concept_id(visit.visit_type)
        visit_start_datetime = parse_datetime(visit.visit_start_datetime)
        visit_end_datetime = (
            parse_datetime(visit.visit_end_datetime)
            if visit.visit_end_datetime
            else None
        )
        events: List[CehrGptEvent] = []
        for event in visit.events:
            try:
                event_time = parse_datetime(event.date_time)
                events.append(
                    CehrGptEvent(
                        time=event_time,
                        code=f"{event.vocabulary_name}//{event.concept_code}",
                        domain=event.domain,
                        code_label=event.concept_name,
                    )
                )
            except ValueError:
                print(f"Invalid event.date_time: {event}")

        cehrgpt_visits.append(
            CehrGptVisit(
                visit_concept_id=visit_concept_id,
                visit_type=visit.visit_type,
                visit_start_datetime=visit_start_datetime,
                visit_end_datetime=visit_end_datetime,
                events=events,
            )
        )

    return CehrGptPatient(
        gender=generated_patient.gender,
        race=generated_patient.race,
        birth_datetime=parse_datetime(generated_patient.birth_datetime),
        gender_concept_id=map_gender_to_gender_concept_id(generated_patient.gender),
        race_concept_id=map_race_race_concept_id(generated_patient.race),
        visits=cehrgpt_visits,
    )


if __name__ == "__main__":
    import argparse
    from dataclasses import asdict

    from tqdm import tqdm

    parser = argparse.ArgumentParser(
        "ChatGPT patient generation using cehrgpt narrative"
    )
    parser.add_argument(
        "--narrative",
        dest="narrative",
        action="store",
        required=True,
    )
    parser.add_argument(
        "--output_folder",
        dest="output_folder",
        action="store",
        help="The path for your output_folder",
        required=True,
    )
    parser.add_argument(
        "--num_sequences",
        dest="num_sequences",
        action="store",
        type=int,
        help="The path for your output_folder",
        required=True,
    )
    args = parser.parse_args()
    # Create a Jinja2 environment and render the template
    env = Environment(loader=BaseLoader(), autoescape=True)
    template = env.from_string(TEMPLATE)
    prompt = template.render(query=args.narrative)
    client = OpenAI(api_key=os.environ.get("OPEN_AI_KEY"))
    for i, _ in tqdm(enumerate(range(args.num_sequences))):
        completion = client.beta.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a medical professional."},
                {"role": "user", "content": dedent(prompt)},
            ],
            response_format=Patient,
        )
        cehrgpt_patient = translate_to_cehrgpt_patient(
            completion.choices[0].message.parsed
        )
        with open(os.path.join(args.output_folder, f"{i + 1}.json"), "w") as f:
            json.dump(asdict(cehrgpt_patient), f, default=datetime_converter)
