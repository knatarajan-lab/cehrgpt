import json
import logging
import os
import random
import uuid
from textwrap import dedent

import polars as pl
from cehrbert.runners.runner_util import load_parquet_as_dataset
from jinja2 import BaseLoader, Environment
from openai import OpenAI

from cehrgpt.generation.cehrgpt_patient.convert_patient_sequence import (
    get_cehrgpt_patient_converter,
)
from cehrgpt.generation.chatgpt.chatgpt_cehrgpt_narrative import (
    MODEL,
    Patient,
    datetime_converter,
    translate_to_cehrgpt_patient,
)
from cehrgpt.omop.vocab_utils import generate_concept_maps
from src.cehrgpt.generation.cehrgpt_patient.cehrgpt_patient_schema import CehrGptPatient

logger = logging.getLogger(__name__)

TEMPLATE = """
Instruction for Generating a Clinically Realistic Patient Timeline

Objective: Enhance an existing synthetic patient timeline to generate a high-quality, clinically feasible timeline that incorporates greater diversity while ensuring medical plausibility.
Requirements:
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
    - Organize clinical events by day of hospitalization

Additional Consideration:
While the input timeline may not encompass all potential visit types, conditions, drugs, and procedures, you are encouraged to introduce novel yet clinically plausible elements. Adjustments to the length and content of the patient timeline are permitted provided they maintain clinical plausibility.

Existing Example:
{{ query }}
"""

if __name__ == "__main__":
    import argparse
    from dataclasses import asdict

    from tqdm import tqdm

    parser = argparse.ArgumentParser(
        "ChatGPT patient generation using Synthea sequences"
    )
    parser.add_argument(
        "--synthea_input_folder",
        dest="synthea_input_folder",
        action="store",
        required=True,
    )
    parser.add_argument(
        "--vocabulary_dir",
        dest="vocabulary_dir",
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
    client = OpenAI(api_key=os.environ.get("OPEN_AI_KEY"))
    dataset = load_parquet_as_dataset(args.synthea_input_folder)
    concept = pl.read_parquet(os.path.join(args.vocabulary_dir, "concept", "*.parquet"))
    concept_name_map, concept_domain_map = generate_concept_maps(concept)
    dataset_size = len(dataset)
    for _ in tqdm(range(args.num_sequences)):
        try:
            randint = random.randint(0, dataset_size - 1)
            record = dataset[randint]
            patient_converter = get_cehrgpt_patient_converter(
                record["concept_ids"],
                concept_domain_map,
            )
            if patient_converter.is_validation_passed:
                synthea_cehrgpt_patient = patient_converter.get_patient(
                    domain_map=concept_domain_map,
                    concept_map=concept_name_map,
                )
                synthea_patient_narrative = synthea_cehrgpt_patient.get_narrative()
                prompt = template.render(query=synthea_patient_narrative)
                completion = client.beta.chat.completions.parse(
                    model=MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a medical professional.",
                        },
                        {"role": "user", "content": dedent(prompt)},
                    ],
                    response_format=Patient,
                )
                cehrgpt_patient = translate_to_cehrgpt_patient(
                    completion.choices[0].message.parsed
                )
                # Adding the patient_id so we can check against the original patient sequence
                cehrgpt_patient = CehrGptPatient(
                    birth_datetime=cehrgpt_patient.birth_datetime,
                    gender_concept_id=cehrgpt_patient.gender_concept_id,
                    gender=cehrgpt_patient.gender,
                    race_concept_id=cehrgpt_patient.race_concept_id,
                    race=cehrgpt_patient.race,
                    patient_id=record["person_id"],
                    visits=cehrgpt_patient.visits,
                )
                with open(
                    os.path.join(args.output_folder, f"{uuid.uuid4()}.json"), "w"
                ) as f:
                    json.dump(asdict(cehrgpt_patient), f, default=datetime_converter)
        except Exception as e:
            logger.error(e)
