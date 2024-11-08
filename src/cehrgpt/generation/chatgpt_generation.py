import os
from textwrap import dedent

import numpy as np
from jinja2 import BaseLoader, Environment
from openai import OpenAI
from pydantic import BaseModel

MODEL = "gpt-4o-2024-08-06"
TEMPLATE = """
You are a medical professional tasked with generating a synthetic patient sequence using the CEHR-GPT format, outlined as follows:

[year]: Represents the start year of the patient sequence.
[age]: Represents the start age of the patient sequence.
[gender]: Patient's gender, allowed values are "Male," "Female," and "Unknown."
[race]: Patient's race, allowed values are "White," "Black," "Asian," and "Unknown."
[VS]: Marks the start of a visit.
[VE]: Marks the end of a visit.
[VT]: Type of visit, with allowed values "9202" (outpatient), "9201" (inpatient), and "9203" (emergency room).
[C_i]: Clinical concept represented by an OMOP concept ID (could be a drug, condition, or procedure).
[ATT]: Artificial time tokens, representing time intervals in days (e.g., "D1," "D10").
[i-ATT]: Inpatient-specific artificial time tokens, representing intervals within inpatient stays (e.g., "i-D1"), these tokens should only appear in inpatient visits.
Each sequence can encompass multiple concepts within each visit and vary from one to ten visits, reflective of real-world clinical scenarios. All clinical concepts must correspond to valid OMOP IDs. The sequence must end on [VE]

Example of a sequence:

{
    "seq": ['year:2008', 'age:28', '8532', '8527', '[VS]', '9202', '4301351',
       '19078924', '35603428', '35603429', '40221381', '40223365',
       '4155151', '4239130', '42536500', '4294382', '2108974', '433736',
       '[VE]', 'D7', '[VS]', '9201', '43011850', '35603429', '35603600',
       '35605482', '40163870', '40169706', '40221381', '35603428',
       '19078921', '40244026', '948080', '1154615', '1593063', '4056973',
       '4155151', '4194550', '3047860', '35604843', '43011962', '4160730',
       'i-D1', '35604843', '40162587', '43011962', '433736', '948080',
       '0', '[VE]', 'D14', '[VS]', '9202', '4019497', '[VE]', 'D26', '[VS]',
       '1', '4019497', '[VE]', 'D198', '[VS]', '581477', '433736',
       '[VE]', 'D19', '[VS]', '581477', '194152', '320128', '40483287', '433736', '[VE]']
}

When creating the sequence, please use the demographic tokens {{ demographic_prompt }} to construct a realistic and medically plausible patient trajectory.
"""


class PatientSequence(BaseModel):
    seq: list[str]


if __name__ == "__main__":
    import argparse
    import uuid

    import pandas as pd
    from tqdm import tqdm

    parser = argparse.ArgumentParser("ChatGPT patient generation")
    parser.add_argument(
        "--demographic_data",
        dest="demographic_data",
        action="store",
        help="The path for your demographic_data",
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
    env = Environment(loader=BaseLoader())
    template = env.from_string(TEMPLATE)
    demographics = pd.read_parquet(args.demographic_data)

    for _ in tqdm(range(args.num_sequences)):
        demographic_tokens = str(demographics.sample(1).concept_ids.iloc[0].tolist())
        prompt = template.render(demographic_prompt=demographic_tokens)
        client = OpenAI(api_key=os.environ.get("OPEN_AI_KEY"))
        completion = client.beta.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a medical professional."},
                {"role": "user", "content": dedent(prompt)},
            ],
            response_format=PatientSequence,
        )
        patient_sequence = completion.choices[0].message.parsed.seq
        pd.DataFrame(
            [
                {
                    "concept_ids": patient_sequence,
                    "concept_values": np.zeros_like(patient_sequence),
                }
            ],
            columns=["concept_ids", "concept_values"],
        ).to_parquet(os.path.join(args.output_folder, f"{uuid.uuid4()}.parquet"))
