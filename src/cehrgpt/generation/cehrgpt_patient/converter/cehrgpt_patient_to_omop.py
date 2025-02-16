import argparse
import glob
import json
import logging
import os
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd
import polars as pl
from tqdm import tqdm

from cehrgpt.generation.cehrgpt_patient.cehrgpt_patient_schema import (
    CehrGptPatient,
    load_cehrgpt_patient_from_json,
)
from cehrgpt.generation.cehrgpt_patient.convert_patient_sequence import IdGenerator
from cehrgpt.generation.omop_entity import (
    ConditionOccurrence,
    DrugExposure,
    OmopEntity,
    Person,
    ProcedureOccurrence,
    VisitOccurrence,
)
from cehrgpt.omop.vocab_utils import (
    generate_to_standard_concept_id_map,
    generate_vocabulary_concept_to_concept_id_map,
)

logger = logging.getLogger(__name__)


def convert_cehrgpt_to_omop_events(
    cehrgpt_patient: CehrGptPatient,
    id_generator: IdGenerator,
    vocabulary_concept_to_concept_id: Dict[str, int],
    to_standard_concept_id_map: Dict[int, List[int]],
) -> List[OmopEntity]:
    omop_entities: List[OmopEntity] = []
    person_id = (
        cehrgpt_patient.patient_id
        if cehrgpt_patient.patient_id
        else id_generator.get_next_person_id()
    )
    gender_concept_id = (
        cehrgpt_patient.gender_concept_id if cehrgpt_patient.gender_concept_id else 0
    )
    race_gender_id = (
        cehrgpt_patient.race_concept_id if cehrgpt_patient.race_concept_id else 0
    )
    birth_datetime = cehrgpt_patient.birth_datetime
    person = Person(person_id, gender_concept_id, birth_datetime.year, race_gender_id)
    omop_entities.append(person)
    for visit in cehrgpt_patient.visits:
        visit_occurrence = VisitOccurrence(
            visit_occurrence_id=(
                visit.visit_id
                if visit.visit_id
                else id_generator.get_next_visit_occurrence_id()
            ),
            visit_concept_id=visit.visit_concept_id if visit.visit_concept_id else 0,
            visit_start_datetime=visit.visit_start_datetime,
            person=person,
            discharged_to_concept_id=visit.discharge_to_concept_id,
            visit_source_value=visit.visit_type,
        )
        omop_entities.append(visit_occurrence)
        for event in visit.events:
            if event.domain.lower() == "condition":
                condition_occurrence = ConditionOccurrence(
                    condition_occurrence_id=id_generator.get_next_condition_occurrence_id(),
                    condition_concept_id=vocabulary_concept_to_concept_id.get(
                        event.code, 0
                    ),
                    visit_occurrence=visit_occurrence,
                    condition_datetime=event.time,
                    condition_source_value=event.code,
                )
                omop_entities.extend(
                    condition_occurrence.map_to_standard(to_standard_concept_id_map)
                )
            elif event.domain.lower() == "procedure":
                procedure_occurrence = ProcedureOccurrence(
                    procedure_occurrence_id=id_generator.get_next_procedure_occurrence_id(),
                    procedure_concept_id=vocabulary_concept_to_concept_id.get(
                        event.code, 0
                    ),
                    visit_occurrence=visit_occurrence,
                    procedure_datetime=event.time,
                    procedure_source_value=event.code,
                )
                omop_entities.extend(
                    procedure_occurrence.map_to_standard(to_standard_concept_id_map)
                )
            elif event.domain.lower() == "drug":
                drug_exposure = DrugExposure(
                    drug_exposure_id=id_generator.get_next_drug_exposure_id(),
                    drug_concept_id=vocabulary_concept_to_concept_id.get(event.code, 0),
                    visit_occurrence=visit_occurrence,
                    drug_datetime=event.time,
                    drug_source_value=event.code,
                )
                omop_entities.extend(
                    drug_exposure.map_to_standard(to_standard_concept_id_map)
                )
        return omop_entities


def main(args):
    concept = pl.read_parquet(os.path.join(args.vocabulary_dir, "concept", "*.parquet"))
    concept_relationship = pl.read_parquet(
        os.path.join(args.vocabulary_dir, "concept_relationship", "*.parquet")
    )
    # Step 1: convert cehrgpt patients to a OMOP instance that contains the non-standard concepts
    id_generator = IdGenerator()
    vocabulary_concept_concept_id = generate_vocabulary_concept_to_concept_id_map(
        concept
    )
    to_standard_concept_id_map = generate_to_standard_concept_id_map(
        concept_relationship
    )

    entities = []
    all_json_files = glob.glob(
        os.path.join(args.cehrgpt_patient_input_folder, "*.json")
    )
    for json_file in tqdm(all_json_files, total=len(all_json_files)):
        try:
            with open(json_file, "r") as f:
                cehrgpt_patient = load_cehrgpt_patient_from_json(json.load(f))
                entities.extend(
                    convert_cehrgpt_to_omop_events(
                        cehrgpt_patient,
                        id_generator,
                        vocabulary_concept_concept_id,
                        to_standard_concept_id_map,
                    )
                )
        except Exception as e:
            logger.error(e)

        if entities and len(entities) % args.batch_size == 0:
            flush_to_disk(entities, args.output_folder)
            entities.clear()

    if entities:
        flush_to_disk(entities, args.output_folder)


def flush_to_disk(entities: List[OmopEntity], output_folder: str):
    entity_by_name = defaultdict(list)
    entity_schema_by_name = dict()
    for entity in entities:
        entity_by_name[entity.get_table_name()].append(entity.export_as_json())
        if entity.get_table_name() not in entity_schema_by_name:
            entity_schema_by_name[entity.get_table_name()] = entity.get_schema()

    for table_name, entities in entity_by_name.items():
        folder = Path(output_folder) / table_name
        if not folder.exists():
            folder.mkdir(parents=True)
        pd.DataFrame(entities, columns=entity_schema_by_name[table_name]).to_parquet(
            folder / f"{uuid.uuid4()}.parquet"
        )


def create_arg_parser():
    parser = argparse.ArgumentParser("Convert cehrgpt patients to omop")
    parser.add_argument(
        "--vocabulary_dir",
        action="store",
        required=True,
    )
    parser.add_argument(
        "--cehrgpt_patient_input_folder",
        action="store",
        required=True,
    )
    parser.add_argument(
        "--output_folder",
        action="store",
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        action="store",
        default=1024,
        required=False,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = create_arg_parser()
    main(args)
