import os
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from cehrgpt.generation.cehrgpt_patient.convert_patient_sequence import (
    get_cehrgpt_patient_converter,
)
from cehrgpt.generation.omop_converter_batch import (
    create_arg_parser,
    get_num_records,
    main_parallel,
    record_generator,
)


def validation_patient(
    patient_sequence_parquet_files: List[str],
    domain_map: Dict[str, str],
    output_folder: str,
    buffer_size: int,
    *args,
) -> None:
    error_messages = []
    patient_record_generator = record_generator(patient_sequence_parquet_files)
    total_record = get_num_records(patient_sequence_parquet_files)
    for index, record in tqdm(enumerate(patient_record_generator), total=total_record):
        concept_ids = getattr(record, "concept_ids")
        person_id = getattr(record, "person_id", None)
        patient_converter = get_cehrgpt_patient_converter(
            concept_ids=concept_ids,
            concept_domain_mapping=domain_map,
        )
        if not patient_converter.is_validation_passed:
            error_messages.append(
                {
                    "person_id": person_id,
                    "error_messages": patient_converter.get_error_messages(),
                }
            )
        if index != 0 and index % buffer_size == 0:
            if error_messages:
                pd.DataFrame(
                    error_messages, columns=["person_id", "error_messages"]
                ).to_parquet(os.path.join(output_folder, f"batch_{index}.parquet"))
                error_messages.clear()

    # Final flush to the disk if there are still records in the cache
    if error_messages:
        pd.DataFrame(
            error_messages, columns=["person_id", "error_messages"]
        ).to_parquet(os.path.join(output_folder, f"batch_final.parquet"))
        error_messages.clear()


if __name__ == "__main__":
    main_parallel(create_arg_parser(), validation_patient)
