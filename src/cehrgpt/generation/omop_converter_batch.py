import argparse
import datetime
import glob
import os
import uuid
from datetime import timedelta
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from cehrgpt.generation.omop_entity import (
    ConditionOccurrence,
    Death,
    DrugExposure,
    Measurement,
    OmopEntity,
    Person,
    ProcedureOccurrence,
    VisitOccurrence,
)
from cehrgpt.gpt_utils import (
    extract_time_interval_in_days,
    generate_artificial_time_tokens,
    is_inpatient_att_token,
    is_visit_end,
    is_visit_start,
)
from cehrgpt.models.tokenization_hf_cehrgpt import END_TOKEN

# TODO: move these to cehrbert_data
STOP_TOKENS = ["VE", "[VE]", END_TOKEN]

OOV = "[OOV]"
CURRENT_PATH = Path(__file__).parent
START_TOKEN_SIZE = 4
ATT_TIME_TOKENS = generate_artificial_time_tokens()
TABLE_LIST = [
    "person",
    "visit_occurrence",
    "condition_occurrence",
    "procedure_occurrence",
    "drug_exposure",
    "death",
    "measurement",
]
DISCHARGE_CONCEPT_LIST = [4216643, 4021968, 4146681, 4161979]
OOV_CONCEPT_MAP = {
    1525734: "Drug",
    779414: "Drug",
    722117: "Drug",
    722118: "Drug",
    722119: "Drug",
    905420: "Drug",
    1525543: "Drug",
}


def extract_gender_concept_id(gender_token: str) -> int:
    if gender_token.startswith("Gender/"):
        return int(gender_token[len("Gender/") :])
    elif gender_token.isnumeric():
        return int(gender_token)
    else:
        return 0


def extract_race_concept_id(race_token: str) -> int:
    if race_token.startswith("Race/"):
        return int(race_token[len("Race/") :])
    elif race_token.isnumeric():
        return int(race_token)
    else:
        return 0


def create_folder_if_not_exists(output_folder, table_name):
    if not os.path.isdir(Path(output_folder) / table_name):
        os.mkdir(Path(output_folder) / table_name)


def generate_omop_concept_domain(concept_parquet) -> Dict[int, str]:
    """
    Generate a dictionary of concept_id to domain_id.

    :param concept_parquet: concept dataframe read from parquet file
    :return: dictionary of concept_id to domain_id
    """
    domain_dict = {}
    for i in concept_parquet.itertuples():
        domain_dict[i.concept_id] = i.domain_id
    return domain_dict


def generate_lab_stats_mapping(
    all_lab_stats: Optional[List[Dict[str, Any]]]
) -> Dict[int, Dict[str, Any]]:
    lab_stats_mapping = {}
    if all_lab_stats is not None:
        for lab_stats in all_lab_stats:
            # TODO: the numeric check will not hold true if we concatenate
            #  the concept with the corresponding unit concept
            if lab_stats["concept_id"].isnumeric():
                concept_id = int(lab_stats["concept_id"])
                count = lab_stats["concept_id"]
                if (concept_id in lab_stats_mapping) and (
                    count > lab_stats_mapping[concept_id]["count"]
                ):
                    lab_stats_mapping[concept_id] = {
                        "mean": lab_stats["mean"],
                        "std": lab_stats["std"],
                        "count": lab_stats["count"],
                    }
                else:
                    lab_stats_mapping[concept_id] = {
                        "mean": lab_stats["mean"],
                        "std": lab_stats["std"],
                        "count": lab_stats["count"],
                    }
    return lab_stats_mapping


def append_to_dict(
    export_dict: Dict[str, Dict[int, OmopEntity]],
    omop_entity: OmopEntity,
    entity_id: int,
):
    if omop_entity.get_table_name() not in export_dict:
        export_dict[omop_entity.get_table_name()] = {}
    export_dict[omop_entity.get_table_name()][entity_id] = omop_entity


def delete_bad_sequence(
    export_dict: Dict[str, Dict[int, OmopEntity]],
    id_mappings: Dict[str, Dict[int, int]],
    person_id: int,
):
    for table_name, id_mapping in id_mappings.items():
        omop_id_mapping = np.array(list(id_mapping.keys()))
        person_id_mapping = np.array(list(id_mapping.values()))
        ids_to_delete = omop_id_mapping[np.where(person_id_mapping == person_id)]
        for id in ids_to_delete:
            export_dict[table_name].pop(id)


def export_and_clear(
    output_folder: str,
    export_dict: Dict[str, Dict[int, OmopEntity]],
    export_error: Dict[str, Dict[str, str]],
    id_mappings_dict: Dict[str, Dict[int, int]],
    pt_seq_dict: Dict[int, str],
    is_parquet: bool = True,
):
    for table_name, records_to_export in export_dict.items():

        records_in_json = []
        # If there is no omop_entity, we skip it
        if len(export_dict[table_name]) == 0:
            continue

        for entity_id, omop_entity in export_dict[table_name].items():
            try:
                records_in_json.append(omop_entity.export_as_json())
            except AttributeError:
                # append patient sequence to export error list using pt_seq_dict.
                if table_name not in export_error:
                    export_error[table_name] = []
                person_id = id_mappings_dict[table_name][entity_id]
                export_error[table_name].append(pt_seq_dict[person_id])
                continue
        schema = next(iter(records_to_export.items()))[1].get_schema()
        output_folder_path = Path(output_folder)
        file_path = output_folder_path / table_name / f"{uuid.uuid4()}.parquet"
        table_df = pd.DataFrame(records_in_json, columns=schema)

        if is_parquet:
            table_df.to_parquet(file_path)
        else:
            table_df.to_csv(file_path, header=schema, index=False)

        export_dict[table_name].clear()


def _is_none(x):
    return x is None or np.isnan(x)


def get_num_records(parquet_files: List[str]):
    total = 0
    for file_path in parquet_files:
        parquet_file = pq.ParquetFile(file_path)
        total += parquet_file.metadata.num_rows
    return total


def record_generator(parquet_files):
    for file_path in parquet_files:
        df = pd.read_parquet(file_path)
        for record in df.itertuples():
            yield record


def gpt_to_omop_converter_batch(
    const: int,
    patient_sequence_parquet_files: List[str],
    domain_map: Dict[int, str],
    output_folder: str,
    buffer_size: int,
    use_original_person_id: bool,
):
    omop_export_dict = {}
    error_dict = {}
    export_error = {}
    id_mappings_dict = {}
    pt_seq_dict = {}

    for tb in TABLE_LIST:
        create_folder_if_not_exists(output_folder, tb)
        id_mappings_dict[tb] = {}

    visit_occurrence_id: int = const + 1
    condition_occurrence_id: int = const + 1
    procedure_occurrence_id: int = const + 1
    drug_exposure_id: int = const + 1
    measurement_id: int = const + 1

    # Default the person_id
    person_id: int = const + 1

    patient_record_generator = record_generator(patient_sequence_parquet_files)
    total_record = get_num_records(patient_sequence_parquet_files)

    for index, record in tqdm(enumerate(patient_record_generator), total=total_record):
        bad_sequence = False
        # If original_person_id is set to true, we retrieve it from the record.
        # If person_id doest not exist in the record, we use the default_person_id
        if use_original_person_id:
            person_id = getattr(record, "person_id", person_id)

        # Retrieve the
        concept_ids = getattr(record, "concept_ids")
        is_numeric_types = getattr(record, "is_numeric_types", None)
        number_as_values = getattr(record, "number_as_values", None)
        concept_as_values = getattr(record, "concept_as_values", None)
        units = getattr(record, "units", None)

        # Skip the start token if it is the first token
        if "start" in concept_ids[0].lower():
            concept_ids = concept_ids[1:]
            if is_numeric_types is not None:
                is_numeric_types = is_numeric_types[1:]
            if number_as_values is not None:
                number_as_values = number_as_values[1:]
            if concept_as_values is not None:
                concept_as_values = concept_as_values[1:]
            if units is not None:
                units = units[1:]

        clinical_events = concept_ids[START_TOKEN_SIZE:]
        # Skip the sequences whose sequence length is 0
        if len(clinical_events) == 0:
            continue
        # Skip the patients whose last token is not a valid end token
        if clinical_events[-1] not in STOP_TOKENS:
            continue

        is_numeric_types = (
            is_numeric_types[START_TOKEN_SIZE:]
            if is_numeric_types is not None and not np.all(pd.isna(is_numeric_types))
            else None
        )
        number_as_values = (
            number_as_values[START_TOKEN_SIZE:]
            if number_as_values is not None and not np.all(pd.isna(number_as_values))
            else None
        )
        concept_as_values = (
            concept_as_values[START_TOKEN_SIZE:]
            if concept_as_values is not None and not np.all(pd.isna(concept_as_values))
            else None
        )
        units = (
            units[START_TOKEN_SIZE:]
            if units is not None and not np.all(pd.isna(units))
            else None
        )

        # TODO:Need to decode if the input is tokenized
        [start_year, start_age, start_gender, start_race] = concept_ids[
            0:START_TOKEN_SIZE
        ]
        if "year" not in start_year.lower():
            continue

        try:
            start_year = start_year.split(":")[1]
            start_age = start_age.split(":")[1]
            birth_year = int(start_year) - int(start_age)
        except Exception as e:
            print(
                f"Failed to convert {concept_ids[0:START_TOKEN_SIZE]} due to {e}, skipping to the next record"
            )
            continue

        # Skip the patients whose birth year is either before 1900 or after this year
        if int(birth_year) < 1900 or int(birth_year) > datetime.date.today().year:
            continue

        p = Person(
            person_id=person_id,
            gender_concept_id=extract_gender_concept_id(start_gender),
            year_of_birth=birth_year,
            race_concept_id=extract_race_concept_id(start_race),
        )

        append_to_dict(omop_export_dict, p, person_id)
        id_mappings_dict["person"][person_id] = person_id
        pt_seq_dict[person_id] = " ".join(concept_ids)
        discharged_to_concept_id = 0
        date_cursor = datetime.datetime(year=int(start_year), month=1, day=1)
        vo = None
        inpatient_visit_indicator = False

        for event_idx, event in enumerate(clinical_events, 0):
            if event == OOV:
                continue
            # For bad sequences, we don't proceed further and break from the for loop
            if bad_sequence:
                break
            if is_visit_start(event):
                if event_idx == len(clinical_events) - 1:
                    break
                elif clinical_events[event_idx + 1] == "[DEATH]":
                    # If the [DEATH] token is not placed at the end of the sequence, this is a bad sequence
                    if event_idx + 2 != len(clinical_events) - 1:
                        bad_sequence = True
                        break
                    death = Death(p, date_cursor.date())
                    append_to_dict(omop_export_dict, death, person_id)
                    id_mappings_dict["death"][person_id] = person_id
                else:
                    try:
                        if clinical_events[event_idx + 1].startswith("Visit/"):
                            visit_concept_id = int(
                                clinical_events[event_idx + 1][len("Visit/") :]
                            )
                        else:
                            visit_concept_id = int(clinical_events[event_idx + 1])
                        inpatient_visit_indicator = visit_concept_id in [
                            9201,
                            262,
                            8971,
                            8920,
                        ]
                        if visit_concept_id in domain_map:
                            if (
                                domain_map[visit_concept_id] != "Visit"
                                and visit_concept_id != 0
                            ):
                                bad_sequence = True
                                break
                        else:
                            bad_sequence = True
                            break

                    except (IndexError, ValueError):
                        error_dict[person_id] = {}
                        error_dict[person_id]["concept_ids"] = " ".join(concept_ids)
                        error_dict[person_id]["error"] = "Wrong visit concept id"
                        bad_sequence = True
                        continue

                    vo = VisitOccurrence(
                        visit_occurrence_id, visit_concept_id, date_cursor, p
                    )
                    append_to_dict(omop_export_dict, vo, visit_occurrence_id)
                    id_mappings_dict["visit_occurrence"][
                        visit_occurrence_id
                    ] = person_id
                    visit_occurrence_id += 1

            elif event in ATT_TIME_TOKENS:
                if event[0] == "D":
                    att_date_delta = int(event[1:])
                elif event[0] == "W":
                    att_date_delta = int(event[1:]) * 7
                elif event[0] == "M":
                    att_date_delta = int(event[1:]) * 30
                elif event == "LT":
                    att_date_delta = 365 * 3
                else:
                    att_date_delta = 0
                # Between visits, the date delta is simply calculated as the date difference
                date_cursor = date_cursor.replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                date_cursor = date_cursor + timedelta(days=att_date_delta)
            elif inpatient_visit_indicator and is_inpatient_att_token(event):
                inpatient_time_span_in_days = extract_time_interval_in_days(event)
                # Reset the data cursor to the start of the day before adding the num of days parsed out from the token
                date_cursor = date_cursor.replace(hour=0, minute=0, second=0)
                date_cursor = date_cursor + timedelta(days=inpatient_time_span_in_days)
            elif inpatient_visit_indicator and event.startswith("i-H"):
                # Handle hour tokens differently than the day tokens
                # The way we construct the inpatient hour tokens is that the sum of the consecutive
                # hour tokens cannot exceed the current day, so the data_cursor is bounded by a
                # theoretical upper limit
                upper_bound = date_cursor.replace(
                    hour=0, minute=0, second=0
                ) + timedelta(hours=23, minutes=59, seconds=59)
                hour_delta = int(event[3:])
                date_cursor = date_cursor + timedelta(hours=hour_delta)
                if date_cursor > upper_bound:
                    date_cursor = upper_bound
            elif is_visit_end(event):
                if vo is None:
                    bad_sequence = True
                    break
                # If it's a VE token, nothing needs to be updated because it just means the visit ended
                if inpatient_visit_indicator:
                    vo.set_discharged_to_concept_id(discharged_to_concept_id)
                    vo.set_visit_end_date(date_cursor)
                    # if the discharged_to_concept_id patient had died, the death record is created
                    if discharged_to_concept_id == 4216643:
                        death = Death(
                            p, date_cursor.date(), death_type_concept_id=32823
                        )
                        append_to_dict(omop_export_dict, death, person_id)
                        id_mappings_dict["death"][person_id] = person_id
                        # If death record is generated, we need to stop the sequence conversion
                        break
                else:
                    pass
            elif event in [
                "START",
                start_year,
                start_age,
                start_gender,
                start_race,
                "[DEATH]",
            ]:
                # If it's a start token, skip it
                pass
            elif event.endswith("/0"):
                # This should capture the concept such as Visit/0, Discharge/0
                pass
            else:
                try:
                    concept_id = int(event)
                    if (
                        concept_id not in domain_map
                        and concept_id not in OOV_CONCEPT_MAP
                    ):
                        error_dict[person_id] = {}
                        error_dict[person_id]["concept_ids"] = " ".join(concept_ids)
                        error_dict[person_id][
                            "error"
                        ] = f"No concept id found: {concept_id}"
                        bad_sequence = True
                        continue
                    else:
                        # If the current concept_id is 'Patient Died', this means it can only occur in the
                        # discharged_to_concept_id field, which indicates the current visit has to be an inpatient
                        # visit, this concept_id can only appear at the second last position
                        if concept_id == 4216643:
                            # If the current visit is not inpatient, reject the sequence
                            if not inpatient_visit_indicator:
                                bad_sequence = True
                                continue
                            # # If the current token is not the second last one of the sequence, reject because
                            # # death can only appear at the end of the sequence
                            # if idx + 1 != len(tokens_generated) - 1:
                            #     bad_sequence = True
                            #     continue
                            # we also enforce the rule where the sequence has to end on a VE token
                            if event_idx + 1 < len(
                                clinical_events
                            ) and not is_visit_end(clinical_events[event_idx + 1]):
                                bad_sequence = True
                                continue

                        if concept_id in domain_map:
                            domain = domain_map[concept_id]
                        elif concept_id in OOV_CONCEPT_MAP:
                            domain = OOV_CONCEPT_MAP[concept_id]
                        else:
                            domain = None

                        if domain == "Visit" or concept_id in DISCHARGE_CONCEPT_LIST:
                            discharged_to_concept_id = concept_id
                        elif domain == "Condition":
                            co = ConditionOccurrence(
                                condition_occurrence_id, concept_id, vo, date_cursor
                            )
                            append_to_dict(
                                omop_export_dict, co, condition_occurrence_id
                            )
                            id_mappings_dict["condition_occurrence"][
                                condition_occurrence_id
                            ] = person_id
                            condition_occurrence_id += 1
                        elif domain == "Procedure":
                            po = ProcedureOccurrence(
                                procedure_occurrence_id, concept_id, vo, date_cursor
                            )
                            append_to_dict(
                                omop_export_dict, po, procedure_occurrence_id
                            )
                            id_mappings_dict["procedure_occurrence"][
                                procedure_occurrence_id
                            ] = person_id
                            procedure_occurrence_id += 1
                        elif domain == "Drug":
                            de = DrugExposure(
                                drug_exposure_id, concept_id, vo, date_cursor
                            )
                            append_to_dict(omop_export_dict, de, drug_exposure_id)
                            id_mappings_dict["drug_exposure"][
                                drug_exposure_id
                            ] = person_id
                            drug_exposure_id += 1
                        elif domain == "Measurement":
                            number_as_value = (
                                number_as_values[event_idx]
                                if number_as_values is not None
                                else None
                            )
                            concept_as_value = (
                                concept_as_values[event_idx]
                                if concept_as_values is not None
                                else None
                            )
                            is_numeric_type = (
                                is_numeric_types[event_idx]
                                if is_numeric_types is not None
                                else None
                            )
                            unit = units[event_idx] if units is not None else None
                            m = Measurement(
                                measurement_id,
                                measurement_concept_id=concept_id,
                                is_numeric_type=is_numeric_type,
                                value_as_number=number_as_value,
                                value_as_concept_id=concept_as_value,
                                visit_occurrence=vo,
                                measurement_datetime=date_cursor,
                                unit_source_value=unit,
                            )
                            append_to_dict(omop_export_dict, m, measurement_id)
                            id_mappings_dict["measurement"][measurement_id] = person_id
                            measurement_id += 1

                except ValueError:
                    error_dict[person_id] = {}
                    error_dict[person_id]["concept_ids"] = " ".join(concept_ids)
                    error_dict[person_id]["error"] = f"Wrong concept id: {event}"
                    bad_sequence = True
                    continue
        if bad_sequence:
            delete_bad_sequence(omop_export_dict, id_mappings_dict, person_id)

        if not use_original_person_id:
            person_id += 1

        if index != 0 and index % buffer_size == 0:
            export_and_clear(
                output_folder,
                omop_export_dict,
                export_error,
                id_mappings_dict,
                pt_seq_dict,
            )

    # Final flush to the disk if there are still records in the cache
    export_and_clear(
        output_folder, omop_export_dict, export_error, id_mappings_dict, pt_seq_dict
    )

    with open(Path(output_folder) / "concept_errors.txt", "w") as f:
        error_dict["total"] = len(error_dict)
        f.write(str(error_dict))
    with open(Path(output_folder) / "export_errors.txt", "w") as f:
        total = 0
        for k, v in export_error.items():
            total += len(v)
        export_error["total"] = total
        f.write(str(export_error))


def main(args):
    all_parquet_files = glob.glob(
        os.path.join(args.patient_sequence_path, "*parquet"), recursive=True
    )
    if len(all_parquet_files) == 0:
        raise RuntimeError(f"No parquet files found in {args.patient_sequence_path}")

    print(
        f"There are total {len(all_parquet_files)} parquet files detected in {args.patient_sequence_path}."
    )
    if not os.path.exists(args.output_folder):
        Path(args.output_folder).mkdir(parents=True, exist_ok=True)

    batched_parquet_files = np.array_split(all_parquet_files, args.cpu_cores)
    concept_pd = pd.read_parquet(args.concept_path)
    domain_map = generate_omop_concept_domain(concept_pd)

    pool_tuples = []
    # TODO: Need to make this dynamic
    const = 10000000
    for i in range(1, args.cpu_cores + 1):
        pool_tuples.append(
            (
                const * i,
                batched_parquet_files[i - 1],
                domain_map,
                args.output_folder,
                args.buffer_size,
                args.use_original_person_id,
            )
        )

    with Pool(processes=args.cpu_cores) as p:
        p.starmap(gpt_to_omop_converter_batch, pool_tuples)
        p.close()
        p.join()

    return print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for converting patient sequences to OMOP"
    )
    parser.add_argument(
        "--output_folder",
        dest="output_folder",
        action="store",
        help="The path for the output_folder",
        required=True,
    )
    parser.add_argument(
        "--concept_path",
        dest="concept_path",
        action="store",
        help="The path for your concept_path",
        required=True,
    )
    parser.add_argument(
        "--buffer_size",
        dest="buffer_size",
        action="store",
        type=int,
        help="The size of the batch",
        required=False,
        default=1024,
    )
    parser.add_argument(
        "--patient_sequence_path",
        dest="patient_sequence_path",
        action="store",
        help="The path for your patient sequence",
        required=True,
    )
    parser.add_argument(
        "--cpu_cores",
        dest="cpu_cores",
        type=int,
        action="store",
        help="The number of cpu cores to use for multiprocessing",
        required=False,
        default=1,
    )
    parser.add_argument(
        "--use_original_person_id",
        dest="use_original_person_id",
        action="store_true",
        help="Whether or not to use the original person id",
    )

    main(parser.parse_args())
