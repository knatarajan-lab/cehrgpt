# Tokenize a Full Dataset for Linear Probing

Tokenize a patient population once with `tokenize_dataset.py`, then reuse that
single tokenized dataset across every cohort task via
`--tokenized_full_dataset_path`, instead of re-tokenizing per cohort.

Cohort scripts (`run_linear_prob.sh` for linear probing,
`time_to_event_prediction.py` for zero-shot prediction) slice out just each
patient's tokens up to their index date from that one tokenized dataset, via
`extract_cohort_sequences`.

## Prerequisites

Ensure you have:

1. **Trained CEHR-GPT Model and Tokenizer**: available at `$CEHR_GPT_MODEL_DIR`
2. **Raw Patient Sequences**: untokenized parquet files (`concept_ids`, `ages`,
   `epoch_times`, etc., no `input_ids` yet) for the full population

## Step 1: Tokenize the Full Dataset

Run `tokenize_dataset.py` once against the full population's raw event
sequences and the pretrained tokenizer:

```bash
python -m cehrgpt.tools.tokenize_dataset \
    --tokenizer_name_or_path $CEHR_GPT_MODEL_DIR \
    --data_folder sample_data/pretrain \
    --output_dir /path/to/tokenized_output \
    --tokenized_dataset_name full_population_tokenized \
    --validation_split_percentage 0.05 \
    --preprocessing_num_workers 4
```

`sample_data/pretrain/patient_sequence.parquet` (100 patients, already in this
repo) is an example of the raw input format the script expects.

### Parameter Details

- `--tokenizer_name_or_path`: path to an existing, pretrained `CehrGptTokenizer`
- `--data_folder`: raw parquet patient sequences to tokenize
- `--output_dir` / `--tokenized_dataset_name`: the tokenized dataset is saved
  to `<output_dir>/<tokenized_dataset_name>/`
- `--validation_split_percentage`: fraction of patients held out as validation

The result is saved as a Hugging Face `DatasetDict` (`train`/`validation`
splits, `dataset_dict.json` at the root). That path is what you pass as
`--tokenized_full_dataset_path` in step 2.

## Step 2: Run Linear Probing Against the Tokenized Dataset

`run_linear_prob.sh` slices cohort sequences directly out of the tokenized
dataset from step 1, instead of re-tokenizing each cohort:

```bash
scripts/run_linear_prob.sh \
    --base_dir /path/to/cohorts \
    --cohort_folder /path/to/cohorts \
    --tokenized_full_dataset_path /path/to/tokenized_output/full_population_tokenized \
    --dataset_prepared_path /path/to/dataset_prepared \
    --model_path $CEHR_GPT_MODEL_DIR \
    --preprocessing_workers 4 \
    --batch_size 32 \
    --output_dir /path/to/results \
    --observation_window 365
```

### Parameter Details

- `--base_dir`: parent directory of per-cohort subdirectories, each
  discovered and processed in turn
- `--cohort_folder`: where cohort person_id/index_date/label parquet files
  live (defaults to `--base_dir` if omitted)
- `--tokenized_full_dataset_path`: the tokenized dataset produced in step 1
- `--observation_window`: days of history before the index date to include;
  omit it to use each patient's full history up to the index date

### Building an Example Cohort

This repo has no prebuilt example cohort, but one can be made up from the same
sample dataset: pick a point partway through each patient's history as the
index date, and any binary value as the label.

```python
import pandas as pd, numpy as np, os

df = pd.read_parquet("sample_data/pretrain/patient_sequence.parquet")
rng = np.random.default_rng(42)

def pick_index_date(epoch_times):
    times = [t for t in epoch_times if t > 0]
    return pd.Timestamp(times[int(len(times) * 0.7)], unit="s") if times else None

cohort = pd.DataFrame({
    "person_id": df["person_id"],
    "index_date": df["epoch_times"].map(pick_index_date),
    "label": rng.integers(0, 2, size=len(df)),
}).dropna()

train, test = cohort.iloc[:70], cohort.iloc[70:]
os.makedirs("cohorts/my_task/train", exist_ok=True)
os.makedirs("cohorts/my_task/test", exist_ok=True)
train.to_parquet("cohorts/my_task/train/cohort.parquet")
test.to_parquet("cohorts/my_task/test/cohort.parquet")
```

`--base_dir` then points at `cohorts/`, with `my_task` as the one cohort
subdirectory `run_linear_prob.sh` discovers and processes.

## Cohort Folder Requirements

`--cohort_folder` is globbed recursively (`cohort_folder/**/*.parquet`), so
files can sit directly in the folder or nested under subdirectories like
`train/`/`test/` — both layouts work.

| Requirement | Detail |
| --- | --- |
| Column names | Auto-detected from the parquet schema: `person_id`/`index_date`/`label` (standard), or `subject_id`/`prediction_time`/`boolean_value` (MEDS) — or force MEDS with `--is_data_in_meds` |
| Population coverage | Every cohort `person_id` must exist in the tokenized dataset; a missing person raises a `RuntimeError` |
| `--observation_window` | Days of history before the index date to include; omit it to use each patient's full history up to the index date |

## Troubleshooting

- **Cohorts skipped, or 0 found**: with `--tokenized_full_dataset_path` set,
  any subdirectory of `--base_dir` counts as a cohort — no `train/`/`test/`
  subdirectories required — except `logs/`, which is reserved for the
  script's own log output.
- **`--data_folder`/`--test_data_folder` still asked for**: the underlying
  Python arg parser requires `--data_folder` regardless of mode;
  `run_linear_prob.sh` points it at the cohort directory itself, since it
  isn't otherwise read once `--tokenized_full_dataset_path` is set.
- **Demographics missing or wrong**: `gender_concept_id`/`race_concept_id`
  are loaded from `--cohort_folder` in this mode, using the same
  MEDS-vs-standard auto-detection as the cohort labels.
