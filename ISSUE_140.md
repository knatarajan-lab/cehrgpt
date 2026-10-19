# Issue 140: automatic CEHR-GPT timelines from MEDS

- Issue: [knatarajan-lab/cehrgpt#140](https://github.com/knatarajan-lab/cehrgpt/issues/140)
- Pull request: [knatarajan-lab/cehrgpt#187](https://github.com/knatarajan-lab/cehrgpt/pull/187)
- Branch: `auto_detect_meds_conversion`

## Objective

Automatically construct CEHR-GPT patient timelines from MEDS data without choosing a dataset-specific Python
conversion class. The same path must support MEDS converted from OMOP and MIMIC-IV and remain extensible for other
MEDS producers through YAML rules.

The resulting sequence needs to represent visit boundaries and elapsed time correctly:

- insert `[VS]` at the beginning of each visit;
- insert `[VE]` for completed visits, but not for an active visit at prediction time;
- insert day-gap tokens between visits;
- insert inpatient day tokens within inpatient and ED visits; and
- optionally insert inpatient hour tokens when enabled.

## Implemented design

### One YAML-driven conversion path

Production code uses `YamlDrivenConversion` and generic timeline builders. It does not instantiate
`MedsToBertMimic4` or `MedsToCehrbertOMOP`.

`src/cehrgpt/data/visit_rules/defaults.yaml` is the only packaged ruleset. It contains standard patterns for both
OMOP MEDS and MIMIC-IV MEDS. Every packaged concept has an inline annotation identifying its corresponding data
format.

A dataset that codes its visits differently supplies its own YAML file with `--meds_visit_rules_file <path>`. The MEDS
dataset carries no rules file: nothing is read from, or written to, its `metadata` folder, because a rules file is
not a standard MEDS file.

The merge order is defaults, then the user file. Pattern lists extend earlier lists with stable de-duplication. A later
text-event rule replaces an earlier rule with the same `code`.

### Visit building: visit ids plus rules

One builder makes the visits of a patient. It does not choose between two methods for the patient:

1. Events with a `visit_id` make one block per id.
2. Events without one, and the ones disconnected from their block, join the visit whose dates contain their date.
3. The events that are still alone are grouped into daily blocks and joined into stays by the ED, admission and
   discharge patterns. A new block never takes the id of an existing visit.

A dataset with visit ids (OMOP) is built by 1 and 2, a dataset without any (MIMIC-IV) by 3, and a patient with some
of both gets both. On 2,863 real MIMIC-IV and OMOP patients (with and without a prediction cutoff in the middle of the
record) it makes the same visits and events as the two builders it replaced.

For visit-ID data, event timestamps are authoritative. An event linked to a visit but occurring more than one day
outside that visit is disconnected and placed according to its own time. This behavior is always enabled and is no
longer exposed as `disconnect_problem_list_events`.

For the events that are placed by the rules, daily blocks are joined using these:

- an ED event can start a stay when an admission follows within 24 hours, and the stay is `Visit/ERIP`; an earlier ED is
  a visit of its own and the admission still starts a `Visit/IP` stay;
- a discharge closes the stay, and a day is cut after a discharge that an ED or admission event follows;
- a block beginning within 12 hours after discharge remains part of that stay, unless it starts a stay itself;
- a block whose events do not identify a visit is `Visit/0` (unknown), not `Visit/OP`;
- a stay that never gets a discharge ends at its last event after `max_stay_gap_days` (rule, default 3) without an
  event, so it cannot last until the next admission or the cutoff; and
- an admission still active at prediction time produces an open visit with `visit_end_datetime: null`, if the cutoff is
  within `max_stay_gap_days` of its last event.

On real MIMIC-IV (950 patients, with cutoffs), this changes 19 patients: stub visits typed by a raw
`HOSPITAL_ADMISSION//` code (an admission with no discharge before the next one) go from 27 to 3, and the longest
inpatient stay is unchanged at 11.9 days.

### Inpatient and ED timing

Each visit carries `is_inpatient_or_ed`. This classification is driven by the resolved YAML rules and preserved in the
Hugging Face dataset schema. The CEHR-GPT mapper uses the field to generate inpatient day tokens for both inpatient
and ED visits and optional inpatient hour tokens. It does not depend only on hard-coded visit concept names, so custom
YAML codes work as well.

## Configuration example

The packaged defaults require no dataset configuration for standard OMOP or MIMIC-IV MEDS. A different producer can
add its concepts as follows:

```yaml
visit_events:
  - ENCOUNTER//
ed_admission_events:
  - ENCOUNTER//ED
admission_events:
  - ENCOUNTER//INPATIENT
discharge_events:
  - DISCHARGE//
```

Patterns are regular expressions matched from the start of the event code with `re.match`.

## Main implementation files

- `src/cehrgpt/data/visit_rules/defaults.yaml`
- `src/cehrgpt/data/meds_visit_rules.py`
- `src/cehrgpt/data/meds_dataset.py`
- `src/cehrgpt/data/hf_cehrgpt_dataset_mapping.py`
- `src/cehrgpt/runners/data_utils.py`
- `src/cehrgpt/runners/hf_cehrgpt_pretrain_runner.py`
- `src/cehrgpt/runners/hf_gpt_runner_argument_dataclass.py`
- `tests/unit_tests/data/meds_visit_rules_test.py`

## Acceptance criteria

- Standard OMOP and MIMIC-IV MEDS use the same packaged defaults.
- No separate OMOP/MIMIC YAML files or production conversion-class selection remain.
- User and dataset YAML rules extend the packaged defaults.
- Visit boundaries are selected from the evidence present in each patient's events.
- Event time remains the ground truth for incorrectly linked events.
- Completed visits emit `[VE]`; an active prediction-time visit does not.
- Inpatient and ED visits emit inpatient day tokens.
- Inpatient hour tokens remain optional.
- Custom YAML visit types retain inpatient/ED classification through dataset mapping.

## Verification

Run:

```bash
PYTHONPATH=src python -m pytest tests/unit_tests/data/meds_visit_rules_test.py -q
git diff --check
```

Current result: `42 passed` for the focused MEDS rules and timeline suite.
