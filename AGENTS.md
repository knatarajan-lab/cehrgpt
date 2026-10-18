# Agent guide: automatic MEDS patient timelines

This branch implements [issue #140](https://github.com/knatarajan-lab/cehrgpt/issues/140) in
[PR #187](https://github.com/knatarajan-lab/cehrgpt/pull/187). See [ISSUE_140.md](ISSUE_140.md) for the issue-specific
implementation record. Use this file as the handoff for contributors and coding agents working on the PR.

## Goal

Construct CEHR-GPT patient sequences directly from MEDS events without selecting or instantiating the legacy
`MedsToBertMimic4` or `MedsToCehrbertOMOP` conversion classes. The implementation must support both OMOP-converted
MEDS and MIMIC-IV MEDS, while allowing datasets to add their own event patterns through YAML.

## Required design

- There is one generic conversion path: `YamlDrivenConversion` plus the visit builders in
  `src/cehrgpt/data/meds_dataset.py`.
- `src/cehrgpt/data/visit_rules/defaults.yaml` is the only packaged rule file. It contains the union of the standard
  OMOP and MIMIC-IV patterns, and every concept is annotated with its source data format.
- Do not recreate separate OMOP and MIMIC YAML files or select rules using
  `meds_to_cehrbert_conversion_type`.
- The only inputs are `defaults.yaml` and a user YAML file passed through `--meds_visit_rules_file`, applied on top of
  the defaults. Do not read or write rules in the `metadata` folder of the MEDS dataset: a rules file is not a standard
  MEDS file, and MEDS metadata is parquet. Pattern lists are merged with stable de-duplication. A later `text_events`
  entry replaces an earlier entry with the same `code`.
- Matching patterns are regular expressions evaluated with `re.match`, so they match from the beginning of an event
  code.
- Event timestamps are the ground truth. Do not restore a configurable `disconnect_problem_list_events` behavior.

## Visit-boundary rules

Boundary selection is automatic for each patient:

1. Use visit-ID boundaries when events contain `visit_id`, or when an explicit `visit_events` pattern matches.
2. Otherwise, construct admission windows from ED, admission, and discharge events.

For visit-ID data, events more than one day outside their referenced visit are disconnected and placed according to
their own timestamps. Unlinked events occurring within a visit's date range are attached to that visit; remaining
events form timestamp-based visits.

For admission-window data, events are first grouped into daily blocks. ED activity can begin a stay when followed by
an admission within 24 hours. A discharge closes the stay. An event block starting within 12 hours after discharge is
kept with that stay. If a prediction cutoff occurs during an active admission, the final visit remains open by setting
`visit_end_datetime` to `None`.

## CEHR-GPT token semantics

- Every visit starts with `[VS]`.
- A completed visit ends with `[VE]`.
- An active visit at prediction time has no `[VE]`.
- Day-gap tokens are inserted between visits.
- Inpatient and ED visits are explicitly marked with `is_inpatient_or_ed`.
- Inpatient day tokens are emitted for both inpatient and ED visits.
- Inpatient hour tokens are emitted only when `include_inpatient_hour_token` is enabled.

Keep `is_inpatient_or_ed` in the Hugging Face dataset schema and preserve it when converting columnar visit data back
to visit objects. The token mapper relies on this field; visit-code heuristics alone are insufficient for custom YAML
codes.

## YAML extension example

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

Only add patterns that are absent from `defaults.yaml`. Keep an inline data-format annotation beside every new
packaged default concept.

## Important files

- `src/cehrgpt/data/visit_rules/defaults.yaml`: standard annotated OMOP and MIMIC-IV rules.
- `src/cehrgpt/data/meds_visit_rules.py`: YAML validation, layered rule merging, and the generic conversion adapter.
- `src/cehrgpt/data/meds_dataset.py`: boundary detection and MEDS-to-visit construction.
- `src/cehrgpt/data/hf_cehrgpt_dataset_mapping.py`: `[VS]`, `[VE]`, inter-visit, inpatient-day, and optional
  inpatient-hour token generation.
- `src/cehrgpt/runners/data_utils.py`: finetuning integration.
- `src/cehrgpt/runners/hf_cehrgpt_pretrain_runner.py`: pretraining integration.
- `tests/unit_tests/data/meds_visit_rules_test.py`: focused unit and integration coverage.

## Verification

Run the focused suite after changing this feature:

```bash
PYTHONPATH=src python -m pytest tests/unit_tests/data/meds_visit_rules_test.py -q
```

The current expected result is 30 passing tests. Also run `git diff --check` and the broader project tests when the
environment has all optional dependencies.

Tests should continue to cover:

- default and user-rule merging;
- both OMOP and MIMIC-IV event shapes through the same defaults;
- visit-ID and admission-window boundary detection;
- event-time handling for events linked to the wrong visit;
- ED-to-inpatient merging and discharge handling;
- open visits at prediction time without `[VE]`;
- `[VS]`, `[VE]`, day-gap, inpatient-day, and optional inpatient-hour tokens;
- custom YAML ED/inpatient concepts; and
- preservation of `is_inpatient_or_ed` through columnar dataset mappings.
