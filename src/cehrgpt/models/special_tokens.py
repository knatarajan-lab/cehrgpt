# Special tokens for the cehr-gpt tokenizer
START_TOKEN = "[START]"
END_TOKEN = "[END]"
PAD_TOKEN = "[PAD]"
OUT_OF_VOCABULARY_TOKEN = "[OOV]"
STOP_TOKENS = ["VE", "[VE]", END_TOKEN]

# TODO: complete the list of inpatient visit concepts
INPATIENT_VISIT_CONCEPT_LIST = [
    "9201",
    "262",
    "8971",
    "8920",
]

# TODO: complete the list of visit concepts
VISIT_CONCEPT_LIST = [
    "9202",
    "9203",
    "581477",
    "9201",
    "5083",
    "262",
    "38004250",
    "0",
    "8883",
    "38004238",
    "38004251",
    "38004222",
    "38004268",
    "38004228",
    "32693",
    "8971",
    "38004269",
    "38004193",
    "32036",
    "8782",
]
# TODO: complete the list of race concepts
RACE_CONCEPT_LIST = [
    "0",
    "8507",
    "8532",
]

# TODO: complete the list of gender concepts
GENDER_CONCEPT_LIST = [
    "0",
    "8515",
    "8516",
    "8522",
    "8527",
    "8552",
    "8557",
    "8657",
    "38003574",
    "38003579",
    "38003581",
    "38003583",
    "38003584",
    "38003585",
    "38003586",
    "38003591",
    "38003595",
    "38003610",
    "38003613",
    "44814649",
    "44814653",
]
# TODO: complete the list of discharge concepts
DISCHARGE_CONCEPT_LIST = ["8536", "4216643", "4021968", "4146681", "4161979"]
# TODO: complete the list of OOV concepts
OOV_CONCEPT_MAP = {
    "1525734": "Drug",
    "779414": "Drug",
    "722117": "Drug",
    "722118": "Drug",
    "722119": "Drug",
    "905420": "Drug",
    "1525543": "Drug",
}
