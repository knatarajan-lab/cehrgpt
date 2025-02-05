# Special tokens for the cehr-gpt tokenizer
START_TOKEN = "[START]"
END_TOKEN = "[END]"
PAD_TOKEN = "[PAD]"
OUT_OF_VOCABULARY_TOKEN = "[OOV]"
STOP_TOKENS = ["VE", "[VE]", END_TOKEN]

INPATIENT_VISIT_CONCEPT_IDS = [
    "9201",
    "262",
    "8971",
    "8920",
]

# OMOP CONCEPT IDs
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

# TODO: add race concepts
RACE_CONCEPT_LIST = []

# TODO: add gender concepts
GENDER_CONCEPT_LIST = []
DISCHARGE_CONCEPT_LIST = [8536, 4216643, 4021968, 4146681, 4161979]
OOV_CONCEPT_MAP = {
    1525734: "Drug",
    779414: "Drug",
    722117: "Drug",
    722118: "Drug",
    722119: "Drug",
    905420: "Drug",
    1525543: "Drug",
}
