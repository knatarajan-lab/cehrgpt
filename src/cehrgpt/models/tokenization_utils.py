# Adapted from cehrbert.models.hf_models.tokenization_utils (MIT License, same copyright holder as cehrgpt).
import json
import pickle


def load_json_file(json_file):
    try:
        with open(json_file, "r", encoding="utf-8") as reader:
            file_contents = reader.read()
            parsed_json = json.loads(file_contents)
            return parsed_json
    except Exception as e:
        raise RuntimeError(f"Can't load the json file at {json_file} due to {e}")


def agg_helper(*args, map_func):
    result = map_func(*args)
    return {"data": [pickle.dumps(result)]}
