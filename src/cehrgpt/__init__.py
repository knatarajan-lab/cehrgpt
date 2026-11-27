def __getattr__(name):
    # Lazy, so importing cehrgpt does not import torch and the data modules
    if name == "load_prepared_dataset":
        from cehrgpt.runners.data_utils import load_prepared_dataset

        return load_prepared_dataset
    raise AttributeError(f"module 'cehrgpt' has no attribute {name!r}")
