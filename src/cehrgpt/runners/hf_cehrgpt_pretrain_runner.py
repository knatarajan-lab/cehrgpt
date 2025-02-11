import os
import pickle
from typing import List, Optional, Union

import torch
from cehrbert.data_generators.hf_data_generator.meds_utils import (
    create_dataset_from_meds_reader,
)
from cehrbert.runners.hf_runner_argument_dataclass import (
    DataTrainingArguments,
    ModelArguments,
)
from cehrbert.runners.runner_util import (
    generate_prepared_ds_path,
    get_last_hf_checkpoint,
    get_meds_extension_path,
    load_parquet_as_dataset,
)
from datasets import Dataset, DatasetDict, IterableDatasetDict, load_from_disk
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.utils import is_flash_attn_2_available, logging

from cehrgpt.data.hf_cehrgpt_dataset import create_cehrgpt_pretraining_dataset
from cehrgpt.data.hf_cehrgpt_dataset_collator import CehrGptDataCollator
from cehrgpt.models.config import CEHRGPTConfig
from cehrgpt.models.encoder_decoder.monkey_patch_cehrgpt import register_cehrgpt_in_hf
from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel
from cehrgpt.models.pretrained_embeddings import PretrainedEmbeddings
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer
from cehrgpt.runners.gpt_runner_util import parse_runner_args
from cehrgpt.runners.hf_gpt_runner_argument_dataclass import CehrGPTArguments

LOG = logging.get_logger("transformers")


def tokenizer_exists(tokenizer_name_or_path: str) -> bool:
    # Try to load the pretrained tokenizer
    try:
        CehrGptTokenizer.from_pretrained(os.path.abspath(tokenizer_name_or_path))
        return True
    except Exception:
        LOG.info(f"The tokenizer does not exist at {tokenizer_name_or_path}")
        return False


def load_and_create_tokenizer(
    data_args: DataTrainingArguments,
    model_args: ModelArguments,
    cehrgpt_args: CehrGPTArguments,
    dataset: Optional[Union[Dataset, DatasetDict]] = None,
) -> CehrGptTokenizer:
    # Try to load the pretrained tokenizer
    tokenizer_abspath = os.path.expanduser(model_args.tokenizer_name_or_path)
    try:
        tokenizer = CehrGptTokenizer.from_pretrained(tokenizer_abspath)
    except Exception as e:
        LOG.warning(e)
        if dataset is None:
            raise RuntimeError(
                f"Failed to load the tokenizer from {tokenizer_abspath} with the error \n{e}\n"
                f"Tried to create the tokenizer, however the dataset is not provided."
            )
        tokenizer = CehrGptTokenizer.train_tokenizer(
            dataset,
            {},
            data_args,
            PretrainedEmbeddings(cehrgpt_args.pretrained_embedding_path),
        )
        tokenizer.save_pretrained(tokenizer_abspath)

    return tokenizer


def load_and_create_model(
    model_args: ModelArguments,
    cehrgpt_args: CehrGPTArguments,
    training_args: TrainingArguments,
    tokenizer: CehrGptTokenizer,
) -> CEHRGPT2LMHeadModel:
    attn_implementation = (
        "flash_attention_2" if is_flash_attn_2_available() else "eager"
    )
    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    model_abspath = os.path.expanduser(model_args.model_name_or_path)
    if cehrgpt_args.continue_pretrain:
        try:
            pretrained = CEHRGPT2LMHeadModel.from_pretrained(
                model_abspath,
                attn_implementation=attn_implementation,
                torch_dtype=torch_dtype,
            )
            # We need to instantiate some layers for cross attention for the pretrained model
            if (
                cehrgpt_args.add_cross_attention
                and not pretrained.config.add_cross_attention
            ):
                pretrained.enable_cross_attention()
            return pretrained
        except Exception as e:
            LOG.error(
                f"When continue_pretrain is set to True, it assumes that CEHR-GPT has been trained "
                f"and will be used to pretrain on new datasets. The CEHR-GPT checkpoint must exist at {model_abspath}"
            )
            raise e
    try:
        model_config = AutoConfig.from_pretrained(
            model_abspath, attn_implementation=attn_implementation
        )
    except Exception as e:
        LOG.warning(e)
        if cehrgpt_args.causal_sfm:
            model_args.max_position_embeddings += 1
        if len(tokenizer.pretrained_token_ids) > 0:
            pretrained_embedding_dim = tokenizer.pretrained_embeddings.shape[1]
        else:
            pretrained_embedding_dim = model_args.hidden_size
        model_config = CEHRGPTConfig(
            vocab_size=tokenizer.vocab_size,
            value_vocab_size=tokenizer.value_vocab_size,
            time_token_vocab_size=tokenizer.time_token_vocab_size,
            bos_token_id=tokenizer.end_token_id,
            eos_token_id=tokenizer.end_token_id,
            lab_token_ids=tokenizer.lab_token_ids,
            token_to_time_token_mapping=tokenizer.token_to_time_token_mapping,
            attn_implementation=attn_implementation,
            causal_sfm=cehrgpt_args.causal_sfm,
            demographics_size=cehrgpt_args.demographics_size,
            lab_token_penalty=cehrgpt_args.lab_token_penalty,
            lab_token_loss_weight=cehrgpt_args.lab_token_loss_weight,
            entropy_penalty=cehrgpt_args.entropy_penalty,
            entropy_penalty_alpha=cehrgpt_args.entropy_penalty_alpha,
            n_pretrained_embeddings_layers=cehrgpt_args.n_pretrained_embeddings_layers,
            use_pretrained_embeddings=len(tokenizer.pretrained_token_ids) > 0,
            pretrained_embedding_dim=pretrained_embedding_dim,
            add_cross_attention=cehrgpt_args.add_cross_attention,
            **model_args.as_dict(),
        )
    model = CEHRGPT2LMHeadModel(model_config)
    if tokenizer.pretrained_token_ids:
        model.cehrgpt.update_pretrained_embeddings(
            tokenizer.pretrained_token_ids,
            tokenizer.pretrained_embeddings,
        )
    if model.config.torch_dtype == torch.bfloat16:
        return model.bfloat16()
    elif model.config.torch_dtype == torch.float16:
        return model.half()
    return model


def main():
    cehrgpt_args, data_args, model_args, training_args = parse_runner_args()

    if data_args.streaming:
        # This is for disabling the warning message https://github.com/huggingface/transformers/issues/5486
        # This happens only when streaming is enabled
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # The iterable dataset doesn't have sharding implemented, so the number of works has to be set to 0
        # Otherwise the trainer will throw an error
        training_args.dataloader_num_workers = 0
        training_args.dataloader_prefetch_factor = None

    prepared_ds_path = generate_prepared_ds_path(data_args, model_args)
    if os.path.exists(os.path.join(data_args.data_folder, "dataset_dict.json")):
        LOG.info(f"Loading prepared dataset from disk at {data_args.data_folder}...")
        processed_dataset = load_from_disk(data_args.data_folder)
        # If the data has been processed in the past, it's assume the tokenizer has been created before.
        # we load the CEHR-GPT tokenizer from the output folder, otherwise an exception will be raised.
        tokenizer_name_or_path = os.path.expanduser(
            training_args.output_dir
            if cehrgpt_args.expand_tokenizer
            else model_args.tokenizer_name_or_path
        )
        if not tokenizer_exists(tokenizer_name_or_path):
            raise RuntimeError(
                f"The dataset has been tokenized but the corresponding tokenizer: "
                f"{model_args.tokenizer_name_or_path} does not exist"
            )
        cehrgpt_tokenizer = CehrGptTokenizer.from_pretrained(tokenizer_name_or_path)
    elif any(prepared_ds_path.glob("*")):
        LOG.info(f"Loading prepared dataset from disk at {prepared_ds_path}...")
        processed_dataset = load_from_disk(str(prepared_ds_path))
        LOG.info("Prepared dataset loaded from disk...")
        # If the data has been processed in the past, it's assume the tokenizer has been created before.
        # we load the CEHR-GPT tokenizer from the output folder, otherwise an exception will be raised.
        tokenizer_name_or_path = os.path.expanduser(
            training_args.output_dir
            if cehrgpt_args.expand_tokenizer
            else model_args.tokenizer_name_or_path
        )
        if not tokenizer_exists(tokenizer_name_or_path):
            raise RuntimeError(
                f"The dataset has been tokenized but the corresponding tokenizer: "
                f"{model_args.tokenizer_name_or_path} does not exist"
            )
        cehrgpt_tokenizer = CehrGptTokenizer.from_pretrained(tokenizer_name_or_path)
    else:
        # If the data is in the MEDS format, we need to convert it to the CEHR-BERT format
        if data_args.is_data_in_meds:
            meds_extension_path = get_meds_extension_path(
                data_folder=data_args.data_folder,
                dataset_prepared_path=data_args.dataset_prepared_path,
            )
            try:
                LOG.info(
                    "Trying to load the MEDS extension from disk at %s...",
                    meds_extension_path,
                )
                dataset = load_from_disk(meds_extension_path)
                if data_args.streaming:
                    if isinstance(dataset, DatasetDict):
                        dataset = {
                            k: v.to_iterable_dataset(
                                num_shards=training_args.dataloader_num_workers
                            )
                            for k, v in dataset.items()
                        }
                    else:
                        dataset = dataset.to_iterable_dataset(
                            num_shards=training_args.dataloader_num_workers
                        )
            except FileNotFoundError as e:
                LOG.exception(e)
                dataset = create_dataset_from_meds_reader(
                    data_args, is_pretraining=True
                )
                if not data_args.streaming:
                    dataset.save_to_disk(meds_extension_path)
        else:
            # Load the dataset from the parquet files
            dataset = load_parquet_as_dataset(
                os.path.expanduser(data_args.data_folder),
                split="train",
                streaming=data_args.streaming,
            )
            # If streaming is enabled, we need to manually split the data into train/val
            if data_args.streaming and data_args.validation_split_num:
                dataset = dataset.shuffle(buffer_size=10_000, seed=training_args.seed)
                train_set = dataset.skip(data_args.validation_split_num)
                val_set = dataset.take(data_args.validation_split_num)
                dataset = DatasetDict({"train": train_set, "test": val_set})
            elif data_args.validation_split_percentage:
                dataset = dataset.train_test_split(
                    test_size=data_args.validation_split_percentage,
                    seed=training_args.seed,
                )
            else:
                raise RuntimeError(
                    f"Can not split the data. If streaming is enabled, validation_split_num needs to be "
                    f"defined, otherwise validation_split_percentage needs to be provided. "
                    f"The current values are:\n"
                    f"validation_split_percentage: {data_args.validation_split_percentage}\n"
                    f"validation_split_num: {data_args.validation_split_num}\n"
                    f"streaming: {data_args.streaming}"
                )

        # Create the CEHR-GPT tokenizer if it's not available in the output folder
        cehrgpt_tokenizer = load_and_create_tokenizer(
            data_args=data_args,
            model_args=model_args,
            cehrgpt_args=cehrgpt_args,
            dataset=dataset,
        )
        # Retrain the tokenizer in case we want to pretrain the model further using different datasets
        if cehrgpt_args.expand_tokenizer:
            tokenizer_output_path = os.path.expanduser(training_args.output_dir)
            if not tokenizer_exists(tokenizer_output_path):
                cehrgpt_tokenizer = CehrGptTokenizer.expand_trained_tokenizer(
                    cehrgpt_tokenizer=cehrgpt_tokenizer,
                    dataset=dataset["train"],
                    data_args=data_args,
                    concept_name_mapping={},
                    pretrained_concept_embedding_model=PretrainedEmbeddings(
                        cehrgpt_args.pretrained_embedding_path
                    ),
                )
                cehrgpt_tokenizer.save_pretrained(tokenizer_output_path)

        # sort the patient features chronologically and tokenize the data
        processed_dataset = create_cehrgpt_pretraining_dataset(
            dataset=dataset, cehrgpt_tokenizer=cehrgpt_tokenizer, data_args=data_args
        )
        # only save the data to the disk if it is not streaming
        if not data_args.streaming:
            processed_dataset.save_to_disk(prepared_ds_path)

    def filter_func(examples):
        if cehrgpt_args.drop_long_sequences:
            return [
                model_args.max_position_embeddings >= _ >= data_args.min_num_tokens
                for _ in examples["num_of_concepts"]
            ]
        else:
            return [_ >= data_args.min_num_tokens for _ in examples["num_of_concepts"]]

    # Create the args for batched filtering
    filter_args = {"batched": True, "batch_size": data_args.preprocessing_batch_size}
    # If the dataset is not in a streaming mode, we could add num_proc to enable parallelization
    if not data_args.streaming:
        filter_args["num_proc"] = data_args.preprocessing_num_workers

    # The filter can't be applied to a DatasetDict of IterableDataset (in case of streaming)
    # we need to iterate through all the datasets and apply the filter separately
    if isinstance(processed_dataset, DatasetDict) or isinstance(
        processed_dataset, IterableDatasetDict
    ):
        for key in processed_dataset.keys():
            processed_dataset[key] = processed_dataset[key].filter(
                filter_func, **filter_args
            )
    else:
        processed_dataset = processed_dataset.filter(filter_func, **filter_args)

    # If we choose to continue to train an existing model, we need to check whether the tokenizer exists in the
    # output_dir, and the tokenizer will be saved if it does not exist.
    if cehrgpt_args.continue_pretrain and not tokenizer_exists(
        os.path.expanduser(training_args.output_dir)
    ):
        cehrgpt_tokenizer.save_pretrained(os.path.expanduser(training_args.output_dir))

    model = load_and_create_model(
        model_args, cehrgpt_args, training_args, cehrgpt_tokenizer
    )

    # Expand tokenizer to adapt to the new pretraining dataset
    if model.config.vocab_size < cehrgpt_tokenizer.vocab_size:
        model.resize_token_embeddings(cehrgpt_tokenizer.vocab_size)
        # Update the pretrained embedding weights if they are available
        if model.config.use_pretrained_embeddings:
            model.cehrgpt.update_pretrained_embeddings(
                cehrgpt_tokenizer.pretrained_token_ids,
                cehrgpt_tokenizer.pretrained_embeddings,
            )
        elif cehrgpt_tokenizer.pretrained_token_ids:
            model.config.pretrained_embedding_dim = (
                cehrgpt_tokenizer.pretrained_embeddings.shape[1]
            )
            model.config.use_pretrained_embeddings = True
            model.cehrgpt.initialize_pretrained_embeddings()
            model.cehrgpt.update_pretrained_embeddings(
                cehrgpt_tokenizer.pretrained_token_ids,
                cehrgpt_tokenizer.pretrained_embeddings,
            )

    if model.config.add_cross_attention:
        # register cehrgpt in the huggingface model list
        register_cehrgpt_in_hf()
        model.config.add_cross_attention = True
        import polars as pl

        from cehrgpt.data.encoder_decoder.instruct_cehrgpt_dataset_collator import (
            InstructCehrGptDataCollator,
        )
        from cehrgpt.generation.cehrgpt_patient.clinical_statement_generator import (
            ClinicalStatementGenerator,
            ConditionDrugKnowledgeGraph,
        )
        from cehrgpt.models.encoder_decoder.instruct_hf_cehrgpt import (
            InstructCEHRGPTModel,
        )
        from cehrgpt.omop.vocab_utils import (
            create_drug_ingredient_to_brand_drug_map,
            generate_concept_maps,
        )

        if (
            not cehrgpt_args.encoder_model_name_or_path
            or not cehrgpt_args.encoder_tokenizer_name_or_path
        ):
            raise RuntimeError(
                "When add_cross_attention is set to True, "
                "encoder_model_name_or_path and encoder_tokenizer_name_or_path must be provided"
            )
        if not cehrgpt_args.knowledge_graph_path:
            raise RuntimeError(
                "When add_cross_attention is set to True, "
                "knowledge_graph_path must be provided"
            )
        if not cehrgpt_args.vocabulary_dir:
            raise RuntimeError(
                "When add_cross_attention is set to True, "
                "vocabulary_dir must be provided"
            )

        encoder_tokenizer = AutoTokenizer.from_pretrained(
            cehrgpt_args.encoder_tokenizer_name_or_path
        )
        encoder_tokenizer.save_pretrained(training_args.output_dir)
        encoder_model = AutoModel.from_pretrained(
            cehrgpt_args.encoder_model_name_or_path
        )
        setattr(
            encoder_model.config, "encoder_trainable", cehrgpt_args.encoder_trainable
        )
        model = InstructCEHRGPTModel.from_encoder_decoder_pretrained(
            encoder_model=encoder_model, decoder_model=model
        )
        with open(cehrgpt_args.knowledge_graph_path, "rb") as f:
            knowledge_graph = pickle.load(f)
        concept = pl.read_parquet(
            os.path.join(cehrgpt_args.vocabulary_dir, "concept", "*parquet")
        )
        concept_ancestor = pl.read_parquet(
            os.path.join(cehrgpt_args.vocabulary_dir, "concept_ancestor", "*parquet")
        )
        drug_ingredient_to_brand_drug_map = create_drug_ingredient_to_brand_drug_map(
            concept, concept_ancestor
        )
        concept_map, concept_domain = generate_concept_maps(concept)
        clinical_statement_generator = ClinicalStatementGenerator(
            condition_drug_knowledge_graph=ConditionDrugKnowledgeGraph(
                knowledge_graph=knowledge_graph,
                drug_ingredient_to_brand_drug_map=drug_ingredient_to_brand_drug_map,
            ),
            allowed_clinical_conditions=load_allowed_clinical_conditions(cehrgpt_args),
        )
        data_collator = InstructCehrGptDataCollator(
            clinical_statement_generator=clinical_statement_generator,
            concept_name_mapping=concept_map,
            concept_domain_mapping=concept_domain,
            encoder_tokenizer=encoder_tokenizer,
            tokenizer=cehrgpt_tokenizer,
            max_length=model_args.max_position_embeddings,
            shuffle_records=data_args.shuffle_records,
            include_ttv_prediction=model_args.include_ttv_prediction,
            use_sub_time_tokenization=model_args.use_sub_time_tokenization,
            include_values=model_args.include_values,
        )
    else:
        data_collator = CehrGptDataCollator(
            tokenizer=cehrgpt_tokenizer,
            max_length=model_args.max_position_embeddings,
            shuffle_records=data_args.shuffle_records,
            include_ttv_prediction=model_args.include_ttv_prediction,
            use_sub_time_tokenization=model_args.use_sub_time_tokenization,
            include_values=model_args.include_values,
        )

    # Detecting last checkpoint.
    last_checkpoint = get_last_hf_checkpoint(training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if not data_args.streaming:
        processed_dataset.set_format("pt")

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["test"],
        args=training_args,
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload
    metrics = train_result.metrics

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


def load_allowed_clinical_conditions(
    cehrgpt_args: CehrGPTArguments,
) -> Optional[List[int]]:
    try:
        with open(cehrgpt_args.allowed_clinical_conditions_path, "rb") as f:
            allowed_conditions = pickle.load(f)
            if not isinstance(allowed_conditions, list):
                allowed_conditions = None
            else:
                allowed_conditions = [
                    int(_)
                    for _ in allowed_conditions
                    if isinstance(int, _) or str.isnumeric(str(_))
                ]
    except Exception as e:
        LOG.warning(
            "The allowed encoder conditions cannot be loaded at %s.\n Setting allowed_conditions=None\n."
            "Caught exception: %s",
            cehrgpt_args.allowed_clinical_conditions_path,
            e,
        )
        allowed_conditions = None
    return allowed_conditions


if __name__ == "__main__":
    main()
