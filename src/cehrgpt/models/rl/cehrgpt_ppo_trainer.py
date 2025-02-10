import time
from typing import List, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from trl.core import (
    WANDB_PADDING,
    PPODecorators,
    convert_to_scalar,
    logprobs_from_logits,
    stack_dicts,
    stats_to_np,
)
from trl.trainer import PPOTrainer

from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer


class CehrGptPPODataCollator:
    def __init__(self, cehrgpt_tokenizer: CehrGptTokenizer, max_length: int):
        self.cehrgpt_tokenizer = cehrgpt_tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        batch = {}
        # Pad sequences to the max length in the batch
        batch["input_ids"] = pad_sequence(
            [example["input_ids"] for example in examples],
            batch_first=True,
            padding_value=self.cehrgpt_tokenizer.pad_token_id,
        ).to(torch.int64)

        batch["attention_mask"] = pad_sequence(
            [example["attention_mask"] for example in examples],
            batch_first=True,
            padding_value=0.0,
        )

        assert (
            batch["input_ids"].shape[1] <= self.max_length
        ), f"Invalid input_ids length: {batch['input_ids'].shape[1]}"

        if "value_indicators" in examples[0]:
            batch["value_indicators"] = pad_sequence(
                [example["value_indicators"] for example in examples],
                batch_first=True,
                padding_value=False,
            )

        if "values" in examples[0]:
            batch["values"] = pad_sequence(
                [example["values"] for example in examples],
                batch_first=True,
                padding_value=self.cehrgpt_tokenizer.pad_value_token_id,
            )
            assert batch["value_indicators"].shape[1] <= self.max_length
            assert batch["values"].shape[1] <= self.max_length

        return batch


class CehrGptPPOTrainer(PPOTrainer):
    def _step_safety_checker(
        self,
        batch_size: int,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
        values: List[torch.Tensor] = None,
        value_indicators: List[torch.BoolTensor] = None,
        masks: Optional[List[torch.LongTensor]] = None,
    ):
        """
        Check if the input data is valid for training.

        Args:
            batch_size (int):
                Batch size from the config file.
            queries (List[`torch.LongTensor`]):
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`torch.LongTensor`]):
                List of tensors containing the encoded responses of shape (`response_length`)
            scores (List[`torch.FloatTensor`]):
                List of tensors containing the scores.
            masks (List[`torch.LongTensor`], *optional*):
                list of optional tensors containing the masks of shape (`response_length`)

        Returns:
            `tuple`: The input processed data.
        """
        for name, tensor_list in zip(
            ["queries", "responses", "scores", "values", "value_indicators"],
            [queries, responses, scores, values, value_indicators],
        ):
            if not isinstance(tensor_list, list):
                raise ValueError(
                    f"{name} must be a list of tensors - got {type(tensor_list)}"
                )
            if not isinstance(tensor_list[0], torch.Tensor):
                raise ValueError(
                    f"Elements in {name} must be tensors - got {type(tensor_list[0])}"
                )
            if batch_size is not None and len(tensor_list) != batch_size:
                raise ValueError(
                    f"Batch size ({batch_size}) does not match number of examples - but got {len(tensor_list)} for: {name}"
                )

        # add queries, scores and responses on the correct device
        queries = [tensor.to(self.current_device) for tensor in queries]
        responses = [tensor.to(self.current_device) for tensor in responses]
        scores = [tensor.to(self.current_device) for tensor in scores]
        masks = (
            [tensor.to(self.current_device) for tensor in masks]
            if masks is not None
            else None
        )
        values = (
            [tensor.to(self.current_device) for tensor in values]
            if values is not None
            else None
        )
        value_indicators = (
            [tensor.to(self.current_device) for tensor in value_indicators]
            if value_indicators is not None
            else None
        )

        # squeeze scores if needed
        for i, score in enumerate(scores):
            if score.dim() > 1:
                raise ValueError(
                    f"Scores must be 1-dimensional - got {score.dim()} for {score}"
                )
            elif score.dim() == 1:
                scores[i] = score.squeeze()

        return queries, responses, scores, values, value_indicators, masks

    @PPODecorators.empty_device_cache()
    def step(
        self,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
        values: List[torch.Tensor] = None,
        value_indicators: List[torch.BoolTensor] = None,
        response_masks: Optional[List[torch.LongTensor]] = None,
    ):

        bs = self.config.batch_size

        queries, responses, scores, values, value_indicators, response_masks = (
            self._step_safety_checker(
                bs, queries, responses, scores, values, value_indicators, response_masks
            )
        )
        scores = torch.tensor(scores, device=self.current_device)
        if self.config.use_score_scaling:
            # Score scaling
            scores_mean, scores_std = self.running.update(scores)
            score_scaling_factor = scores_std + torch.finfo(scores.dtype).eps
            if self.config.use_score_norm:
                scores = (scores - scores_mean) / score_scaling_factor
            else:
                scores /= score_scaling_factor

        if self.config.score_clip is not None:
            # Score clipping
            scores_dtype = scores.dtype
            scores = torch.clip(
                scores.float(), -self.config.score_clip, self.config.score_clip
            ).to(dtype=scores_dtype)

        # if we want to push best model to the hub
        if hasattr(self, "highest_reward"):
            if self.compare_step % self.config.compare_steps == 0:
                curr_mean_reward = scores.mean()
                # if the best reward ever seen
                if curr_mean_reward > self.highest_reward:
                    self.highest_reward = curr_mean_reward
                    # push model to hub
                    self.push_to_hub(**self.push_to_hub_kwargs)
            self.compare_step += 1

        timing = dict()
        t0 = time.time()

        t = time.time()

        model_inputs = self.prepare_model_inputs(
            queries, responses, values=values, value_indicators=value_indicators
        )

        if self.is_distributed:
            pad_first = self.tokenizer.padding_side == "left"

            model_inputs["input_ids"] = self.accelerator.pad_across_processes(
                model_inputs["input_ids"],
                dim=1,
                pad_index=self.tokenizer.pad_token_id,
                pad_first=pad_first,
            )
            model_inputs["attention_mask"] = self.accelerator.pad_across_processes(
                model_inputs["attention_mask"], dim=1, pad_index=0, pad_first=pad_first
            )
            if values is not None:
                model_inputs["values"] = self.accelerator.pad_across_processes(
                    model_inputs["values"],
                    dim=1,
                    pad_index=self.tokenizer.pad_value_token_id,
                    pad_first=pad_first,
                )
            if value_indicators is not None:
                model_inputs["value_indicators"] = (
                    self.accelerator.pad_across_processes(
                        model_inputs["value_indicators"],
                        dim=1,
                        pad_index=False,
                        pad_first=pad_first,
                    )
                )
            if self.is_encoder_decoder:
                model_inputs["decoder_input_ids"] = (
                    self.accelerator.pad_across_processes(
                        model_inputs["decoder_input_ids"],
                        dim=1,
                        pad_index=self.tokenizer.pad_token_id,
                        pad_first=pad_first,
                    )
                )
                model_inputs["decoder_attention_mask"] = (
                    self.accelerator.pad_across_processes(
                        model_inputs["decoder_attention_mask"],
                        dim=1,
                        pad_index=0,
                        pad_first=pad_first,
                    )
                )

        model_inputs_names = list(model_inputs.keys())

        full_kl_penalty = self.config.kl_penalty == "full"

        with torch.no_grad():
            all_logprobs, logits_or_none, states_values, masks = (
                self.batched_forward_pass(
                    self.model,
                    queries,
                    responses,
                    model_inputs,
                    response_masks=response_masks,
                    return_logits=full_kl_penalty,
                )
            )
            with self.optional_peft_ctx():
                ref_logprobs, ref_logits_or_none, _, _ = self.batched_forward_pass(
                    self.model if self.is_peft_model else self.ref_model,
                    queries,
                    responses,
                    model_inputs,
                    return_logits=full_kl_penalty,
                )

        timing["time/ppo/forward_pass"] = time.time() - t

        with torch.no_grad():
            t = time.time()
            if full_kl_penalty:
                active_full_logprobs = logprobs_from_logits(
                    logits_or_none, None, gather=False
                )
                ref_full_logprobs = logprobs_from_logits(
                    ref_logits_or_none, None, gather=False
                )

                rewards, non_score_reward, kls = self.compute_rewards(
                    scores, active_full_logprobs, ref_full_logprobs, masks
                )
            else:
                rewards, non_score_reward, kls = self.compute_rewards(
                    scores, all_logprobs, ref_logprobs, masks
                )
            timing["time/ppo/compute_rewards"] = time.time() - t

            t = time.time()
            states_values, advantages, returns = self.compute_advantages(
                states_values, rewards, masks
            )
            timing["time/ppo/compute_advantages"] = time.time() - t

        # upcast to float32 to avoid dataset issues
        batch_dict = {
            "queries": queries,
            "responses": responses,
            "logprobs": all_logprobs.to(torch.float32),
            "states_values": states_values.to(torch.float32),
            "masks": masks,
            "advantages": advantages,
            "returns": returns,
        }
        batch_dict.update(model_inputs)

        t = time.time()
        all_stats = []
        early_stop = False
        for _ in range(self.config.ppo_epochs):
            if early_stop:
                break
            b_inds = np.random.permutation(bs)
            for backward_batch_start in range(0, bs, self.config.backward_batch_size):
                backward_batch_end = (
                    backward_batch_start + self.config.backward_batch_size
                )
                backward_batch_inds = b_inds[backward_batch_start:backward_batch_end]

                for mini_batch_start in range(
                    0, self.config.backward_batch_size, self.config.mini_batch_size
                ):
                    mini_batch_end = mini_batch_start + self.config.mini_batch_size
                    mini_batch_inds = backward_batch_inds[
                        mini_batch_start:mini_batch_end
                    ]
                    mini_batch_dict = {
                        "logprobs": batch_dict["logprobs"][mini_batch_inds],
                        "states_values": batch_dict["states_values"][mini_batch_inds],
                        "masks": batch_dict["masks"][mini_batch_inds],
                        # hacks: the queries and responses are ragged.
                        "queries": [batch_dict["queries"][i] for i in mini_batch_inds],
                        "responses": [
                            batch_dict["responses"][i] for i in mini_batch_inds
                        ],
                        "advantages": batch_dict["advantages"][mini_batch_inds],
                        "returns": batch_dict["returns"][mini_batch_inds],
                    }
                    for k in model_inputs_names:
                        mini_batch_dict[k] = batch_dict[k][mini_batch_inds]
                    with self.accelerator.accumulate(self.model):
                        model_inputs = {
                            k: mini_batch_dict[k] for k in model_inputs_names
                        }

                        logprobs, logits, vpreds, _ = self.batched_forward_pass(
                            self.model,
                            mini_batch_dict["queries"],
                            mini_batch_dict["responses"],
                            model_inputs,
                            return_logits=True,
                        )
                        train_stats = self.train_minibatch(
                            mini_batch_dict["logprobs"],
                            mini_batch_dict["states_values"],
                            logprobs,
                            logits,
                            vpreds,
                            mini_batch_dict["masks"],
                            mini_batch_dict["advantages"],
                            mini_batch_dict["returns"],
                        )
                        all_stats.append(train_stats)

            # typically, early stopping is done at the epoch level
            if self.config.early_stopping:
                policykl = train_stats["policy/policykl"]
                early_stop = self._early_stop(policykl)
                if early_stop:
                    break

        timing["time/ppo/optimize_step"] = time.time() - t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats["policy/advantages"] = torch.flatten(
            train_stats["policy/advantages"]
        ).unsqueeze(0)
        train_stats["policy/advantages"] = torch.nan_to_num(
            train_stats["policy/advantages"], WANDB_PADDING
        )
        train_stats["policy/ratio"] = torch.flatten(
            train_stats["policy/ratio"]
        ).unsqueeze(0)

        stats = self.record_step_stats(
            scores=scores,
            logprobs=all_logprobs,
            ref_logprobs=ref_logprobs,
            non_score_reward=non_score_reward,
            train_stats=train_stats,
            kl_coef=self.kl_ctl.value,
            masks=masks,
            queries=queries,
            responses=responses,
            kls=kls,
        )
        # Gather/Reduce stats from all processes
        if self.is_distributed:
            stats = self.gather_stats(stats)
        stats = stats_to_np(stats)
        timing["time/ppo/calc_stats"] = time.time() - t
        stats["ppo/learning_rate"] = self.optimizer.param_groups[0]["lr"]

        # Update the KL control - multiply the batch_size by the number of processes
        self.kl_ctl.update(
            stats["objective/kl"],
            self.config.batch_size * self.accelerator.num_processes,
        )

        # Log the total ppo time
        timing["time/ppo/total"] = time.time() - t0
        stats.update(timing)

        # post-process stats for tensorboard and other loggers
        if self.config.log_with != "wandb":
            stats = convert_to_scalar(stats)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return stats

    def prepare_model_inputs(
        self, queries: torch.Tensor, responses: torch.Tensor, **kargs
    ):
        batched_values: torch.LongTensor = kargs.get("values")
        batched_value_indicators: torch.LongTensor = kargs.get("value_indicators")
        if self.is_encoder_decoder:
            input_data = self.data_collator(
                [
                    {
                        "input_ids": query,
                        "attention_mask": torch.ones_like(query),
                        "decoder_input_ids": response,
                        "decoder_attention_mask": torch.ones_like(response),
                        "values": values,
                        "value_indicators": value_indicators,
                    }
                    for query, response, values, value_indicators in zip(
                        queries, responses, batched_values, batched_value_indicators
                    )
                ]
            )
        else:
            input_ids = [torch.cat([q, r]) for q, r in zip(queries, responses)]
            input_data = self.data_collator(
                [
                    {
                        "input_ids": ids,
                        "attention_mask": torch.ones_like(ids),
                        "values": values,
                        "value_indicators": value_indicators,
                    }
                    for ids, values, value_indicators in zip(
                        input_ids, batched_values, batched_value_indicators
                    )
                ]
            )
        input_data.pop("labels", None)  # we don't want to compute LM losses
        return input_data
