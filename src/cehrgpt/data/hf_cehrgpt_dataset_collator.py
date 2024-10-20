import random
from typing import Any, Dict

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from cehrgpt.gpt_utils import (
    extract_time_interval_in_days,
    is_att_token,
    is_inpatient_att_token,
    random_slice_gpt_sequence,
)
from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer

import dgl
import pickle

INPATIENT_STAY_DURATION_LIMIT = 30


class CehrGptDataCollator:
    def __init__(
        self,
        tokenizer: CehrGptTokenizer,
        max_length: int,
        shuffle_records: bool = False,
        include_values: bool = False,
        include_ttv_prediction: bool = False,
        use_sub_time_tokenization: bool = False,
        pretraining: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Pre-compute these so we can use them later on
        # We used VS for the historical data, currently, we use the new [VS] for the newer data
        # so we need to check both cases.
        self.vs_token_id = tokenizer._convert_token_to_id("VS")
        if self.vs_token_id == tokenizer._oov_token_id:
            self.vs_token_id = tokenizer._convert_token_to_id("[VS]")
        self.ve_token_id = tokenizer._convert_token_to_id("VE")
        if self.ve_token_id == tokenizer._oov_token_id:
            self.ve_token_id = tokenizer._convert_token_to_id("[VE]")

        self.shuffle_records = shuffle_records
        self.include_values = include_values
        self.include_ttv_prediction = include_ttv_prediction
        self.use_sub_time_tokenization = use_sub_time_tokenization
        self.pretraining = pretraining

        if self.use_sub_time_tokenization:
            token_to_time_token_mapping = tokenizer.token_to_time_token_mapping
            if not token_to_time_token_mapping:
                raise ValueError(
                    "The token_to_time_token_mapping in CehrGptTokenizer cannot be None "
                    "when use_sub_time_tokenization is enabled"
                )
            # Create the tensors for converting time tokens to the sub time tokens
            self.time_tokens = torch.tensor(
                list(tokenizer.token_to_time_token_mapping.keys()), dtype=torch.int64
            )
            self.mapped_sub_time_tokens = torch.tensor(
                list(token_to_time_token_mapping.values()), dtype=torch.int64
            )

        with open("/home/jason/workspace/cehr_gpt_graphs/hierarchy_knowledge_graph.pkl", 'rb') as f:
            kg = pickle.load(f)

        g = dgl.from_networkx(kg,
                            node_attrs=['concept_id'],
                            edge_attrs=['id', 'rel_type'])
        self.g = g
        self.concept_id_to_g_index_map = dict(zip(g.ndata['concept_id'].tolist(), g.nodes().tolist()))

        graph_concept_ids = list(map(str, self.g.ndata['concept_id'].tolist()))
        added_tokens = list(set(graph_concept_ids) - set(list(tokenizer._tokenizer.get_vocab().keys())))
        _ = tokenizer._tokenizer.add_tokens(added_tokens)
        self.g.ndata['vocab_id'] = torch.tensor([tokenizer._convert_token_to_id(token) for token in graph_concept_ids])

        # self.g = self.g.to('cuda')

    def _try_reverse_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.pretraining:
            return torch.flip(tensor, dims=[-1])
        return tensor

    @staticmethod
    def _convert_to_tensor(features: Any) -> torch.Tensor:
        if isinstance(features, torch.Tensor):
            return features
        else:
            return torch.tensor(features)

    @staticmethod
    def _convert_time_to_event(concept_ids):
        def default_value(c):
            try:
                if is_att_token(c):
                    time_to_visit = extract_time_interval_in_days(c)
                    if (
                        is_inpatient_att_token(c)
                        and time_to_visit > INPATIENT_STAY_DURATION_LIMIT
                    ):
                        return -100
                    return time_to_visit
                return -100
            except ValueError:
                return -100

        return [float(default_value(_)) for _ in concept_ids]

    def _connect_sequence_to_graph(self, g, input_ids):
        # g_merged = deepcopy(g)
        next_node_id = g.num_nodes()
        
        edges_to_add_u = []
        edges_to_add_v = []
        nodes_to_add = []

        sequence_indices = []
        agg_edge_ids_u = []
        agg_edge_ids_v = []
        
        for i, input_id in enumerate(input_ids):
            if input_id in g.ndata['vocab_id']:
                connected_node_id = int(torch.argwhere(g.ndata['vocab_id'] == input_id)[0].item())
                nodes_to_add.append(input_id)
                
                edges_to_add_u.append(next_node_id)
                edges_to_add_v.append(connected_node_id)
                
                sequence_indices.append(i)
                agg_edge_ids_u.append(next_node_id)
                agg_edge_ids_v.append(connected_node_id)

                next_node_id += 1
        
        nodes_to_add = torch.tensor(nodes_to_add).to(g.device)

        g = dgl.add_nodes(g, len(nodes_to_add), data={'vocab_id': nodes_to_add})
        g = dgl.add_edges(g, edges_to_add_u, edges_to_add_v)

        g = dgl.add_self_loop(g)


        return g, sequence_indices, (agg_edge_ids_u, agg_edge_ids_v)

    def _generate_batched_graph(self, input_ids, layers=3):
        connected_sequence_mask = torch.zeros_like(input_ids, dtype=torch.bool).to(input_ids.device)
        agg_edge_ids_src = torch.zeros_like(input_ids).long().to(input_ids.device)
        agg_edge_ids_dst = torch.zeros_like(input_ids).long().to(input_ids.device)

        induced_subgraphs = []

        cumulative_num_nodes = 0

        for batch in range(input_ids.shape[0]):
            
            batch_input_ids = input_ids[batch]
            flattened_batch_input_ids = torch.flatten(batch_input_ids)
            concept_ids = [self.tokenizer._convert_id_to_token(id) for id in flattened_batch_input_ids]
            concept_ids = [int(token) for token in concept_ids[4:] if token.isnumeric()]

            concept_node_ids = [self.concept_id_to_g_index_map[token] for token in concept_ids if token in self.concept_id_to_g_index_map]
            induced_subgraph, _ = dgl.khop_in_subgraph(self.g, concept_node_ids, layers)

            induced_subgraph = induced_subgraph.to(input_ids.device)

            is_input_id_in_graph = torch.isin(flattened_batch_input_ids, induced_subgraph.ndata['vocab_id'])
            input_ids_in_graph = torch.where(is_input_id_in_graph, flattened_batch_input_ids, torch.zeros_like(flattened_batch_input_ids))

            induced_subgraph, connected_sequence_indices, agg_edge_ids_batch = self._connect_sequence_to_graph(induced_subgraph, input_ids_in_graph)
            agg_edge_ids_src_batch, agg_edge_ids_dst_batch = agg_edge_ids_batch
            agg_edge_ids_src_batch, agg_edge_ids_dst_batch = torch.LongTensor(agg_edge_ids_src_batch).to(input_ids.device), torch.LongTensor(agg_edge_ids_dst_batch).to(input_ids.device)
            agg_edge_ids_src_batch, agg_edge_ids_dst_batch = agg_edge_ids_src_batch + cumulative_num_nodes, agg_edge_ids_dst_batch + cumulative_num_nodes 

            connected_sequence_mask[batch, connected_sequence_indices] = 1
            agg_edge_ids_src[batch, connected_sequence_indices] = agg_edge_ids_src_batch
            agg_edge_ids_dst[batch, connected_sequence_indices] = agg_edge_ids_dst_batch
            # induced_subgraph.ndata['orig_id'] = torch.arange(induced_subgraph.num_nodes())

            induced_subgraphs.append(induced_subgraph)
            cumulative_num_nodes += induced_subgraph.num_nodes()
        
        bg = dgl.batch(induced_subgraphs)

        return bg, connected_sequence_mask, agg_edge_ids_src, agg_edge_ids_dst


    def __call__(self, examples):
        examples = [self.generate_start_end_index(_) for _ in examples]
        examples = [self.random_sort(_) for _ in examples]
        batch = {}

        # Assume that each example in the batch is a dictionary with 'input_ids' and 'attention_mask'
        batch_input_ids = [
            self._try_reverse_tensor(self._convert_to_tensor(example["input_ids"]))
            for example in examples
        ]
        batch_attention_mask = [
            self._try_reverse_tensor(
                torch.ones_like(
                    self._convert_to_tensor(example["input_ids"]), dtype=torch.float
                )
            )
            for example in examples
        ]

        # Pad sequences to the max length in the batch
        batch["input_ids"] = self._try_reverse_tensor(
            pad_sequence(
                batch_input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
                padding_side='right'
            ).to(torch.int64)
        )

        batch["attention_mask"] = self._try_reverse_tensor(
            pad_sequence(batch_attention_mask, batch_first=True, padding_value=0.0, padding_side='right')
        )
        assert batch["input_ids"].shape[1] <= self.max_length
        assert batch["attention_mask"].shape[1] <= self.max_length

        if self.pretraining:
            batch["labels"] = batch["input_ids"].clone()

        if self.use_sub_time_tokenization:
            time_token_indicators = torch.isin(batch["input_ids"], self.time_tokens)
            masked_tokens = batch["input_ids"].clone()
            masked_tokens[~time_token_indicators] = -1
            # Get the index of the sub_time_tokens from the time_tokens tensor
            sub_time_token_indices = torch.argmax(
                (
                    masked_tokens.unsqueeze(-1)
                    == self.time_tokens.unsqueeze(0).unsqueeze(0)
                ).to(torch.int32),
                dim=-1,
            )
            sub_time_tokens = self.mapped_sub_time_tokens[sub_time_token_indices]
            batch["time_token_indicators"] = time_token_indicators
            batch["sub_time_tokens"] = sub_time_tokens

        if self.include_ttv_prediction:
            batch_time_to_visits = [
                self._try_reverse_tensor(
                    self._convert_to_tensor(example["time_to_visits"])
                )
                for example in examples
            ]
            batch["time_to_visits"] = self._try_reverse_tensor(
                pad_sequence(
                    batch_time_to_visits, batch_first=True, padding_value=-100.0,
                    padding_side='right'
                )
            )

        if self.include_values:
            batch_value_indicators = [
                self._try_reverse_tensor(
                    self._convert_to_tensor(example["value_indicators"])
                )
                for example in examples
            ]
            batch_values = [
                self._try_reverse_tensor(self._convert_to_tensor(example["values"]))
                for example in examples
            ]

            batch["value_indicators"] = self._try_reverse_tensor(
                pad_sequence(
                    batch_value_indicators, batch_first=True, padding_value=False,
                    padding_side='right'
                )
            )
            batch["values"] = self._try_reverse_tensor(
                pad_sequence(batch_values, batch_first=True, padding_value=-1.0, padding_side='right')
            )

            assert batch["value_indicators"].shape[1] <= self.max_length
            assert batch["values"].shape[1] <= self.max_length
            batch["true_value_indicators"] = batch["value_indicators"].clone()
            batch["true_values"] = batch["values"].clone()

        if "person_id" in examples[0]:
            batch["person_id"] = torch.cat(
                [
                    self._convert_to_tensor(example["person_id"]).reshape(-1, 1)
                    for example in examples
                ],
                dim=0,
            ).to(torch.int32)

        if "index_date" in examples[0]:
            batch["index_date"] = torch.cat(
                [
                    self._convert_to_tensor(example["index_date"]).reshape(-1, 1)
                    for example in examples
                ],
                dim=0,
            ).to(torch.float32)

        if "age_at_index" in examples[0]:
            batch["age_at_index"] = torch.cat(
                [
                    self._convert_to_tensor(example["age_at_index"]).reshape(-1, 1)
                    for example in examples
                ],
                dim=0,
            ).to(torch.float32)

        if "classifier_label" in examples[0]:
            batch["classifier_label"] = torch.cat(
                [
                    self._convert_to_tensor(example["classifier_label"]).reshape(-1, 1)
                    for example in examples
                ],
                dim=0,
            ).to(torch.float32)

        batched_graph, connected_sequence_mask, agg_edge_ids_src, agg_edge_ids_dst = self._generate_batched_graph(batch["input_ids"], layers=4)

        # batch["input_ids"] = batch["input_ids"].to('cuda')
        batch["bg"] = batched_graph.to(batch["input_ids"].device)
        batch["connected_sequence_mask"] = connected_sequence_mask.to(batch["input_ids"].device)
        batch["agg_edge_ids_src"] = agg_edge_ids_src.to(batch["input_ids"].device)
        batch["agg_edge_ids_dst"] = agg_edge_ids_dst.to(batch["input_ids"].device)

        return batch

    def random_sort(self, record: Dict[str, Any]) -> Dict[str, Any]:

        if not self.shuffle_records:
            return record

        if "record_ranks" not in record:
            return record

        sorting_column = record["record_ranks"]
        random_order = np.random.rand(len(sorting_column))

        if self.include_values:
            iterator = zip(
                sorting_column,
                random_order,
                record["input_ids"],
                record["value_indicators"],
                record["values"],
            )
            sorted_list = sorted(iterator, key=lambda tup2: (tup2[0], tup2[1], tup2[2]))
            _, _, sorted_input_ids, sorted_value_indicators, sorted_values = zip(
                *list(sorted_list)
            )
            record["input_ids"] = self._convert_to_tensor(sorted_input_ids)
            record["value_indicators"] = self._convert_to_tensor(
                sorted_value_indicators
            )
            record["values"] = self._convert_to_tensor(sorted_values)
        else:
            iterator = zip(sorting_column, random_order, record["input_ids"])
            sorted_list = sorted(iterator, key=lambda tup2: (tup2[0], tup2[1], tup2[2]))
            _, _, sorted_input_ids = zip(*list(sorted_list))
            record["input_ids"] = self._convert_to_tensor(sorted_input_ids)
        return record

    def generate_start_end_index(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Adding the start and end indices to extract a portion of the patient sequence."""
        # concept_ids will be used to for time to event predictions and identifying the visit starts

        input_ids = record["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.detach().tolist()
        concept_ids = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        seq_length = len(record["input_ids"])
        new_max_length = self.max_length - 1  # Subtract one for the [END] token

        # Return the record directly if the actual sequence length is less than the max sequence
        if seq_length <= new_max_length:
            record["input_ids"] = torch.concat(
                [
                    self._convert_to_tensor(record["input_ids"]),
                    self._convert_to_tensor([self.tokenizer.end_token_id]),
                ]
            )
            if self.include_values:
                record["value_indicators"] = torch.concat(
                    [
                        self._convert_to_tensor(record["value_indicators"]),
                        self._convert_to_tensor([0]),
                    ]
                ).to(torch.bool)
                record["values"] = torch.concat(
                    [
                        self._convert_to_tensor(record["values"]),
                        self._convert_to_tensor([-1.0]),
                    ]
                )
            if self.include_ttv_prediction:
                record["time_to_visits"] = torch.concat(
                    [
                        self._convert_to_tensor(
                            self._convert_time_to_event(concept_ids)
                        ),
                        self._convert_to_tensor([-100.0]),
                    ]
                )

            return record

        if self.pretraining:
            # There is a 50% chance we randomly slice out a portion of the patient history and update the demographic
            # prompt depending on the new starting point
            if random.random() < 0.5:
                start_index, end_index, demographic_tokens = random_slice_gpt_sequence(
                    concept_ids, new_max_length
                )
                if start_index != end_index:
                    record["input_ids"] = self._convert_to_tensor(
                        record["input_ids"][start_index : end_index + 1]
                    )
                    if self.include_values:
                        record["value_indicators"] = self._convert_to_tensor(
                            record["value_indicators"][start_index : end_index + 1]
                        ).to(torch.bool)
                        record["values"] = self._convert_to_tensor(
                            record["values"][start_index : end_index + 1]
                        )
                    if self.include_ttv_prediction:
                        record["time_to_visits"] = self._convert_to_tensor(
                            self._convert_time_to_event(
                                concept_ids[start_index : end_index + 1]
                            )
                        )
                    return record

            # The default employs a right truncation strategy, where the demographic prompt is reserved
            end_index = new_max_length
            for i in reversed(list(range(0, end_index))):
                current_token = record["input_ids"][i]
                if current_token == self.ve_token_id:
                    end_index = i
                    break

            record["input_ids"] = record["input_ids"][0:end_index]
            if self.include_values:
                record["value_indicators"] = self._convert_to_tensor(
                    record["value_indicators"][0:end_index]
                ).to(torch.bool)
                record["values"] = self._convert_to_tensor(
                    record["values"][0:end_index]
                )
            if self.include_ttv_prediction:
                record["time_to_visits"] = self._convert_to_tensor(
                    self._convert_time_to_event(concept_ids[0:end_index])
                )
            return record

        else:
            # We employ a left truncation strategy, where the most recent patient history is reserved for fine-tuning
            start_index = seq_length - new_max_length
            end_index = seq_length
            for i in range(start_index, end_index):
                current_token = record["input_ids"][i]
                if current_token == self.vs_token_id:
                    start_index = i
                    break
            record["input_ids"] = record["input_ids"][start_index:end_index]
            if self.include_values:
                record["value_indicators"] = self._convert_to_tensor(
                    record["value_indicators"][start_index:end_index]
                ).to(torch.bool)
                record["values"] = self._convert_to_tensor(
                    record["values"][start_index:end_index]
                )
            if self.include_ttv_prediction:
                record["time_to_visits"] = self._convert_to_tensor(
                    self._convert_time_to_event(concept_ids[start_index:end_index])
                )
            return record
