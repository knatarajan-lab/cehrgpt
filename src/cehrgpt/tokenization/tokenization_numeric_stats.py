from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from cehrgpt.tokenization.tokenization_bin_utils import create_value_bin, is_valid_valid_bin
from cehrgpt.tokenization.tokenization_constants import NA, UNKNOWN_BIN
from cehrgpt.tokenization.tokenization_statistics import create_numeric_concept_unit_mapping


class NumericEventStatistics:
    def __init__(self, lab_stats: List[Dict[str, Any]]):
        self._lab_stats = lab_stats
        self._lab_stats_mapping = {
            (lab_stat["concept_id"], lab_stat["unit"]): {
                "unit": lab_stat["unit"],
                "mean": lab_stat["mean"],
                "std": lab_stat["std"],
                "value_outlier_std": lab_stat["value_outlier_std"],
                "bins": lab_stat["bins"],
            }
            for lab_stat in lab_stats
        }
        self._concept_prob_mapping, self._concept_unit_mapping = (
            create_numeric_concept_unit_mapping(lab_stats)
        )

    def get_numeric_concept_ids(self) -> List[str]:
        return [_["concept_id"] for _ in self._lab_stats]

    def get_random_unit(self, concept_id: str) -> str:
        if concept_id in self._concept_prob_mapping:
            unit_probs = self._concept_prob_mapping[concept_id]
            return np.random.choice(
                self._concept_unit_mapping[concept_id], p=unit_probs
            )
        return NA

    def normalize(
            self, concept_id: str, unit: str, concept_value: Union[float, str]
    ) -> str:
        if isinstance(concept_value, float):
            if (concept_id, unit) in self._lab_stats_mapping:
                concept_unit_stats = self._lab_stats_mapping[(concept_id, unit)]
                bins = concept_unit_stats["bins"]
                if bins:
                    for each_bin in bins:
                        if (
                                each_bin["start_val"]
                                <= concept_value
                                <= each_bin["end_val"]
                        ):
                            return create_value_bin(each_bin["bin_index"])
        return UNKNOWN_BIN

    def denormalize(
            self, concept_id: str, value_bin: str
    ) -> Tuple[Optional[Union[float, str]], str]:
        unit = self.get_random_unit(concept_id)
        concept_value = value_bin
        if (
                is_valid_valid_bin(value_bin)
                and (concept_id, unit) in self._lab_stats_mapping
        ):
            lab_stats = self._lab_stats_mapping[(concept_id, unit)]
            bin_index = value_bin.split(":")[1]
            if bin_index.isnumeric():
                bin_index = int(bin_index)
                # There are rare cases during sequence generation where bin_index could be out of range
                # when there are no bins for (concept_id, unit) due to the small number of values in the source data
                if len(lab_stats["bins"]) > bin_index:
                    assert bin_index == lab_stats["bins"][bin_index]["bin_index"]
                    bin_spline = lab_stats["bins"][bin_index]["spline"]
                    x = np.random.uniform(
                        bin_spline.get_knots()[0], bin_spline.get_knots()[-1]
                    )
                    concept_value = bin_spline(x).item()
        return concept_value, unit
