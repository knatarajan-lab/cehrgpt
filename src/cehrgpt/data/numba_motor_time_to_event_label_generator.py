from typing import Any, Dict

import numpy as np
from numba import njit, prange

from cehrgpt.models.tokenization_hf_cehrgpt import CehrGptTokenizer


@njit(cache=True, fastmath=True, inline="always")
def process_valid_entries_inline(motor_censor_times):
    """Inline filtering of valid entries."""
    count = 0
    for i in range(len(motor_censor_times)):
        if motor_censor_times[i] != -100:
            count += 1

    if count == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)

    valid_indices = np.empty(count, dtype=np.int32)
    valid_censor_times = np.empty(count, dtype=np.float32)

    idx = 0
    for i in range(len(motor_censor_times)):
        if motor_censor_times[i] != -100:
            valid_indices[idx] = i
            valid_censor_times[idx] = motor_censor_times[i]
            idx += 1

    return valid_indices, valid_censor_times


@njit(cache=True, fastmath=True)
def compute_time_bins_batch_optimized(
    time_vectors, event_indicators, motor_time_bins, vocab_size
):
    """
    Radically optimized batch processing with minimal memory allocation.

    Key optimizations:
    1. Process entire vocab dimension at once per prediction
    2. Vectorized log2 computation where possible
    3. Minimal branching in inner loops
    4. Pre-computed constants
    """
    n_predictions, _ = time_vectors.shape
    n_bins = len(motor_time_bins) - 1

    # Pre-allocate with proper initialization
    motor_tte_time = np.full(
        (n_predictions, n_bins, vocab_size), -np.inf, dtype=np.float32
    )
    motor_tte_event_indicator = np.zeros(
        (n_predictions, n_bins, vocab_size), dtype=np.bool_
    )
    motor_tte_mask = np.zeros((n_predictions, n_bins, vocab_size), dtype=np.bool_)

    # Pre-compute all bin boundaries and widths
    bin_starts = motor_time_bins[:-1]
    bin_ends = motor_time_bins[1:]

    # Process each prediction independently for better cache locality
    for pred_idx in range(n_predictions):
        # Get the time and event vectors for this prediction
        pred_times = time_vectors[pred_idx, :]
        pred_events = event_indicators[pred_idx, :]

        # For each vocab token, compute all bins at once
        for vocab_idx in range(vocab_size):
            time_val = pred_times[vocab_idx]
            event_occurred = pred_events[vocab_idx]

            # Process bins in order - early termination when time_val < bin_start
            for bin_idx in range(n_bins):
                bin_start = bin_starts[bin_idx]
                bin_end = bin_ends[bin_idx]

                if time_val < bin_start:
                    break  # All remaining bins will also be < bin_start

                # Calculate time in bin
                time_in_bin = min(time_val - bin_start, bin_end - bin_start)

                if time_in_bin > 1e-10:
                    motor_tte_time[pred_idx, bin_idx, vocab_idx] = np.log2(time_in_bin)
                    motor_tte_mask[pred_idx, bin_idx, vocab_idx] = True

                    # Event indicator: event occurred AND time is within this bin
                    if event_occurred and time_val < bin_end:
                        motor_tte_event_indicator[pred_idx, bin_idx, vocab_idx] = True

    return motor_tte_time, motor_tte_event_indicator, motor_tte_mask


@njit(cache=True, fastmath=True)
def vectorized_assignment_optimized(
    valid_censor_times,
    vocab_size,
    valid_indices,
    motor_tte_label_offsets,
    all_tte_tasks,
    all_tte_times,
):
    """Ultra-fast vectorized assignment with minimal overhead."""
    n_predictions = len(valid_censor_times)

    # Initialize time vectors with broadcast equivalent
    time_vectors = np.empty((n_predictions, vocab_size), dtype=np.float32)
    for i in range(n_predictions):
        censor_time = valid_censor_times[i]
        for j in range(vocab_size):
            time_vectors[i, j] = censor_time

    event_indicators = np.zeros((n_predictions, vocab_size), dtype=np.bool_)

    # Direct assignment without intermediate arrays
    for local_idx in range(n_predictions):
        global_idx = valid_indices[local_idx]
        if global_idx < len(motor_tte_label_offsets) - 1:
            start_idx = motor_tte_label_offsets[global_idx]
            end_idx = motor_tte_label_offsets[global_idx + 1]

            for task_idx in range(start_idx, min(end_idx, len(all_tte_tasks))):
                vocab_token = all_tte_tasks[task_idx]
                if 0 <= vocab_token < vocab_size:
                    time_vectors[local_idx, vocab_token] = all_tte_times[task_idx]
                    event_indicators[local_idx, vocab_token] = True

    return time_vectors, event_indicators


@njit(cache=True, fastmath=True)
def compute_time_bins_memory_optimized(
    time_vectors, event_indicators, motor_time_bins, vocab_size
):
    """
    Memory-optimized version that processes one prediction at a time.

    Reduces memory pressure for very large datasets.
    """
    n_predictions, _ = time_vectors.shape
    n_bins = len(motor_time_bins) - 1

    # Allocate output arrays
    motor_tte_time = np.full(
        (n_predictions, n_bins, vocab_size), -np.inf, dtype=np.float32
    )
    motor_tte_event_indicator = np.zeros(
        (n_predictions, n_bins, vocab_size), dtype=np.bool_
    )
    motor_tte_mask = np.zeros((n_predictions, n_bins, vocab_size), dtype=np.bool_)

    # Pre-compute constants
    np.log2(1e-10)

    # Process one prediction at a time to minimize memory access patterns
    for pred_idx in range(n_predictions):
        # Cache prediction data
        pred_time_row = time_vectors[pred_idx, :]
        pred_event_row = event_indicators[pred_idx, :]

        # Process vocab tokens in chunks for better cache performance
        chunk_size = min(64, vocab_size)  # Process 64 tokens at a time

        for vocab_start in range(0, vocab_size, chunk_size):
            vocab_end = min(vocab_start + chunk_size, vocab_size)

            for vocab_idx in range(vocab_start, vocab_end):
                time_val = pred_time_row[vocab_idx]
                event_occurred = pred_event_row[vocab_idx]

                # Find first relevant bin (binary search could be used here for many bins)
                first_bin = 0
                for bin_idx in range(n_bins):
                    if time_val >= motor_time_bins[bin_idx]:
                        first_bin = bin_idx
                    else:
                        break

                # Process only relevant bins
                for bin_idx in range(first_bin, n_bins):
                    bin_start = motor_time_bins[bin_idx]
                    bin_end = motor_time_bins[bin_idx + 1]

                    if time_val < bin_start:
                        break

                    time_in_bin = min(time_val - bin_start, bin_end - bin_start)

                    if time_in_bin > 1e-10:
                        motor_tte_time[pred_idx, bin_idx, vocab_idx] = np.log2(
                            time_in_bin
                        )
                        motor_tte_mask[pred_idx, bin_idx, vocab_idx] = True

                        if event_occurred and time_val < bin_end:
                            motor_tte_event_indicator[pred_idx, bin_idx, vocab_idx] = (
                                True
                            )

    return motor_tte_time, motor_tte_event_indicator, motor_tte_mask


class ExtremelyOptimizedTransformation:
    """Extremely optimized transformation targeting sub-second performance."""

    def __init__(self, tokenizer: CehrGptTokenizer, motor_num_time_pieces: int):
        self.tokenizer = tokenizer
        self.motor_time_bins = np.array(
            tokenizer.get_motor_time_bins(motor_num_time_pieces), dtype=np.float64
        )
        self.motor_num_time_pieces = motor_num_time_pieces

        # Minimal warmup
        self._fast_warmup()

    def _fast_warmup(self):
        """Fast warmup with minimal overhead."""
        # Create minimal test data
        test_censor = np.array([10.0, 20.0], dtype=np.float32)
        test_offsets = np.array([0, 1, 1], dtype=np.int32)
        test_tasks = np.array([5], dtype=np.int32)
        test_times = np.array([15.0], dtype=np.float32)

        # Warm up key functions
        valid_idx, valid_censor = process_valid_entries_inline(test_censor)
        if len(valid_idx) > 0:
            time_vec, event_ind = vectorized_assignment_optimized(
                valid_censor, 50, valid_idx, test_offsets, test_tasks, test_times
            )
            compute_time_bins_batch_optimized(
                time_vec, event_ind, self.motor_time_bins, 50
            )
            compute_time_bins_memory_optimized(
                time_vec, event_ind, self.motor_time_bins, 50
            )

    def create_time_to_event_labels_extreme(
        self, record: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extremely optimized version targeting sub-second performance."""
        motor_tte_label_offsets = np.array(
            record["motor_tte_label_offsets"], dtype=np.int32
        )
        motor_censor_times = np.array(record["motor_censor_times"], dtype=np.float32)

        # Fast filtering
        valid_indices, valid_censor_times = process_valid_entries_inline(
            motor_censor_times
        )

        if len(valid_indices) == 0:
            empty_shape = (
                0,
                self.motor_num_time_pieces,
                self.tokenizer.motor_tte_vocab_size,
            )
            record["motor_tte_times"] = np.zeros(empty_shape, dtype=np.float32)
            record["motor_tte_event_indicators"] = np.zeros(empty_shape, dtype=bool)
            record["motor_tte_masks"] = np.zeros(empty_shape, dtype=bool)
            return record

        vocab_size = self.tokenizer.motor_tte_vocab_size
        all_tte_tasks = np.array(record["motor_tte_tasks"], dtype=np.int32)
        all_tte_times = np.array(record["motor_tte_times"], dtype=np.float32)

        # Combined initialization and assignment
        time_vectors, event_indicators = vectorized_assignment_optimized(
            valid_censor_times,
            vocab_size,
            valid_indices,
            motor_tte_label_offsets,
            all_tte_tasks,
            all_tte_times,
        )

        # Choose processing strategy based on data characteristics
        n_predictions = len(valid_indices)
        total_size = n_predictions * vocab_size * len(self.motor_time_bins)

        if total_size > 1000000:  # Very large datasets
            motor_tte_time, motor_tte_event_indicator, motor_tte_mask = (
                compute_time_bins_memory_optimized(
                    time_vectors, event_indicators, self.motor_time_bins, vocab_size
                )
            )
        else:  # Standard processing
            motor_tte_time, motor_tte_event_indicator, motor_tte_mask = (
                compute_time_bins_batch_optimized(
                    time_vectors, event_indicators, self.motor_time_bins, vocab_size
                )
            )

        # Assign results
        record["motor_tte_times"] = motor_tte_time
        record["motor_tte_event_indicators"] = motor_tte_event_indicator
        record["motor_tte_masks"] = motor_tte_mask

        # Validation
        assert (
            sum(record["motor_tte_task_indicators"]) == n_predictions
        ), f'sum(record["motor_tte_task_indicators"]) == n_predictions must be true'

        # Clean up
        del record["motor_tte_tasks"]
        del record["motor_censor_times"]
        del record["motor_tte_label_offsets"]

        return record
