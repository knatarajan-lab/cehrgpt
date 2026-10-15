import json
import math
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset

from cehrgpt.models.config import CEHRGPTConfig
from cehrgpt.models.hf_cehrgpt import CEHRGPT2LMHeadModel
from cehrgpt.tools import zero_shot_motor_predictions as zs

REPO = os.path.join(os.path.dirname(__file__), "..", "..", "..")
BINS = [0.0, 1.0, 7.0, 365.0, float("inf")]
N_EMBD = 32
MOTOR_VOCAB = 40


def tiny_model(vocab_size=64, motor_vocab=MOTOR_VOCAB, bins=BINS, pieces=4):
    torch.manual_seed(0)
    config = CEHRGPTConfig(
        vocab_size=vocab_size, n_positions=64, n_embd=N_EMBD, n_layer=2, n_head=4, n_inner=64,
        bos_token_id=0, eos_token_id=1, pad_token_id=0, include_values=False,
        use_sub_time_tokenization=False, include_ttv_prediction=False,
        include_motor_time_to_event=True, motor_tte_vocab_size=motor_vocab,
        motor_num_time_pieces=pieces, motor_time_bins=list(bins), linear_prob_n_layer=1,
        linear_prob_token_id=8, ve_token_id=5, vs_token_id=4, att_token_ids=[6, 7],
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    )
    return CEHRGPT2LMHeadModel(config).eval()


def randomise_head(model, seed=1):
    """Give the head non trivial weights, so a wrong formula cannot hide behind zeros."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in model.motor_tte.parameters():
            parameter.copy_(0.3 * torch.randn(parameter.shape, generator=generator))
        model.motor_tte.task_layer.bias.copy_(-4.0 + 0.5 * torch.randn(model.motor_tte.task_layer.bias.shape, generator=generator))
    return model


class WindowTest(unittest.TestCase):
    def test_parse_window(self):
        self.assertEqual(zs.parse_window("90"), (0.0, 90.0))
        self.assertEqual(zs.parse_window("30-90"), (30.0, 90.0))
        self.assertEqual(zs.window_name((30.0, 90.0)), "30-90")
        for bad in ("0", "-5", "90-30", "30-30", "a"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                zs.parse_window(bad)

    def hazards(self, rates):
        """log2 hazards of one row and one task from the events per day in each bin."""
        return torch.log2(torch.tensor(rates)).reshape(1, len(rates), 1)

    def test_cumulative_hazard_of_a_piecewise_exponential(self):
        bins = torch.tensor(BINS)
        h = self.hazards([1.0, 2.0, 0.5, 0.25])
        at = lambda t: zs.cumulative_hazard(h, bins, torch.tensor([t])).item()  # noqa: E731
        self.assertAlmostEqual(at(0.0), 0.0)
        self.assertAlmostEqual(at(0.5), 0.5)                       # inside the first bin
        self.assertAlmostEqual(at(1.0), 1.0)
        self.assertAlmostEqual(at(3.0), 1.0 + 2.0 * 2.0)           # a day at rate 1, two days at rate 2
        self.assertAlmostEqual(at(7.0), 1.0 + 2.0 * 6.0)
        self.assertAlmostEqual(at(10.0), 1.0 + 12.0 + 0.5 * 3.0)
        self.assertAlmostEqual(at(400.0), 1.0 + 12.0 + 0.5 * 358.0 + 0.25 * 35.0)  # into the open last bin

    def test_window_probabilities(self):
        bins = torch.tensor(BINS)
        h = self.hazards([1.0, 2.0, 0.5, 0.25])
        probability = lambda s, e, g=None: zs.window_probability(h, bins, s, e, g).item()  # noqa: E731
        self.assertAlmostEqual(probability(0, 3), 1 - math.exp(-5.0), places=6)
        # a window that starts later: the chance of the event in days 3 to 7, from the start
        self.assertAlmostEqual(probability(3, 7), math.exp(-5.0) - math.exp(-13.0), places=6)
        # given no event during a gap of 2 days: H(2) = 3, H(4) = 7
        gap = torch.tensor([2.0])
        self.assertAlmostEqual(probability(0, 2, gap), 1 - math.exp(-(7.0 - 3.0)), places=6)
        self.assertAlmostEqual(
            probability(2, 5, gap), math.exp(-(zs.cumulative_hazard(h, bins, torch.tensor([4.0])).item() - 3.0))
            - math.exp(-(zs.cumulative_hazard(h, bins, torch.tensor([7.0])).item() - 3.0)), places=6,
        )

    def test_probabilities_are_proper(self):
        bins = torch.tensor(BINS)
        torch.manual_seed(0)
        h = -3 + torch.randn(50, 4, 6)
        previous = torch.zeros(50, 6)
        for end in (1, 5, 30, 365, 3650):
            p = zs.window_probability(h, bins, 0, end)
            self.assertTrue(((p >= 0) & (p <= 1)).all())
            self.assertTrue((p >= previous - 1e-7).all())  # a longer window cannot be less likely
            previous = p
        # the windows (0, a], (a, b] split the window (0, b] without conditioning on survival
        whole = zs.window_probability(h, bins, 0, 30)
        split = zs.window_probability(h, bins, 0, 7) + zs.window_probability(h, bins, 7, 30)
        torch.testing.assert_close(whole, split, atol=1e-6, rtol=1e-5)

    def test_a_tiny_hazard_does_not_round_to_zero(self):
        bins = torch.tensor(BINS)
        h = self.hazards([1e-9] * 4)
        p = zs.window_probability(h, bins, 0, 1).item()
        self.assertAlmostEqual(p / 1e-9, 1.0, places=3)


TASK_CONFIGS = os.path.join(REPO, "src", "cehrgpt", "time_to_event", "config")


class TaskConfigTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def write(self, **fields):
        path = os.path.join(self.tmp, "task.yaml")
        with open(path, "w") as f:
            yaml.safe_dump({"task_name": "demo", **fields}, f)
        return path

    def load(self, name, **kwargs):
        return zs.outcome_from_task_config(os.path.join(TASK_CONFIGS, name), **kwargs)

    def test_the_readmission_file_as_shipped(self):
        task = self.load("30_day_readmission.yaml")
        self.assertEqual(task.task_name, "30_day_readmission_prediction")
        self.assertEqual(task.outcome_events, ["9201", "262", "8971", "8920"])
        self.assertFalse(task.include_descendants)
        self.assertEqual(task.window, "0-30")
        self.assertEqual(zs.parse_window(task.window), (0.0, 30.0))

    def test_the_cabg_file_includes_descendants(self):
        task = self.load("1_year_cabg.yaml")
        self.assertTrue(task.include_descendants)
        self.assertEqual(task.window, "0-365")
        self.assertEqual(len(task.outcome_events), 14)
        self.assertEqual(task.outcome_events[0], "43528001")

    def test_a_window_without_an_end_needs_a_horizon(self):
        with self.assertRaisesRegex(ValueError, "open_window_days"):
            self.load("t2dm_hf.yaml")
        task = self.load("t2dm_hf.yaml", open_window_days=365)
        self.assertEqual(task.window, "30-365")
        self.assertEqual(task.outcome_events, ["316139"])

    def test_a_visit_limit_cannot_be_expressed(self):
        with self.assertRaisesRegex(ValueError, "future visits"):
            self.load("next_visit_type_prediction.yaml")
        task = self.load("next_visit_type_prediction.yaml", ignore_visit_limits=True)
        self.assertEqual(task.window, "0-365")  # the default window of the file

    def test_every_shipped_file_loads_or_says_why(self):
        for name in sorted(os.listdir(TASK_CONFIGS)):
            with self.subTest(name=name):
                try:
                    task = self.load(name, open_window_days=365, ignore_visit_limits=True)
                except ValueError as error:
                    self.fail(f"{name}: {error}")
                self.assertTrue(task.outcome_events)
                start, end = zs.parse_window(task.window)
                self.assertLess(start, end)

    def test_unquoted_ids_become_strings(self):
        task = zs.outcome_from_task_config(self.write(outcome_events=[9201, 262], prediction_window_end=90))
        self.assertEqual(task.outcome_events, ["9201", "262"])
        self.assertEqual(task.window, "0-90")

    def test_tokens_cannot_have_descendants_expanded(self):
        path = self.write(outcome_events=["Visit/IP"], include_descendants=True)
        with self.assertRaisesRegex(ValueError, "include_descendants"):
            zs.outcome_from_task_config(path)

    def test_invalid_files(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            zs.outcome_from_task_config(os.path.join(self.tmp, "missing.yaml"))
        with self.assertRaisesRegex(ValueError, "empty"):
            zs.outcome_from_task_config(self.write(outcome_events=[]))
        with self.assertRaisesRegex(ValueError, "not valid"):
            zs.outcome_from_task_config(self.write(outcome_events=["1"], prediction_window_start=50, prediction_window_end=20))
        with self.assertRaisesRegex(ValueError, "mix"):
            zs.outcome_from_task_config(self.write(outcome_events=["1", "Visit/IP"]))

    def test_the_shared_loader_reports_a_missing_file(self):
        # it used to raise a TypeError from a malformed except clause
        from cehrgpt.time_to_event.time_to_event_prediction import load_task_config_from_yaml

        with self.assertRaisesRegex(ValueError, "Could not open the task_config"):
            load_task_config_from_yaml(os.path.join(self.tmp, "missing.yaml"))


class OutcomeHelpersTest(unittest.TestCase):
    def test_combine_probabilities(self):
        p = np.array([[0.5, 0.5], [0.1, 0.0], [1.0, 0.3]])
        np.testing.assert_allclose(zs.combine_probabilities(p, "independent_or"), [0.75, 0.1, 1.0], atol=1e-9)
        np.testing.assert_allclose(zs.combine_probabilities(p, "max"), [0.5, 0.1, 1.0])
        np.testing.assert_allclose(zs.combine_probabilities(p[:, :1], "max"), p[:, 0])
        self.assertTrue(np.isfinite(zs.combine_probabilities(p, "independent_or")).all())
        with self.assertRaises(ValueError):
            zs.combine_probabilities(p, "mean")

    def test_sampled_events_estimate_the_combined_probability(self):
        rng = np.random.RandomState(0)
        p = np.clip(rng.rand(40, 3) * 0.6, 0, 1)
        exact = zs.combine_probabilities(p, "independent_or")
        any_event, mean_events = zs.sample_outcome_events(p, num_trials=20000, seed=1)
        # within four standard errors of the closed form, and the mean count is the sum of the probabilities
        self.assertTrue((np.abs(any_event - exact) < 4 * np.sqrt(exact * (1 - exact) / 20000) + 1e-3).all())
        self.assertTrue((np.abs(mean_events - p.sum(axis=1)) < 0.05).all())
        # the same seed gives the same draws, another seed does not
        again, _ = zs.sample_outcome_events(p, num_trials=100, seed=1)
        np.testing.assert_array_equal(again, zs.sample_outcome_events(p, num_trials=100, seed=1)[0])
        self.assertFalse(np.array_equal(again, zs.sample_outcome_events(p, num_trials=100, seed=2)[0]))
        # the estimates are multiples of 1 / num_trials
        np.testing.assert_allclose(again * 100, np.round(again * 100), atol=1e-9)

    def test_sampling_edge_cases(self):
        certain = np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
        any_event, mean_events = zs.sample_outcome_events(certain, num_trials=50)
        np.testing.assert_array_equal(any_event, [1.0, 0.0, 1.0])
        np.testing.assert_array_equal(mean_events, [1.0, 0.0, 2.0])
        # chunking the rows does not change what is estimated
        big = np.full((50, 2), 0.3)
        a, _ = zs.sample_outcome_events(big, num_trials=5000, chunk_rows=7)
        self.assertAlmostEqual(a.mean(), 1 - 0.7 ** 2, delta=0.01)
        with self.assertRaises(ValueError):
            zs.sample_outcome_events(big, num_trials=0)
        # a single task is not sampled: its probability is used as it is
        np.testing.assert_allclose(zs.combine_probabilities(np.array([[0.3], [0.8]]), "sampled"), [0.3, 0.8])

    def test_a_concept_that_is_a_task_covers_its_descendants_itself(self):
        # a > b > c is a chain of tasks, d is unrelated; ancestors include the concept itself
        ancestors = {"a": {"a"}, "b": {"b", "a"}, "c": {"c", "b", "a"}, "d": {"d"}}
        tasks = ["a", "b", "c", "d"]
        expand = lambda concepts: zs.expand_to_tasks(concepts, tasks, ancestors.__getitem__)  # noqa: E731
        self.assertEqual(expand(["a"]), ["a"])
        self.assertEqual(expand(["b"]), ["b"])   # c is below b, so b alone covers it
        self.assertEqual(expand(["c"]), ["c"])
        self.assertEqual(expand(["a", "d"]), ["a", "d"])
        self.assertEqual(expand(["c", "b"]), ["b"])  # one concept's task contains the other's
        self.assertEqual(expand(["nothing"]), [])

    def test_a_broad_concept_that_is_not_a_task_is_covered_by_the_highest_tasks_below_it(self):
        ancestors = {
            "a": {"a", "x"}, "b": {"b", "a", "x"}, "c": {"c", "b", "a", "x"},
            "d": {"d", "x"}, "e": {"e", "b", "a", "d", "x"},  # e is below both b and d
            "f": {"f"},
        }
        tasks = ["a", "b", "c", "d", "e", "f"]
        self.assertEqual(zs.expand_to_tasks(["x"], tasks, ancestors.__getitem__), ["a", "d"])
        # without a covering parent, siblings stay separate
        self.assertEqual(zs.expand_to_tasks(["b", "d"], tasks, ancestors.__getitem__), ["b", "d"])


class HeadTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_the_restricted_forward_equals_the_head(self):
        model = randomise_head(tiny_model())
        predictor = zs.MotorPredictor(model.motor_tte, BINS, "cpu")
        features = torch.randn(7, N_EMBD)
        tasks = [3, 0, 39, 12]
        torch.testing.assert_close(
            predictor.log2_hazards(features, tasks), model.motor_tte(features)[:, :, tasks], atol=1e-5, rtol=1e-5
        )

    def test_the_survival_term_of_the_training_loss_is_the_cumulative_hazard(self):
        """The probability formula uses the same hazard convention as the loss the head was trained on."""
        model = randomise_head(tiny_model())
        length = 10
        hidden = torch.randn(1, length, N_EMBD)
        indicators = torch.zeros(1, length, dtype=torch.bool)
        indicators[0, 3] = True
        censor_days = 20.0
        censor = torch.zeros(1, length)
        censor[0, 3] = censor_days
        empty_long = torch.zeros(0, dtype=torch.long)
        loss = model._motor_nll_loss(
            hidden, empty_long, empty_long, torch.zeros(0), censor, indicators, torch.tensor([0])
        )
        # no events, so the loss is the cumulative hazard summed over the tasks, over rows * tasks
        predictor = zs.MotorPredictor(model.motor_tte, BINS, "cpu")
        h = predictor.log2_hazards(hidden[0, 3][None], list(range(MOTOR_VOCAB)))
        total = zs.cumulative_hazard(h, predictor.bins, torch.tensor([censor_days])).sum().item()
        self.assertAlmostEqual(loss.item(), total / MOTOR_VOCAB, places=4)

    def test_the_event_term_of_the_training_loss_is_the_log_hazard_in_the_bin(self):
        model = randomise_head(tiny_model())
        length = 10
        hidden = torch.randn(1, length, N_EMBD)
        indicators = torch.zeros(1, length, dtype=torch.bool)
        indicators[0, 3] = True
        censor = torch.zeros(1, length)
        censor[0, 3] = 30.0
        task, event_day = 5, 4.0  # in the second bin, (1, 7], which is piece 1
        args = (hidden, torch.tensor([0]), torch.tensor([task]), torch.tensor([event_day]), censor, indicators, torch.tensor([1]))
        with_event = model._motor_nll_loss(*args)
        empty = torch.zeros(0, dtype=torch.long)
        without = model._motor_nll_loss(hidden, empty, empty, torch.zeros(0), censor, indicators, torch.tensor([0]))
        # the event term is -ln2 * logit of that bin, and the survival term now stops at the event time
        predictor = zs.MotorPredictor(model.motor_tte, BINS, "cpu")
        h = predictor.log2_hazards(hidden[0, 3][None], list(range(MOTOR_VOCAB)))
        survival_change = (
            zs.cumulative_hazard(h, predictor.bins, torch.tensor([event_day]))[0, task]
            - zs.cumulative_hazard(h, predictor.bins, torch.tensor([30.0]))[0, task]
        ).item()
        expected = (-math.log(2) * h[0, 1, task].item() + survival_change) / MOTOR_VOCAB
        self.assertAlmostEqual((with_event - without).item(), expected, places=4)

    def test_loading_the_head_from_every_checkpoint_layout(self):
        model = randomise_head(tiny_model())
        reference = {k: v.clone() for k, v in model.motor_tte.state_dict().items()}
        layouts = {
            "safetensors": dict(safe_serialization=True),
            "bin": dict(safe_serialization=False),
            "sharded": dict(safe_serialization=True, max_shard_size="40KB"),
        }
        for name, options in layouts.items():
            with self.subTest(layout=name):
                folder = os.path.join(self.tmp, name)
                model.save_pretrained(folder, **options)
                head, config, bins = zs.load_motor_head(folder)
                self.assertEqual(bins, BINS)
                self.assertEqual(config.motor_tte_vocab_size, MOTOR_VOCAB)
                for key, value in reference.items():
                    torch.testing.assert_close(head.state_dict()[key], value)
        sharded = os.listdir(os.path.join(self.tmp, "sharded"))
        self.assertTrue(any(f.endswith(".index.json") for f in sharded))

    def test_helpful_errors(self):
        model = tiny_model()
        model.save_pretrained(os.path.join(self.tmp, "ok"))
        # a model that was not pretrained with MOTOR
        plain = tiny_model()
        plain.config.include_motor_time_to_event = False
        plain.save_pretrained(os.path.join(self.tmp, "plain"))
        with self.assertRaisesRegex(ValueError, "not pretrained with MOTOR"):
            zs.load_motor_head(os.path.join(self.tmp, "plain"))
        # weights without the head
        from safetensors.torch import save_file

        os.makedirs(os.path.join(self.tmp, "nohead"))
        model.config.save_pretrained(os.path.join(self.tmp, "nohead"))
        save_file({"cehrgpt.wte.weight": torch.zeros(2, 2)}, os.path.join(self.tmp, "nohead", "model.safetensors"))
        with self.assertRaisesRegex(ValueError, "No MOTOR head weights"):
            zs.load_motor_head(os.path.join(self.tmp, "nohead"))


class EndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments

        from cehrgpt.models.pretrained_embeddings import PretrainedEmbeddings
        from cehrgpt.tokenization.tokenization_cehrgpt import CehrGptTokenizer

        cls.tmp = tempfile.mkdtemp()
        sample = os.path.join(REPO, "sample_data")
        concept = pd.read_parquet(os.path.join(sample, "omop_vocab", "concept"))
        names = {str(r.concept_id): r.concept_name for r in concept.itertuples()}
        data = Dataset.from_parquet(os.path.join(sample, "pretrain", "patient_sequence.parquet"))
        args = DataTrainingArguments(
            data_folder="x", dataset_prepared_path=os.path.join(cls.tmp, "prepared"),
            preprocessing_num_workers=1, preprocessing_batch_size=100,
        )
        cls.tokenizer = CehrGptTokenizer.train_tokenizer(
            data, names, args, PretrainedEmbeddings(os.path.join(sample, "pretrained_embeddings")),
            num_motor_tasks=40, ontology=None,
        )
        # An ontology over the tasks: c1 and c3 are below c0, c2 is below c1; "999001" is a concept
        # that is not a task (a numeric id, as a real concept) and sits above c0 and c5
        from cehrgpt.omop.ontology import Ontology

        codes = sorted(cls.tokenizer._motor_time_to_event_codes)
        ontology = Ontology.__new__(Ontology)
        ontology.parents_map = {
            codes[1]: {codes[0]}, codes[2]: {codes[1]}, codes[3]: {codes[0]},
            codes[0]: {"999001"}, codes[5]: {"999001"},
        }
        ontology.all_parents_map = {}
        ontology.concept_domain_map = {}
        ontology.concept_vocabulary_map = {}
        cls.tokenizer._ontology = ontology
        cls.model_dir = os.path.join(cls.tmp, "model")
        cls.bins = [float(b) for b in cls.tokenizer.get_motor_time_bins(4)]
        model = randomise_head(
            tiny_model(vocab_size=cls.tokenizer.vocab_size, motor_vocab=cls.tokenizer.motor_tte_vocab_size, bins=cls.bins)
        )
        model.save_pretrained(cls.model_dir)
        cls.tokenizer.save_pretrained(cls.model_dir)
        cls.model = model
        cls.codes = sorted(cls.tokenizer._motor_time_to_event_codes)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def features(self, n=30, dim=N_EMBD, folder="features", seed=0, signal=False):
        rng = np.random.RandomState(seed)
        path = os.path.join(self.tmp, folder)
        os.makedirs(path, exist_ok=True)
        vectors = [rng.randn(dim).astype(np.float32) for _ in range(n)]
        # with a signal the label is a linear function of the features, so a fitted model finds it
        labels = np.array([int(v[0] + 0.5 * v[1] > 0) for v in vectors]) if signal else rng.randint(0, 2, n)
        frame = pd.DataFrame(
            {
                "subject_id": np.arange(n),
                "prediction_time_posix": np.full(n, 1.6e9),
                "boolean_value": labels,
                "features": vectors,
            }
        )
        frame.iloc[: n // 2].to_parquet(os.path.join(path, "a.parquet"))
        frame.iloc[n // 2:].to_parquet(os.path.join(path, "b.parquet"))
        return path, frame

    def run_cli(self, *extra, folder="features", out="out", n=30):
        path, frame = self.features(n=n, folder=folder)
        output = os.path.join(self.tmp, out)
        shutil.rmtree(output, ignore_errors=True)
        args = zs.create_arg_parser().parse_args(
            ["--model_path", self.model_dir, "--features_folder", path, "--output_dir", output, *extra]
        )
        return zs.run(args), output, frame

    def test_predictions_match_the_model_head(self):
        concepts = [self.codes[3], self.codes[11]]
        metrics, output, frame = self.run_cli("--concepts", *concepts, "--windows", "30", "30-90", "--batch_size", "7")
        predictions = pd.read_parquet(os.path.join(output, "predictions.parquet"))
        self.assertEqual(len(predictions), 30 * 2 * 2)
        self.assertEqual(set(predictions["concept_id"]), set(concepts))
        self.assertEqual(set(predictions["window_days"]), {"0-30", "30-90"})
        self.assertTrue(predictions["probability"].between(0, 1).all())
        self.assertEqual(sorted(predictions["subject_id"].unique()), list(range(30)))
        # recompute one patient by hand through the model's own head
        row = 4
        features = torch.tensor(frame["features"].iloc[row])[None]
        for concept in concepts:
            index = self.tokenizer.get_motor_token_id(concept)
            with torch.no_grad():
                log2 = self.model.motor_tte(features)[:, :, [index]]
            for window in ((0.0, 30.0), (30.0, 90.0)):
                expected = zs.window_probability(log2, torch.tensor(self.bins), *window).item()
                found = predictions[
                    (predictions.subject_id == row) & (predictions.concept_id == concept)
                    & (predictions.window_days == zs.window_name(window))
                ]["probability"].item()
                self.assertAlmostEqual(found, expected, places=5)
        self.assertEqual(metrics["num_patients"], 30)

    def test_a_longer_window_is_never_less_likely(self):
        _, output, _ = self.run_cli("--concepts", self.codes[0], "--windows", "10", "100", "1000")
        p = pd.read_parquet(os.path.join(output, "predictions.parquet"))
        wide = p.pivot(index="subject_id", columns="window_days", values="probability")
        self.assertTrue((wide["0-100"] >= wide["0-10"] - 1e-7).all())
        self.assertTrue((wide["0-1000"] >= wide["0-100"] - 1e-7).all())

    def test_the_gap_conditions_on_the_time_since_the_last_event(self):
        # Inside one time bin the hazard is constant, so the past does not matter and a gap
        # changes nothing. The sample tokenizer has a bin edge at 875 days, so a window that
        # crosses it, after a gap that gets close, does depend on the gap.
        self.assertEqual(self.bins[2], 875.0)
        crossing = ["--concepts", self.codes[0], "--windows", "200"]
        _, with_gap, _ = self.run_cli(*crossing, "--gap_days", "800", out="gap")
        _, without, _ = self.run_cli(*crossing, out="nogap")
        a = pd.read_parquet(os.path.join(with_gap, "predictions.parquet"))["probability"].to_numpy()
        b = pd.read_parquet(os.path.join(without, "predictions.parquet"))["probability"].to_numpy()
        self.assertFalse(np.allclose(a, b))
        # ... and inside a bin it changes nothing, as it should
        inside = ["--concepts", self.codes[0], "--windows", "30"]
        _, g1, _ = self.run_cli(*inside, "--gap_days", "20", out="gap_inside")
        _, g0, _ = self.run_cli(*inside, out="nogap_inside")
        np.testing.assert_allclose(
            pd.read_parquet(os.path.join(g1, "predictions.parquet"))["probability"].to_numpy(),
            pd.read_parquet(os.path.join(g0, "predictions.parquet"))["probability"].to_numpy(), rtol=1e-4,
        )

    def test_a_gap_per_patient_from_a_column(self):
        path, frame = self.features(folder="gapcol")
        frame = frame.assign(gap=np.where(frame.subject_id % 2 == 0, 800.0, 0.0))
        shutil.rmtree(path)
        os.makedirs(path)
        frame.to_parquet(os.path.join(path, "all.parquet"))
        out = os.path.join(self.tmp, "gapcol_out")
        zs.run(zs.create_arg_parser().parse_args(
            ["--model_path", self.model_dir, "--features_folder", path, "--output_dir", out,
             "--concepts", self.codes[0], "--windows", "200", "--gap_days_column", "gap"]
        ))
        p = pd.read_parquet(os.path.join(out, "predictions.parquet")).set_index("subject_id")["probability"]
        _, base, _ = self.run_cli("--concepts", self.codes[0], "--windows", "200", out="gapcol_base")
        q = pd.read_parquet(os.path.join(base, "predictions.parquet")).set_index("subject_id")["probability"]
        odd = [i for i in range(30) if i % 2 == 1]
        even = [i for i in range(30) if i % 2 == 0]
        np.testing.assert_allclose(p.loc[odd].to_numpy(), q.loc[odd].to_numpy(), rtol=1e-5)  # gap 0: unchanged
        self.assertFalse(np.allclose(p.loc[even].to_numpy(), q.loc[even].to_numpy()))

    def test_binary_outcome_against_the_label(self):
        metrics, output, frame = self.run_cli(
            "--outcome_concepts", self.codes[1], "--windows", "30", "90", "--outcome_window", "90",
        )
        outcome = metrics["outcome"]
        self.assertEqual(outcome["concepts"], [self.codes[1]])
        self.assertEqual(outcome["window_days"], "0-90")
        self.assertEqual(outcome["combine"], "single task")
        self.assertEqual(outcome["zero_shot"]["n"], 30)
        self.assertEqual(outcome["zero_shot"]["positives"], int(frame["boolean_value"].sum()))
        self.assertTrue(0 <= outcome["zero_shot"]["auroc"] <= 1)
        # the same number from the written scores
        scores = pd.read_parquet(os.path.join(output, "outcome_scores.parquet")).sort_values("subject_id")
        from sklearn.metrics import roc_auc_score

        self.assertAlmostEqual(outcome["zero_shot"]["auroc"], roc_auc_score(scores["label"], scores["probability"]), places=6)
        self.assertEqual(list(scores.columns), ["subject_id", "prediction_time_posix", "probability", "label"])
        # the per-concept predictions are written as well
        p = pd.read_parquet(os.path.join(output, "predictions.parquet"))
        self.assertEqual(set(p["concept_id"]), {self.codes[1]})
        with open(os.path.join(output, "metrics.json")) as f:
            self.assertEqual(json.load(f)["outcome"]["zero_shot"]["n"], 30)

    def test_the_outcome_window_is_predicted_even_if_not_listed(self):
        metrics, output, _ = self.run_cli("--outcome_concepts", self.codes[0], "--windows", "30", "--outcome_window", "60-120")
        self.assertIn("60-120", metrics["windows"])
        self.assertEqual(metrics["outcome"]["window_days"], "60-120")

    def test_several_outcome_concepts_are_combined(self):
        concepts = [self.codes[0], self.codes[1]]
        both, out_both, _ = self.run_cli("--outcome_concepts", *concepts, "--windows", "90")
        biggest, out_max, _ = self.run_cli("--outcome_concepts", *concepts, "--windows", "90", "--combine", "max", out="out_max")
        self.assertEqual(both["outcome"]["combine"], "independent_or")
        p = pd.read_parquet(os.path.join(out_both, "predictions.parquet"))
        wide = p.pivot(index="subject_id", columns="concept_id", values="probability").sort_index()
        expected = 1 - (1 - wide[concepts[0]]) * (1 - wide[concepts[1]])
        combined = pd.read_parquet(os.path.join(out_both, "outcome_scores.parquet")).set_index("subject_id").sort_index()
        np.testing.assert_allclose(combined["probability"].to_numpy(), expected.to_numpy(), atol=1e-6)
        as_max = pd.read_parquet(os.path.join(out_max, "outcome_scores.parquet")).set_index("subject_id").sort_index()
        np.testing.assert_allclose(as_max["probability"].to_numpy(), wide.max(axis=1).to_numpy(), atol=1e-6)
        self.assertEqual(set(both["outcome"]["per_concept"]), set(concepts))
        self.assertTrue((combined["probability"] >= wide.max(axis=1) - 1e-7).all())

    def test_sampled_combination_from_the_command_line(self):
        concepts = [self.codes[0], self.codes[1], self.codes[2]]
        metrics, output, _ = self.run_cli(
            "--outcome_concepts", *concepts, "--windows", "365", "--combine", "sampled",
            "--num_trials", "200", "--seed", "3", out="sampled",
        )
        outcome = metrics["outcome"]
        self.assertEqual(outcome["combine"], "sampled")
        self.assertEqual(outcome["num_trials"], 200)
        by_count = outcome["zero_shot_by_mean_events"]
        self.assertIsNotNone(by_count)
        # an expected number of events is a ranking score, not a probability
        self.assertIsNone(by_count["brier_score"])
        self.assertIsNone(by_count["mean_predicted_probability"])
        self.assertIsNotNone(outcome["zero_shot"]["brier_score"])
        scores = pd.read_parquet(os.path.join(output, "outcome_scores.parquet")).sort_values("subject_id")
        self.assertIn("mean_events_per_trial", scores.columns)
        np.testing.assert_allclose(scores["probability"] * 200, np.round(scores["probability"] * 200), atol=1e-4)
        # close to the closed form, which is what it estimates
        exact, out_exact, _ = self.run_cli("--outcome_concepts", *concepts, "--windows", "365", out="exact")
        closed = pd.read_parquet(os.path.join(out_exact, "outcome_scores.parquet")).sort_values("subject_id")
        self.assertLess(np.abs(scores["probability"].to_numpy() - closed["probability"].to_numpy()).max(), 0.2)
        self.assertIsNone(exact["outcome"]["zero_shot_by_mean_events"])
        # reproducible
        again, out_again, _ = self.run_cli(
            "--outcome_concepts", *concepts, "--windows", "365", "--combine", "sampled",
            "--num_trials", "200", "--seed", "3", out="sampled_again",
        )
        pd.testing.assert_frame_equal(
            scores.reset_index(drop=True),
            pd.read_parquet(os.path.join(out_again, "outcome_scores.parquet")).sort_values("subject_id").reset_index(drop=True),
        )

    def test_the_supervised_baseline(self):
        train, _ = self.features(n=400, folder="train", seed=1, signal=True)
        path, _ = self.features(n=200, folder="evalset", seed=2, signal=True)
        out = os.path.join(self.tmp, "baseline_out")
        metrics = zs.run(zs.create_arg_parser().parse_args(
            ["--model_path", self.model_dir, "--features_folder", path, "--output_dir", out,
             "--outcome_concepts", self.codes[0], "--baseline_train_features", train]
        ))["outcome"]
        baseline = metrics["supervised_baseline"]
        self.assertEqual(baseline["num_train_rows"], 400)
        self.assertEqual(baseline["n"], 200)
        self.assertGreater(baseline["auroc"], 0.95)       # the label is a linear function of the features
        self.assertLess(abs(metrics["zero_shot"]["auroc"] - 0.5), 0.2)  # a random head knows nothing about it
        # a limit on the training rows
        limited = zs.run(zs.create_arg_parser().parse_args(
            ["--model_path", self.model_dir, "--features_folder", path, "--output_dir", out,
             "--outcome_concepts", self.codes[0], "--baseline_train_features", train, "--baseline_max_rows", "100"]
        ))["outcome"]["supervised_baseline"]
        self.assertLess(limited["num_train_rows"], 200)

    def test_descendants_are_covered_through_the_tasks_below_a_concept(self):
        c = self.codes
        ancestors_of = zs.tokenizer_ancestors(self.tokenizer)
        self.assertEqual(ancestors_of(c[2]), {c[2], c[1], c[0], "999001"})
        # a concept that is a task covers its descendants itself
        metrics, _, _ = self.run_cli("--outcome_concepts", c[0], "--expand_descendants", out="expand_task")
        self.assertEqual(metrics["outcome"]["concepts"], [c[0]])
        # "999001" is not a task: its highest tasks are c0 (with c1, c2, c3 below it) and c5
        metrics, output, _ = self.run_cli("--outcome_concepts", "999001", "--expand_descendants", out="expand_broad")
        outcome = metrics["outcome"]
        self.assertEqual(set(outcome["concepts"]), {c[0], c[5]})
        self.assertEqual(outcome["outcome_input"], ["999001"])
        self.assertEqual(set(outcome["tasks_per_input_concept"]["999001"]), {c[0], c[5]})
        self.assertEqual(set(pd.read_parquet(os.path.join(output, "predictions.parquet"))["concept_id"]), {c[0], c[5]})
        # a set of concepts from a file; c1 is below c0, so only c0 and c5 remain
        listing = os.path.join(self.tmp, "set.txt")
        with open(listing, "w") as f:
            f.write(f"{c[1]}\n{c[0]}\n\n{c[5]}\n")
        path, _ = self.features()
        args = zs.create_arg_parser().parse_args(
            ["--model_path", self.model_dir, "--features_folder", path, "--output_dir", os.path.join(self.tmp, "from_file"),
             "--outcome_concepts_file", listing, "--expand_descendants"]
        )
        self.assertEqual(set(zs.run(args)["outcome"]["concepts"]), {c[0], c[5]})

    def write_task(self, **fields):
        path = os.path.join(self.tmp, "task.yaml")
        with open(path, "w") as f:
            yaml.safe_dump({"task_name": "demo_task", **fields}, f)
        return path

    def test_the_time_to_event_task_file_defines_the_outcome(self):
        c = self.codes
        task = self.write_task(outcome_events=[c[1]], prediction_window_start=0, prediction_window_end=60)
        path, frame = self.features()
        out = os.path.join(self.tmp, "from_task")
        metrics = zs.run(zs.create_arg_parser().parse_args(
            ["--model_path", self.model_dir, "--features_folder", path, "--output_dir", out, "--task_config", task]
        ))
        self.assertEqual(metrics["task_name"], "demo_task")
        self.assertEqual(metrics["outcome"]["concepts"], [c[1]])
        self.assertEqual(metrics["outcome"]["window_days"], "0-60")
        # the same numbers as asking for it on the command line
        direct, _, _ = self.run_cli("--outcome_concepts", c[1], "--outcome_window", "0-60", out="direct")
        self.assertEqual(metrics["outcome"]["zero_shot"], direct["outcome"]["zero_shot"])
        self.assertIsNone(direct["task_name"])

    def test_the_file_can_include_descendants_and_the_command_line_wins(self):
        c = self.codes
        task = self.write_task(outcome_events=["999001"], include_descendants=True, prediction_window_end=45)
        path, _ = self.features()
        base = ["--model_path", self.model_dir, "--features_folder", path, "--task_config", task]
        parse = zs.create_arg_parser().parse_args
        resolved = zs.run(parse(base + ["--output_dir", os.path.join(self.tmp, "t1")]))["outcome"]
        self.assertEqual(set(resolved["concepts"]), {c[0], c[5]})
        self.assertEqual(resolved["window_days"], "0-45")
        # --outcome_window and --outcome_concepts override the file
        overridden = zs.run(parse(base + ["--output_dir", os.path.join(self.tmp, "t2"), "--outcome_window", "10-20", "--outcome_concepts", c[2]]))["outcome"]
        self.assertEqual(overridden["window_days"], "10-20")
        self.assertEqual(set(overridden["concepts"]), {c[2]})  # c2 is itself a task; its descendants are in it

    def test_an_open_window_needs_a_horizon_on_the_command_line(self):
        task = self.write_task(outcome_events=[self.codes[0]], prediction_window_start=30, prediction_window_end=-1)
        path, _ = self.features()
        base = ["--model_path", self.model_dir, "--features_folder", path, "--task_config", task]
        parse = zs.create_arg_parser().parse_args
        with self.assertRaisesRegex(ValueError, "open_window_days"):
            zs.run(parse(base + ["--output_dir", os.path.join(self.tmp, "o1")]))
        metrics = zs.run(parse(base + ["--output_dir", os.path.join(self.tmp, "o2"), "--open_window_days", "365"]))
        self.assertEqual(metrics["outcome"]["window_days"], "30-365")

    def test_descendants_from_an_omop_concept_ancestor_table(self):
        c = self.codes
        # a numeric stand-in for the broad concept, and the same hierarchy as the tokenizer's ontology
        broad = 999001
        parents = {c[1]: [c[0]], c[2]: [c[1]], c[3]: [c[0]], c[0]: [str(broad)], c[5]: [str(broad)]}
        nodes = set(c) | {str(broad)}
        rows = []
        for node in nodes:
            distance, queue = {node: 0}, [node]
            while queue:
                current = queue.pop(0)
                for parent in parents.get(current, []):
                    if parent not in distance:
                        distance[parent] = distance[current] + 1
                        queue.append(parent)
            rows += [{"ancestor_concept_id": int(a), "descendant_concept_id": int(node),
                      "min_levels_of_separation": d, "max_levels_of_separation": d} for a, d in distance.items()]
        table = os.path.join(self.tmp, "concept_ancestor.parquet")
        pd.DataFrame(rows).to_parquet(table)
        ancestors_of = zs.ancestors_from_table(table, c)
        self.assertEqual(ancestors_of(c[2]), {c[2], c[1], c[0], str(broad)})
        self.assertEqual(ancestors_of("not a task"), {"not a task"})
        # a tokenizer without an ontology: the table is what makes the expansion possible
        bare = os.path.join(self.tmp, "bare_tokenizer")
        self.tokenizer.save_pretrained(bare)
        os.remove(os.path.join(bare, "ontology.pickle"))
        path, _ = self.features()
        base = ["--model_path", self.model_dir, "--tokenizer_path", bare, "--features_folder", path,
                "--outcome_concepts", str(broad), "--expand_descendants"]
        parse = zs.create_arg_parser().parse_args
        with self.assertRaisesRegex(ValueError, "concept_ancestor"):
            zs.run(parse(base + ["--output_dir", os.path.join(self.tmp, "nt")]))
        resolved = zs.run(parse(base + ["--output_dir", os.path.join(self.tmp, "yt"), "--concept_ancestor", table]))["outcome"]
        self.assertEqual(set(resolved["concepts"]), {c[0], c[5]})

    def not_task_but_in_vocabulary(self):
        tasks = set(self.codes)
        return next(t for t in sorted(self.tokenizer.get_vocab()) if t.isdigit() and t not in tasks)

    def test_concepts_without_a_task_are_removed_with_a_warning(self):
        c = self.codes
        in_vocabulary_concept = self.not_task_but_in_vocabulary()
        path, _ = self.features()
        for expand in (False, True):
            with self.subTest(expand=expand):
                args = zs.create_arg_parser().parse_args(
                    ["--model_path", self.model_dir, "--features_folder", path,
                     "--output_dir", os.path.join(self.tmp, f"removed_{expand}"),
                     "--outcome_concepts", c[0], "123456789", in_vocabulary_concept, c[1]]
                    + (["--expand_descendants"] if expand else [])
                )
                with self.assertLogs("cehrgpt.tools.zero_shot_motor_predictions", level="WARNING") as logs:
                    metrics = zs.run(args)
                outcome = metrics["outcome"]
                # the run goes on with the concepts that have a task ...
                self.assertEqual(set(outcome["concepts"]), {c[0]} if expand else {c[0], c[1]})  # c1 is below c0
                # ... and says which ones were left out, and why
                self.assertEqual(set(outcome["removed_concepts"]), {"123456789", in_vocabulary_concept})
                self.assertIn("not in the tokenizer vocabulary", outcome["removed_concepts"]["123456789"])
                self.assertIn("in the vocabulary, but", outcome["removed_concepts"][in_vocabulary_concept])
                warning = "\n".join(logs.output)
                self.assertIn("Removed 2 of 4 outcome concepts", warning)
                self.assertIn("123456789", warning)
                self.assertIn(in_vocabulary_concept, warning)

    def test_nothing_is_removed_when_every_concept_has_a_task(self):
        metrics, _, _ = self.run_cli("--outcome_concepts", self.codes[0], self.codes[1], out="nothing_removed")
        self.assertIsNone(metrics["outcome"]["removed_concepts"])

    def test_a_set_where_nothing_has_a_task_cannot_be_predicted(self):
        path, _ = self.features()
        for expand in (False, True):
            with self.subTest(expand=expand):
                args = zs.create_arg_parser().parse_args(
                    ["--model_path", self.model_dir, "--features_folder", path,
                     "--output_dir", os.path.join(self.tmp, f"err_expand_{expand}"),
                     "--outcome_concepts", "123456789", "987654321"] + (["--expand_descendants"] if expand else [])
                )
                with self.assertRaisesRegex(ValueError, "None of the outcome concepts can be predicted") as error:
                    zs.run(args)
                self.assertIn("123456789", str(error.exception))
                self.assertIn("987654321", str(error.exception))

    def test_expanding_needs_an_ontology(self):
        no_ontology = os.path.join(self.tmp, "tok_without_ontology")
        tokenizer_copy = os.path.join(self.tmp, "tok_copy")
        self.tokenizer.save_pretrained(tokenizer_copy)
        os.remove(os.path.join(tokenizer_copy, "ontology.pickle"))
        path, _ = self.features()
        args = zs.create_arg_parser().parse_args(
            ["--model_path", self.model_dir, "--tokenizer_path", tokenizer_copy, "--features_folder", path,
             "--output_dir", no_ontology, "--outcome_concepts", self.codes[0], "--expand_descendants"]
        )
        with self.assertRaisesRegex(ValueError, "ontology"):
            zs.run(args)

    def test_the_brier_score_is_read_against_the_prevalence(self):
        labels = np.array([0, 0, 0, 1])
        # the same probability for everyone, equal to the prevalence, is the line to beat
        constant = zs.evaluate_predictions(labels, np.full(4, 0.25))
        self.assertAlmostEqual(constant["brier_score_of_the_prevalence"], 0.25 * 0.75)
        self.assertAlmostEqual(constant["brier_score"], 0.25 * 0.75)
        self.assertAlmostEqual(constant["brier_skill_score"], 0.0, places=9)
        perfect = zs.evaluate_predictions(labels, labels.astype(float))
        self.assertAlmostEqual(perfect["brier_skill_score"], 1.0)
        # probabilities that are worse than not using the model are below zero
        wrong = zs.evaluate_predictions(labels, np.array([0.9, 0.9, 0.9, 0.0]))
        self.assertLess(wrong["brier_skill_score"], 0)
        # with no variation in the labels there is no reference
        self.assertIsNone(zs.evaluate_predictions(np.zeros(4), np.full(4, 0.1))["brier_skill_score"])

    def test_a_score_that_is_not_a_probability_gets_ranking_metrics_only(self):
        result = zs.evaluate_predictions(np.array([0, 1, 1, 0]), np.array([0.5, 2.0, 1.5, 0.0]), is_probability=False)
        self.assertEqual(result["auroc"], 1.0)
        self.assertIsNone(result["brier_score"])
        self.assertIsNone(result["brier_skill_score"])

    def test_evaluate_metrics_function(self):
        result = zs.evaluate_predictions(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
        self.assertEqual(result["auroc"], 1.0)
        self.assertAlmostEqual(result["brier_score"], np.mean([0.01, 0.04, 0.04, 0.01]))
        self.assertEqual(result["prevalence"], 0.5)
        self.assertIsNone(zs.evaluate_predictions(np.zeros(3), np.array([0.1, 0.2, 0.3]))["auroc"])

    def test_list_tasks_and_search(self):
        import contextlib
        import io

        buffer = io.StringIO()
        args = zs.create_arg_parser().parse_args(["--model_path", self.model_dir, "--list_tasks"])
        with contextlib.redirect_stdout(buffer):
            zs.run(args)
        self.assertEqual(len(buffer.getvalue().strip().split("\n")), 40)
        first = buffer.getvalue().split("\n")[0].split("\t")
        buffer = io.StringIO()
        args.search = first[1][:5] if first[1] else first[0]
        with contextlib.redirect_stdout(buffer):
            zs.run(args)
        self.assertIn(first[0], buffer.getvalue())

    def test_clear_errors(self):
        path, _ = self.features()
        out = os.path.join(self.tmp, "err")
        base = ["--model_path", self.model_dir, "--features_folder", path, "--output_dir", out]
        parse = zs.create_arg_parser().parse_args
        with self.assertRaisesRegex(ValueError, "Not MOTOR tasks"):
            zs.run(parse(base + ["--concepts", "123456789"]))
        with self.assertRaisesRegex(ValueError, "Give the concepts"):
            zs.run(parse(base))
        with self.assertRaisesRegex(ValueError, "--expand_descendants"):
            zs.run(parse(base + ["--concepts", "123456789"]))
        unlabeled = os.path.join(self.tmp, "unlabeled")
        os.makedirs(unlabeled)
        pd.read_parquet(os.path.join(path, "a.parquet")).drop(columns=["boolean_value"]).to_parquet(os.path.join(unlabeled, "u.parquet"))
        with self.assertRaisesRegex(ValueError, "label column"):
            zs.run(parse(["--model_path", self.model_dir, "--features_folder", unlabeled, "--output_dir", out, "--outcome_concepts", self.codes[0]]))
        wide, _ = self.features(dim=2 * N_EMBD, folder="wide")
        with self.assertRaisesRegex(ValueError, "disable_combine_global_local"):
            zs.run(parse(["--model_path", self.model_dir, "--features_folder", wide, "--output_dir", out, "--concepts", self.codes[0]]))
        with self.assertRaises(FileNotFoundError):
            zs.run(parse(["--model_path", self.model_dir, "--features_folder", out + "_nothing", "--output_dir", out, "--concepts", self.codes[0]]))

    def test_the_tokenizer_must_match_the_model(self):
        other = os.path.join(self.tmp, "other_tokenizer")
        self.tokenizer.save_pretrained(other)
        wrong = tiny_model(vocab_size=self.tokenizer.vocab_size, motor_vocab=self.tokenizer.motor_tte_vocab_size + 5, bins=self.bins)
        folder = os.path.join(self.tmp, "wrong_model")
        wrong.save_pretrained(folder)
        path, _ = self.features()
        args = zs.create_arg_parser().parse_args(
            ["--model_path", folder, "--tokenizer_path", other, "--features_folder", path,
             "--output_dir", os.path.join(self.tmp, "o"), "--concepts", self.codes[0]]
        )
        with self.assertRaisesRegex(ValueError, "MOTOR tasks but the model"):
            zs.run(args)


if __name__ == "__main__":
    unittest.main()
