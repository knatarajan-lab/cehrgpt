import json
import math
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
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

    def features(self, n=30, dim=N_EMBD, folder="features", seed=0):
        rng = np.random.RandomState(seed)
        path = os.path.join(self.tmp, folder)
        os.makedirs(path, exist_ok=True)
        frame = pd.DataFrame(
            {
                "subject_id": np.arange(n),
                "prediction_time_posix": np.full(n, 1.6e9),
                "boolean_value": rng.randint(0, 2, n),
                "features": [rng.randn(dim).astype(np.float32) for _ in range(n)],
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

    def test_evaluation_against_the_label(self):
        metrics, output, frame = self.run_cli(
            "--concepts", self.codes[0], self.codes[1], "--windows", "30", "90",
            "--evaluate_concept", self.codes[1], "--evaluate_window", "0-90",
        )
        evaluation = metrics["evaluation"]
        self.assertEqual(evaluation["concept_id"], self.codes[1])
        self.assertEqual(evaluation["window_days"], "0-90")
        self.assertEqual(evaluation["n"], 30)
        self.assertEqual(evaluation["positives"], int(frame["boolean_value"].sum()))
        self.assertTrue(0 <= evaluation["auroc"] <= 1)
        # the same number from the written predictions
        p = pd.read_parquet(os.path.join(output, "predictions.parquet"))
        p = p[(p.concept_id == self.codes[1]) & (p.window_days == "0-90")].sort_values("subject_id")
        from sklearn.metrics import roc_auc_score

        self.assertAlmostEqual(evaluation["auroc"], roc_auc_score(p["label"], p["probability"]), places=6)
        with open(os.path.join(output, "metrics.json")) as f:
            self.assertEqual(json.load(f)["evaluation"]["n"], 30)

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
        with self.assertRaisesRegex(ValueError, "must be one of the predicted"):
            zs.run(parse(base + ["--concepts", self.codes[0], "--evaluate_concept", self.codes[1]]))
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
