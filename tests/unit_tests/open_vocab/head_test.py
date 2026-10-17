import math
import unittest

import torch

from cehrgpt.models.hf_cehrgpt import MotorTaskHead
from cehrgpt.open_vocab.head import (
    OpenVocabularyHead,
    open_vocab_nll_loss,
    window_probabilities,
)

BINS = [0.0, 1.0, 7.0, 30.0, 365.0]  # four pieces
PIECES = len(BINS) - 1


def naive_loss(hazards, rows, questions, times, censor, bins):
    """Piecewise exponential NLL with plain loops, averaged over rows x questions."""
    n_rows, pieces, n_questions = hazards.shape
    happened = {(int(r), int(q)): float(t) for r, q, t in zip(rows, questions, times)}
    total = 0.0
    for r in range(n_rows):
        for q in range(n_questions):
            event = (r, q) in happened
            time = happened[(r, q)] if event else float(censor[r])
            for p in range(pieces):
                days = max(0.0, min(time, bins[p + 1]) - bins[p])
                total += 2.0 ** float(hazards[r, p, q]) * days
                if event and bins[p] <= time < bins[p + 1]:
                    total -= math.log(2) * float(hazards[r, p, q])
    return total / (n_rows * n_questions)


class OpenVocabularyHeadTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.hidden, self.question_dim = 12, 20

    def head(self, **kwargs):
        return OpenVocabularyHead(2 * self.hidden, self.question_dim, self.hidden, PIECES, **kwargs)

    def test_shapes_and_the_initial_hazard(self):
        head = self.head(bias_init=-9.0)
        out = head(torch.randn(5, 2 * self.hidden), torch.randn(7, self.question_dim))
        self.assertEqual(out.shape, (5, PIECES, 7))
        # nothing is learned yet: every question has the initial log2 hazard
        torch.testing.assert_close(out, torch.full_like(out, -9.0))

    def test_any_embedding_can_be_asked(self):
        head = self.head()
        for p in head.parameters():
            torch.nn.init.normal_(p, std=0.1)
        features = torch.randn(3, 2 * self.hidden)
        unseen = torch.randn(1, self.question_dim)  # a question that was never in the training set
        out = head(features, unseen)
        self.assertTrue(torch.isfinite(out).all())
        # a question does not depend on the others asked with it
        both = head(features, torch.cat([unseen, torch.randn(4, self.question_dim)]))
        torch.testing.assert_close(both[:, :, :1], out)

    def test_warm_start_reproduces_the_motor_head_and_ignores_the_local_half(self):
        tasks = 6  # fewer than the embedding width, so the fit is exact
        motor = MotorTaskHead(self.hidden, tasks, PIECES)
        for p in motor.parameters():
            torch.nn.init.normal_(p, std=0.3)
        with torch.no_grad():
            motor.task_time_bias.copy_(motor.task_time_bias[:, :, :1].expand(-1, -1, tasks))  # shared over tasks
        embeddings = torch.randn(tasks, self.question_dim)
        head = OpenVocabularyHead.from_motor_head(motor, embeddings, ridge=1e-9)

        global_state = torch.randn(9, self.hidden)
        local_state = torch.randn(9, self.hidden)
        expected = motor(global_state)  # (rows, pieces, tasks)
        got = head(torch.cat([global_state, local_state], dim=1), embeddings)
        torch.testing.assert_close(got, expected, rtol=1e-3, atol=1e-3)
        # the local half starts at zero, so changing it changes nothing
        other = head(torch.cat([global_state, torch.randn_like(local_state)], dim=1), embeddings)
        torch.testing.assert_close(other, got)

    def test_the_local_half_can_be_learned_after_the_warm_start(self):
        motor = MotorTaskHead(self.hidden, 4, PIECES)
        embeddings = torch.randn(4, self.question_dim)
        head = OpenVocabularyHead.from_motor_head(motor, embeddings)
        features = torch.randn(6, 2 * self.hidden)
        before = head(features, embeddings).detach()
        torch.testing.assert_close(before, head(torch.cat([features[:, : self.hidden], torch.zeros(6, self.hidden)], 1), embeddings))
        head.final_layer.weight.data[:, self.hidden :] += 0.5
        after = head(features, embeddings)
        self.assertFalse(torch.allclose(before, after))

    def test_warm_start_checks_its_inputs(self):
        motor = MotorTaskHead(self.hidden, 4, PIECES)
        with self.assertRaisesRegex(ValueError, "task embeddings"):
            OpenVocabularyHead.from_motor_head(motor, torch.randn(5, self.question_dim))
        with self.assertRaisesRegex(ValueError, "smaller than"):
            OpenVocabularyHead.from_motor_head(motor, torch.randn(4, self.question_dim), feature_dim=self.hidden - 1)


class LossTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.rows, self.questions = 5, 7
        self.hazards = torch.randn(self.rows, PIECES, self.questions, dtype=torch.float64) - 6
        self.event_rows = torch.tensor([0, 0, 2, 3, 3, 4])
        self.event_questions = torch.tensor([1, 4, 0, 6, 2, 5])
        self.event_times = torch.tensor([0.5, 20.0, 3.0, 400.0, 1.0, 7.0], dtype=torch.float64)  # pieces 0, 2, 1, 3, 1(edge 1.0), 2(edge 7.0)
        self.censor = torch.tensor([30.0, 100.0, 365.0, 500.0, 50.0], dtype=torch.float64)

    def test_matches_a_loop_over_rows_and_questions(self):
        # the censor time of a row must not be before its events
        self.censor = torch.tensor([30.0, 100.0, 365.0, 500.0, 50.0], dtype=torch.float64)
        got = open_vocab_nll_loss(self.hazards, self.event_rows, self.event_questions, self.event_times, self.censor, BINS)
        want = naive_loss(self.hazards, self.event_rows, self.event_questions, self.event_times, self.censor, BINS)
        self.assertAlmostEqual(float(got), want, places=9)

    def test_chunks_give_the_same_loss(self):
        full = open_vocab_nll_loss(self.hazards, self.event_rows, self.event_questions, self.event_times, self.censor, BINS)
        for chunk in (1, 2, 3, 100):
            part = open_vocab_nll_loss(
                self.hazards, self.event_rows, self.event_questions, self.event_times, self.censor, BINS, chunk_rows=chunk
            )
            self.assertAlmostEqual(float(part), float(full), places=10)

    def test_a_higher_hazard_at_the_event_lowers_the_loss(self):
        base = open_vocab_nll_loss(self.hazards, self.event_rows, self.event_questions, self.event_times, self.censor, BINS)
        boosted = self.hazards.clone()
        boosted[0, 0, 1] += 2  # row 0, question 1 has its event at 0.5 days, in piece 0
        better = open_vocab_nll_loss(boosted, self.event_rows, self.event_questions, self.event_times, self.censor, BINS)
        self.assertLess(float(better), float(base))

    def test_the_gradient_reaches_the_head_and_the_question_embeddings(self):
        head = OpenVocabularyHead(8, 5, 8, PIECES)
        for p in head.parameters():
            torch.nn.init.normal_(p, std=0.1)
        questions = torch.randn(self.questions, 5, requires_grad=True)
        hazards = head(torch.randn(self.rows, 8), questions)
        loss = open_vocab_nll_loss(hazards, self.event_rows, self.event_questions, self.event_times.float(), self.censor.float(), BINS)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters() if p.requires_grad))
        self.assertGreater(float(questions.grad.abs().sum()), 0)

    def test_wrong_number_of_bin_edges_fails(self):
        with self.assertRaisesRegex(ValueError, "bin edges"):
            open_vocab_nll_loss(self.hazards, self.event_rows, self.event_questions, self.event_times, self.censor, BINS[:-1])

    def test_training_on_a_known_rate_recovers_it(self):
        # one question that happens at a constant rate of 0.01 per day, observed for 100 days
        torch.manual_seed(3)
        rate, n = 0.01, 4000
        times = torch.distributions.Exponential(rate).sample((n,)).double()
        censor = torch.full((n,), 100.0, dtype=torch.float64)
        happened = times < censor
        rows = torch.arange(n)[happened]
        log2_rate = torch.nn.Parameter(torch.tensor(-3.0, dtype=torch.float64))
        optimiser = torch.optim.Adam([log2_rate], lr=0.05)
        for _ in range(300):
            optimiser.zero_grad()
            hazards = log2_rate.expand(n, PIECES, 1)
            open_vocab_nll_loss(hazards, rows, torch.zeros_like(rows), times[happened], censor, BINS).backward()
            optimiser.step()
        self.assertAlmostEqual(2 ** float(log2_rate), rate, delta=0.002)


class ProbabilityTest(unittest.TestCase):
    def test_window_probabilities(self):
        hazards = torch.full((2, PIECES, 3), math.log2(0.01))  # 0.01 events per day for every question
        windows = [(0.0, 30.0), (0.0, 365.0), (30.0, 365.0)]
        probs = window_probabilities(hazards, BINS, windows)
        self.assertEqual(probs.shape, (2, 3, 3))
        torch.testing.assert_close(probs[0, 0, 0], torch.tensor(1 - math.exp(-0.3)), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(probs[0, 1, 0], torch.tensor(1 - math.exp(-3.65)), rtol=1e-5, atol=1e-6)
        # the chance over the whole year is the chance over the first month plus what remains
        first, rest, whole = probs[0, 0, 0], probs[0, 2, 0], probs[0, 1, 0]
        torch.testing.assert_close(whole, first + (1 - first) * (rest / (1 - first)), rtol=1e-5, atol=1e-6)
        self.assertTrue(((probs >= 0) & (probs <= 1)).all())

    def test_a_longer_window_is_never_less_likely(self):
        torch.manual_seed(2)
        hazards = torch.randn(4, PIECES, 5) - 7
        probs = window_probabilities(hazards, BINS, [(0.0, 7.0), (0.0, 30.0), (0.0, 365.0)])
        self.assertTrue((probs[:, 1] >= probs[:, 0] - 1e-7).all())
        self.assertTrue((probs[:, 2] >= probs[:, 1] - 1e-7).all())


if __name__ == "__main__":
    unittest.main()
