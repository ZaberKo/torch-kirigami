"""Independent covariance, MP, reconstruction and public structural lifecycle checks.

These unittest tests also run under pytest. No pretrained downloads or dataset
packages are required; CUDA variants execute when a GPU is available.
"""

import copy
import math
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import (
    ActivationCovariance,
    DenoisedVariance,
    Greedy,
    ParameterBudget,
    Pruner,
    apply_ob2c_updates,
    collect_covariances,
    denoise_covariance,
    load_checkpoint,
    mp_median,
    prepare_ob2c_updates,
    save_checkpoint,
    solve_ob2c,
)


def test_mp_unit_aspect_against_analytic_cdf():
    """Check the continuous-bulk median against the closed-form q=1 CDF."""
    median = mp_median(1.0)
    theta = math.acos(1 - median / 2)
    assert abs((theta + math.sin(theta)) / math.pi - 0.5) < 1e-6


class DVBPTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.old_threads)

    def test_streaming_covariance_matches_centered_reference(self):
        x = torch.randn(7, 3, 5, dtype=torch.float64)
        x[2:] += 100
        reference = x.reshape(-1, 5)
        centered = reference - reference.mean(0)
        expected = centered.T @ centered / len(reference)
        for sizes in ((7,), (1, 2, 4), (3, 4)):
            stats = ActivationCovariance(torch.float64, chunk_size=4)
            offset = 0
            for size in sizes:
                stats.update(x[offset : offset + size])
                offset += size
            self.assertEqual(stats.images, 7)
            self.assertEqual(stats.count, 21)
            torch.testing.assert_close(stats.mean, reference.mean(0))
            torch.testing.assert_close(stats.covariance(), expected)

    def test_mp_median_matches_independent_analytic_unit_aspect_cdf(self):
        median = mp_median(1.0)
        # For q=1, x=2-2cos(theta), CDF=(theta+sin(theta))/pi.
        theta = math.acos(1 - median / 2)
        self.assertAlmostEqual((theta + math.sin(theta)) / math.pi, 0.5, places=6)
        self.assertAlmostEqual(mp_median(4), 4 * mp_median(0.25), places=12)
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                mp_median(value)

    def test_spectral_scores_match_independent_spiked_diagonal_example(self):
        stats = ActivationCovariance(torch.float64)
        stats.count = stats.images = 400
        stats.mean = torch.zeros(4, dtype=torch.float64)
        stats.m2 = torch.diag(torch.tensor([1.0, 1.0, 1.0, 100.0], dtype=torch.float64)) * 400
        result = denoise_covariance(stats, 0.4, damping=0.1)
        self.assertEqual(result.signal_components, 1)
        torch.testing.assert_close(
            result.denoised_scores, torch.tensor([0.0, 0.0, 0.0, 100.0], dtype=torch.float64)
        )
        torch.testing.assert_close(
            result.scores, torch.tensor([0.6, 0.6, 0.6, 100.0], dtype=torch.float64)
        )
        self.assertAlmostEqual(result.ridge, 2.5)
        raw = denoise_covariance(stats, 0.4, repair_covariance="raw")
        self.assertEqual(raw.repair_covariance, "raw")
        torch.testing.assert_close(raw.raw_scores, result.raw_scores)

    def test_constant_activations_have_zero_scores_and_finite_regularized_system(self):
        stats = ActivationCovariance(torch.float64)
        stats.update(torch.full((4, 2, 6), 5.0, dtype=torch.float64))
        result = denoise_covariance(stats, 0.5)
        torch.testing.assert_close(result.scores, torch.zeros(6, dtype=torch.float64))
        self.assertGreater(result.ridge, 0)
        self.assertEqual(result.images, 4)  # Not eight token observations.
        torch.linalg.cholesky(result.system)

    def test_ob2c_matches_independent_augmented_least_squares(self):
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for device in devices:
            with self.subTest(device=device):
                x = torch.randn(40, 6, dtype=torch.float64, device=device) + 3
                w = torch.randn(3, 6, dtype=torch.float64, device=device)
                b = torch.randn(3, dtype=torch.float64, device=device)
                mu = x.mean(0)
                centered = x - mu
                h = centered.T @ centered / len(x)
                removed, kept = [1, 4], [0, 2, 3, 5]
                updated, bias = solve_ob2c(w, b, h, mu, removed)
                design = torch.cat(
                    (x[:, kept], torch.ones(len(x), 1, dtype=x.dtype, device=device)), 1
                )
                target = x @ w.T + b
                independent = torch.linalg.lstsq(design, target).solution
                torch.testing.assert_close(updated.T, independent[:-1], atol=1e-9, rtol=1e-9)
                torch.testing.assert_close(bias, independent[-1], atol=1e-9, rtol=1e-9)
                torch.testing.assert_close(updated @ mu[kept] + bias, w @ mu + b)

    def test_ob2c_regularization_matches_augmented_ridge_reference(self):
        x = torch.randn(30, 5, dtype=torch.float64)
        w, b = torch.randn(2, 5, dtype=torch.float64), torch.randn(2, dtype=torch.float64)
        mu = x.mean(0)
        centered = x - mu
        ridge = 0.2
        h = centered.T @ centered / len(x) + ridge * torch.eye(5, dtype=torch.float64)
        kept = [0, 2, 3, 4]
        updated, bias = solve_ob2c(w, b, h, mu, [1])
        design = torch.cat(
            (
                centered[:, kept] / math.sqrt(len(x)),
                math.sqrt(ridge) * torch.eye(4, dtype=torch.float64),
            )
        )
        targets = torch.cat((centered @ w.T / math.sqrt(len(x)), math.sqrt(ridge) * w[:, kept].T))
        reference = torch.linalg.lstsq(design, targets).solution
        torch.testing.assert_close(updated.T, reference)
        torch.testing.assert_close(bias, b + w @ mu - updated @ mu[kept])

    def test_invalid_or_singular_reconstruction_fails_without_mutation(self):
        w, b = torch.ones(2, 3, dtype=torch.float64), torch.zeros(2, dtype=torch.float64)
        h, mu = torch.zeros(3, 3, dtype=torch.float64), torch.ones(3, dtype=torch.float64)
        for indices in ([0, 0], [3], [0, 1, 2], [1]):
            with self.assertRaises(ValueError):
                solve_ob2c(w, b, h, mu, indices)
        torch.testing.assert_close(w, torch.ones_like(w))
        torch.testing.assert_close(b, torch.zeros_like(b))

    def test_calibration_failure_restores_hooks_modes_and_gradients(self):
        model = nn.Sequential(nn.Linear(3, 6), nn.GELU(), nn.Dropout(0.4), nn.Linear(6, 2)).double()
        model[0].eval()
        modes = tuple(module.training for module in model.modules())
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        gradients = tuple(parameter.grad for parameter in model.parameters())
        x = torch.randn(5, 3, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "batches"):
            collect_covariances(model, ("3",), [(x, torch.zeros(5))], "cpu", 2)
        self.assertEqual(tuple(module.training for module in model.modules()), modes)
        self.assertTrue(all(not module._forward_pre_hooks for module in model.modules()))
        self.assertTrue(
            all(p.grad is old for p, old in zip(model.parameters(), gradients, strict=True))
        )
        x[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            collect_covariances(model, ("3",), [(x, torch.zeros(5))], "cpu", 1)
        self.assertTrue(all(not module._forward_pre_hooks for module in model.modules()))

    def test_public_calibrate_plan_apply_backward_and_checkpoint_lifecycle(self):
        model = nn.Sequential(nn.Linear(3, 6), nn.GELU(), nn.Linear(6, 2)).double().eval()
        dense = copy.deepcopy(model)
        x = torch.randn(15, 3, dtype=torch.float64)
        graph = DependencyGraph.build(model, args=(x[:1],))
        moments = collect_covariances(
            model, ("2",), [(x, torch.zeros(15))], "cpu", 1, dtype=torch.float64
        )
        statistics = {"2": denoise_covariance(moments["2"], 0.2)}
        pairs = {"0": "2"}
        pruner = Pruner(model, graph=graph)
        before_topology = tuple((name, type(module)) for name, module in model.named_modules())
        plan = pruner.plan(
            pruner.discover_candidates(targets=("0",)),
            budget=ParameterBudget.from_ratio(model, 0.1),
            strategy=Greedy(DenoisedVariance(graph, pairs, statistics)),
        )
        removed = tuple(
            plan.analysis.selection(graph.parameter("0.weight")).fully_selected_indices(0)
        )
        self.assertTrue(removed)
        updates = prepare_ob2c_updates(graph, plan, pairs, statistics)
        torch.testing.assert_close(model[2].weight, dense[2].weight)
        pruner.apply(plan)
        apply_ob2c_updates(model, updates)
        self.assertEqual(
            tuple((name, type(module)) for name, module in model.named_modules()), before_topology
        )
        kept = [i for i in range(6) if i not in removed]
        hidden = nn.functional.gelu(dense[0](x))[:, kept]
        expected = nn.functional.linear(hidden, updates["2.weight"], updates["2.bias"])
        torch.testing.assert_close(model(x), expected)
        model(x).sum().backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(model, path)
            restored = load_checkpoint(dense, path, map_location="cpu").eval()
            torch.testing.assert_close(restored(x), model(x))

    def test_unrelated_output_deletion_rejected_before_any_value_update(self):
        model = nn.Sequential(nn.Linear(3, 6), nn.GELU(), nn.Linear(6, 2)).double().eval()
        x = torch.randn(5, 3, dtype=torch.float64)
        graph = DependencyGraph.build(model, args=(x[:1],))
        moments = collect_covariances(
            model, ("2",), [(x, torch.zeros(5))], "cpu", dtype=torch.float64
        )
        stats = {"2": denoise_covariance(moments["2"], 0.2)}
        plan = Pruner(model, graph=graph, preserve_io=False).plan_remove(
            [graph.parameter("2.weight").axis(0).select([0])]
        )
        before = copy.deepcopy(model.state_dict())
        with self.assertRaisesRegex(ValueError, "matched hidden-width"):
            prepare_ob2c_updates(graph, plan, {"0": "2"}, stats)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name])

    def test_apply_values_prevalidates_every_update(self):
        model = nn.Linear(3, 2)
        old = model.weight.detach().clone()
        with self.assertRaisesRegex(ValueError, "Invalid compact"):
            apply_ob2c_updates(model, {"weight": torch.zeros_like(old), "bias": torch.ones(3)})
        torch.testing.assert_close(model.weight, old)


if __name__ == "__main__":
    unittest.main()
