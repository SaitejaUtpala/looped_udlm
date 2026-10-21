import collections

import pytest

import flops
import loop_schedule as ls

ARCH = flops.Arch(vocab_size=50257, loop_enabled=True, loop_num_layers=2)


def test_uniform_pmf():
  vals, probs = ls.uniform_pmf(1, 12)
  assert vals == list(range(1, 13))
  assert sum(probs) == pytest.approx(1.0)
  assert ls.expected_value(vals, probs) == pytest.approx(6.5)


def test_flop_proportional_pmf_equalizes_flops_per_depth():
  vals, probs = ls.flop_proportional_pmf(1, 12, ARCH)
  assert sum(probs) == pytest.approx(1.0)
  assert all(a > b for a, b in zip(probs, probs[1:]))
  share = [p * flops.forward_flops_per_seq(ARCH, s)
           for s, p in zip(vals, probs)]
  assert max(share) == pytest.approx(min(share))


def test_flop_power_zero_is_uniform():
  _, probs = ls.flop_proportional_pmf(1, 12, ARCH, flop_power=0.0)
  assert probs == pytest.approx(ls.uniform_pmf(1, 12)[1])


def test_paper_iso_flop_step_budgets():
  for schedule, steps in (('uniform', 655_420),
                          ('noise_proportional', 655_420),
                          ('flop_proportional', 688_915)):
    vals, probs = ls.schedule_pmf(schedule, 1, 12, ARCH)
    mean = ls.expected_value(vals, probs)
    assert flops.iso_flop_steps(ARCH, mean, 1_000_000) == steps


def test_executed_depth_and_conventions():
  assert flops.executed_blocks(ARCH, 1) == 12
  assert flops.executed_blocks(ARCH, 12) == 34
  base = flops.Arch(vocab_size=50257)
  assert flops.executed_blocks(base, 12) == 12
  assert (flops.forward_flops_per_seq(ARCH, 6, 'kaplan')
          < flops.forward_flops_per_seq(ARCH, 6, 'chinchilla'))
  assert flops.flop_multiplier(ARCH, 1) == pytest.approx(1.0)


def test_sampling_is_deterministic_and_matches_pmf():
  vals, probs = ls.flop_proportional_pmf(1, 12, ARCH)
  seeds = [ls.step_seed(1, step) for step in range(20_000)]
  draws = [ls.sample_from_pmf(vals, probs, s) for s in seeds]
  assert draws == [ls.sample_from_pmf(vals, probs, s) for s in seeds]
  freq = collections.Counter(draws)
  for v, p in zip(vals, probs):
    assert freq[v] / len(draws) == pytest.approx(p, abs=0.01)


def _noise_prop(rank, accum, step, world_size=4, grad_accum=4, bands=None):
  return ls.noise_proportional_loop_count(
    global_rank=rank, accum_step=accum, world_size=world_size,
    grad_accum=grad_accum, train_min=1, train_max=12, seed=1,
    global_step=step, requested_bands=bands)


def test_noise_proportional_couples_depth_to_noise_band():
  # 4 GPUs x 4 accumulation steps -> M=16 micro-batches, B=gcd(16,12)=4 bands.
  assert ls.num_noise_bands(4, 4, 1, 12) == 4
  for step in range(50):
    for micro in range(16):
      rank, accum = divmod(micro, 4)
      lo, hi = ls.band_subrange(micro // 4, 4, 1, 12)
      assert lo <= _noise_prop(rank, accum, step) <= hi


def test_noise_proportional_marginal_is_uniform():
  counts = collections.Counter(
    _noise_prop(rank, accum, step)
    for step in range(3000) for rank in range(4) for accum in range(4))
  total = sum(counts.values())
  for s in range(1, 13):
    assert counts[s] / total == pytest.approx(1 / 12, abs=0.01)


def test_noise_bands_must_divide_micro_batches_and_span():
  with pytest.raises(ValueError):
    ls.num_noise_bands(4, 4, 1, 12, requested=3)
  assert ls.num_noise_bands(4, 4, 1, 12, requested=2) == 2
  assert ls.num_noise_bands(1, 1, 1, 12) == 1
