"""Training-time loop-count schedules.

  uniform              S ~ U{S_min, ..., S_max}, one draw per optimizer step.
  flop_proportional    S ~ p(S) proportional to F(S)^(-beta), one draw per step;
                       beta = 1 gives every depth an equal share of training FLOPs.
  noise_proportional   Deeper loops on noisier inputs. With antithetic time
                       sampling each (rank, accumulation) micro-batch owns a
                       contiguous band of diffusion time. The M = world_size *
                       grad_accum micro-batches are grouped into B = gcd(M, span)
                       noise bands, and band b draws S uniformly from the b-th
                       contiguous sub-range of {S_min..S_max}. The marginal over S
                       stays uniform, so it matches the uniform schedule's FLOPs.

All draws are seeded by (seed, global_step[, band]) so every DDP rank agrees
without communication and resumption is deterministic.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

import flops

SCHEDULES = ('uniform', 'flop_proportional', 'noise_proportional')


def step_seed(seed: int, global_step: int) -> int:
  return int(seed) * 1_000_003 + int(global_step)


def uniform_pmf(train_min: int, train_max: int):
  vals = list(range(int(train_min), int(train_max) + 1))
  return vals, [1.0 / len(vals)] * len(vals)


def flop_proportional_pmf(train_min: int, train_max: int, arch: flops.Arch,
                          flop_power: float = 1.0):
  """p(S) proportional to F(S)^(-beta) with F the per-sequence forward FLOPs."""
  vals = list(range(int(train_min), int(train_max) + 1))
  weights = [flops.forward_flops_per_seq(arch, s) ** (-float(flop_power))
             for s in vals]
  z = sum(weights)
  return vals, [w / z for w in weights]


def sample_from_pmf(vals: Sequence[int], probs: Sequence[float],
                    seed: int) -> int:
  u = random.Random(int(seed)).random()
  c = 0.0
  for v, p in zip(vals, probs):
    c += p
    if u <= c:
      return int(v)
  return int(vals[-1])


def expected_value(vals: Sequence[int], probs: Sequence[float]) -> float:
  return float(sum(v * p for v, p in zip(vals, probs)))


def num_noise_bands(world_size: int, grad_accum: int, train_min: int,
                    train_max: int, requested: int | None = None) -> int:
  """B must divide both M (equal band probability) and the depth span (equal
  sub-range widths) for the marginal over S to stay exactly uniform."""
  m = max(1, int(world_size)) * max(1, int(grad_accum))
  span = int(train_max) - int(train_min) + 1
  if span <= 1:
    return 1
  if requested is None:
    return math.gcd(m, span)
  b = int(requested)
  if b < 1 or m % b or span % b:
    raise ValueError(f'num_noise_bands={b} must divide M={m} and span={span}')
  return b


def band_subrange(band: int, n_bands: int, train_min: int, train_max: int):
  width = (int(train_max) - int(train_min) + 1) // n_bands
  lo = int(train_min) + band * width
  return lo, lo + width - 1


def noise_proportional_loop_count(*, global_rank: int, accum_step: int,
                                  world_size: int, grad_accum: int,
                                  train_min: int, train_max: int, seed: int,
                                  global_step: int,
                                  requested_bands: int | None = None) -> int:
  """Loop count for one (rank, accumulation) micro-batch.

  Micro-batch index = rank * grad_accum + accum_step follows the order in which
  antithetic time samples are chunked, so index 0 holds the lowest noise.
  """
  m = max(1, int(world_size)) * max(1, int(grad_accum))
  b = num_noise_bands(world_size, grad_accum, train_min, train_max,
                      requested_bands)
  micro = int(global_rank) * max(1, int(grad_accum)) + int(accum_step)
  band = min(b - 1, micro // (m // b))
  lo, hi = band_subrange(band, b, train_min, train_max)
  return random.Random(step_seed(seed, global_step) * 1009 + band).randint(lo, hi)


def schedule_pmf(schedule: str, train_min: int, train_max: int,
                 arch: flops.Arch, flop_power: float = 1.0):
  """Marginal distribution over S (noise_proportional's marginal is uniform)."""
  if schedule in ('uniform', 'noise_proportional'):
    return uniform_pmf(train_min, train_max)
  if schedule == 'flop_proportional':
    return flop_proportional_pmf(train_min, train_max, arch, flop_power)
  raise ValueError(f'unknown schedule {schedule!r}; choose from {SCHEDULES}')
