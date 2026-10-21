"""Analytic FLOP accounting for the (looped) DiT.

Counts multiply-accumulates (MACs) per sequence for `models/dit.py` and converts
to FLOPs (2 FLOPs per MAC, backward = 2x forward). Two conventions:

  chinchilla  total parameters: includes the input-embedding and logits matmuls
  kaplan      non-embedding parameters: excludes both vocabulary terms

A looped model with slice width n_m executes depth + (S - 1) * n_m blocks.
"""

from __future__ import annotations

from dataclasses import dataclass

MAC_TO_FLOP = 2
BACKWARD_MULTIPLIER = 2


@dataclass(frozen=True)
class Arch:
  d_model: int = 768
  n_heads: int = 12
  depth: int = 12
  mlp_ratio: int = 4
  cond_dim: int = 128
  vocab_size: int = 50258
  seq_len: int = 1024
  freq_embed: int = 256
  loop_enabled: bool = False
  loop_num_layers: int = 2


def from_config(config, vocab_size: int) -> Arch:
  loop = config.model.get('loop', None)
  enabled = bool(loop is not None and loop.enabled)
  return Arch(
    d_model=int(config.model.hidden_size),
    n_heads=int(config.model.n_heads),
    depth=int(config.model.n_blocks),
    cond_dim=int(config.model.cond_dim),
    vocab_size=int(vocab_size),
    seq_len=int(config.model.length),
    loop_enabled=enabled,
    loop_num_layers=int(loop.num_layers) if enabled else 0)


def executed_blocks(arch: Arch, loop_count: float) -> float:
  if not arch.loop_enabled:
    return float(arch.depth)
  return float(arch.depth + (loop_count - 1) * arch.loop_num_layers)


def block_macs_per_seq(arch: Arch) -> int:
  d, L = arch.d_model, arch.seq_len
  linear = (4 + 2 * arch.mlp_ratio) * d * d * L  # qkv, out, mlp
  attention = 2 * L * L * d                       # QK^T and AV
  adaln = 6 * d * arch.cond_dim
  return linear + attention + adaln


def forward_macs_per_seq(arch: Arch, loop_count: float,
                         convention: str = 'chinchilla') -> float:
  if convention not in ('chinchilla', 'kaplan'):
    raise ValueError(f'unknown convention {convention!r}')
  fixed = (arch.freq_embed * arch.cond_dim + arch.cond_dim * arch.cond_dim
           + 2 * arch.d_model * arch.cond_dim)
  if convention == 'chinchilla':
    fixed += 2 * arch.seq_len * arch.vocab_size * arch.d_model
  return executed_blocks(arch, loop_count) * block_macs_per_seq(arch) + fixed


def forward_flops_per_seq(arch: Arch, loop_count: float,
                          convention: str = 'chinchilla') -> float:
  return MAC_TO_FLOP * forward_macs_per_seq(arch, loop_count, convention)


def training_flops_per_seq(arch: Arch, loop_count: float,
                           convention: str = 'chinchilla') -> float:
  return (1 + BACKWARD_MULTIPLIER) * forward_flops_per_seq(
    arch, loop_count, convention)


def flop_multiplier(arch: Arch, mean_loop_count: float,
                    convention: str = 'chinchilla') -> float:
  """Per-step training cost relative to the non-looped model (r = F_loop / F_base)."""
  base = Arch(**{**arch.__dict__, 'loop_enabled': False})
  return (training_flops_per_seq(arch, mean_loop_count, convention)
          / training_flops_per_seq(base, 1, convention))


def iso_flop_steps(arch: Arch, mean_loop_count: float, baseline_steps: int,
                   convention: str = 'chinchilla') -> int:
  """Looped-run step budget matching `baseline_steps` of the non-looped model."""
  return int(round(baseline_steps
                   / flop_multiplier(arch, mean_loop_count, convention)))
