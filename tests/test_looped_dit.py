import copy

import torch

from conftest import make_model, token_batch


def _backbone(**kwargs):
  return make_model(**kwargs).backbone.eval()


def _run(backbone, x, **kwargs):
  sigma = torch.full((x.shape[0],), 0.5)
  with torch.no_grad():
    return backbone(x, sigma, **kwargs)


def test_loop_count_one_equals_plain_stack():
  looped = _backbone()
  x = token_batch()
  torch.testing.assert_close(_run(looped, x, loop_count=1),
                             _run(looped, x, loop_count=None))


def test_disabled_loop_ignores_loop_count():
  plain = _backbone(loop=False)
  x = token_batch()
  torch.testing.assert_close(_run(plain, x), _run(plain, x, loop_count=5))


def test_looping_shares_weights_and_matches_unrolled_stack():
  looped = _backbone()
  n_params = sum(p.numel() for p in looped.parameters())
  x = token_batch()
  out = _run(looped, x, loop_count=3)
  assert sum(p.numel() for p in looped.parameters()) == n_params
  assert not torch.allclose(out, _run(looped, x, loop_count=1))

  # blocks [0, 1, 2, 1, 2, 1, 2, 3, 4, 5] as an explicit, untied stack
  unrolled = copy.deepcopy(looped)
  unrolled.loop_enabled = False
  order = [0] + [1, 2] * 3 + [3, 4, 5]
  unrolled.blocks = torch.nn.ModuleList(
    [copy.deepcopy(looped.blocks[i]) for i in order])
  torch.testing.assert_close(out, _run(unrolled, x))


def test_adaptive_large_epsilon_stops_after_one_traversal():
  looped = _backbone()
  x = token_batch()
  out = _run(looped, x, loop_count=12, adaptive_epsilon=1e9)
  execution = looped.last_loop_execution
  assert execution['policy'] == 'adaptive'
  assert execution['realized'].tolist() == [1] * 4
  assert execution['converged'].all()
  torch.testing.assert_close(out, _run(looped, x, loop_count=1))


def test_adaptive_tiny_epsilon_runs_to_cap():
  looped = _backbone()
  x = token_batch()
  out = _run(looped, x, loop_count=5, adaptive_epsilon=1e-12)
  assert looped.last_loop_execution['realized'].tolist() == [5] * 4
  assert not looped.last_loop_execution['converged'].any()
  torch.testing.assert_close(out, _run(looped, x, loop_count=5))


def _pick_mixed_epsilon(looped, x, cap=12):
  """An epsilon at which samples in the batch stop at different depths."""
  for eps in torch.logspace(-3, 0, 40).tolist():
    _run(looped, x, loop_count=cap, adaptive_epsilon=eps)
    if len(set(looped.last_loop_execution['realized'].tolist())) > 1:
      return eps
  raise AssertionError('no epsilon produced mixed depths')


def test_adaptive_is_per_sample_with_compaction():
  looped = _backbone()
  x = token_batch(n=8, seed=3)
  eps = _pick_mixed_epsilon(looped, x)
  out = _run(looped, x, loop_count=12, adaptive_epsilon=eps)
  execution = looped.last_loop_execution
  realized = execution['realized']
  assert execution['dense_traversals'] < 12 * len(x)
  # Each sample equals running it alone at its realized fixed depth.
  for i in range(len(x)):
    alone = _run(looped, x[i:i + 1], loop_count=int(realized[i]))
    torch.testing.assert_close(out[i:i + 1], alone, rtol=1e-4, atol=1e-4)


def test_adaptive_is_disabled_in_training_mode():
  looped = _backbone().train()
  x = token_batch()
  sigma = torch.full((4,), 0.5)
  looped(x, sigma, loop_count=3, adaptive_epsilon=1e9)
  assert looped.last_loop_execution['policy'] == 'fixed'
  assert looped.last_loop_execution['realized'].tolist() == [3] * 4
