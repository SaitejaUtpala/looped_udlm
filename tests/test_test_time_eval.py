import lightning as L
import omegaconf
import pytest
import torch

import algo
from conftest import TinyTokenizer, loader, make_config, make_model


def _validate(model):
  trainer = L.Trainer(accelerator='cpu', devices=1, logger=False,
                      enable_progress_bar=False, enable_model_summary=False,
                      limit_val_batches=2)
  return trainer.validate(model, loader(n=8), verbose=False)[0]


def test_eval_depth_resolution():
  model = make_model('model.loop.eval_count_sweep=[1,2]').eval()
  assert model._resolve_loop(None) == (4, None)
  assert model._resolve_loop(7) == (7, None)
  model._eval_count_override = 2
  assert model._resolve_loop(None) == (2, None)
  model._eval_count_override = None
  model.loop_eval_policy = 'adaptive'
  assert model._resolve_loop(None) == (4, 0.1)
  assert model._resolve_loop(3) == (3, None)  # explicit depth stays fixed
  model.train()
  s, eps = model._resolve_loop(None)
  assert eps is None and 1 <= s <= 12


def test_validation_depth_sweep_logs_every_depth():
  model = make_model('model.loop.eval_count_sweep=[1,2,4]')
  metrics = _validate(model)
  for s in (1, 2, 4):
    assert f'val/nll@{s}' in metrics and f'val/ppl@{s}' in metrics
  # eval_count=4 is also the primary depth.
  assert metrics['val/nll@4'] == pytest.approx(metrics['val/nll'])
  assert metrics['val/nll@1'] != pytest.approx(metrics['val/nll@4'])
  assert 'val/loop_mean_depth' not in metrics


def test_adaptive_validation_reports_telemetry():
  model = make_model('model.loop.eval_policy=adaptive',
                     'model.loop.eval_count=12',
                     'model.loop.adaptive_epsilon=0.05')
  metrics = _validate(model)
  tel = model.validation_loop_telemetry
  assert tel['samples'] == 8 and tel['calls'] == 2
  assert sum(tel['depth_histogram'].values()) == 8
  assert 1 <= tel['mean_depth'] <= 12
  assert 0 <= tel['compaction_saving'] <= tel['saving_vs_max'] <= 1
  assert metrics['val/loop_mean_depth'] == pytest.approx(tel['mean_depth'])
  assert torch.isfinite(torch.tensor(metrics['val/nll']))


def test_adaptive_generation_telemetry_counts_every_denoiser_call():
  model = make_model('model.loop.eval_policy=adaptive',
                     'model.loop.eval_count=6',
                     'sampling.steps=4',
                     'sampling.noise_removal=greedy').eval()
  model.reset_loop_telemetry()
  with torch.no_grad():
    samples = model.generate_samples(num_samples=3, num_steps=4)
  assert samples.shape == (3, model.num_tokens)
  tel = model.loop_telemetry()
  assert tel['samples'] == 3 * tel['calls'] and tel['calls'] >= 4
  assert tel['max_count'] == 6 and 1 <= tel['mean_depth'] <= 6


def test_fixed_depth_generation_records_no_adaptive_telemetry():
  model = make_model('sampling.steps=4').eval()
  model.reset_loop_telemetry()
  with torch.no_grad():
    model.generate_samples(num_samples=2, num_steps=4)
  assert model.loop_telemetry() is None


def test_legacy_checkpoint_loop_config_is_supported():
  config = make_config()
  omegaconf.OmegaConf.set_struct(config, False)
  loop = config.model.loop
  del loop['schedule'], loop['eval_policy'], loop['adaptive_epsilon']
  loop.sample_mode = 'banded_uniform'
  model = algo.DUO_BASE(config, TinyTokenizer())
  assert model.loop_schedule == 'noise_proportional'
  assert model.loop_eval_policy == 'fixed'
  assert model.loop_adaptive_epsilon == 0.1
