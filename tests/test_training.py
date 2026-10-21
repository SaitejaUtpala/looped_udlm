import lightning as L
import pytest
import torch

import flops
import loop_schedule as ls
from conftest import loader, make_model


class RecordLoopCounts(L.Callback):
  def __init__(self):
    self.records = []  # (global_step, accum_step, loop_count)

  def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
    self.records.append((module.global_step, module._cur_accum_step,
                         module._last_loop_count))


def _fit(model, steps=4, **trainer_kwargs):
  recorder = RecordLoopCounts()
  trainer = L.Trainer(
    accelerator='cpu', devices=1, max_steps=steps, logger=False,
    enable_checkpointing=False, enable_progress_bar=False,
    enable_model_summary=False, num_sanity_val_steps=0,
    accumulate_grad_batches=model.config.trainer.accumulate_grad_batches,
    limit_val_batches=0, callbacks=[recorder], **trainer_kwargs)
  model.on_train_start = lambda: None  # keep the plain test dataloader
  trainer.fit(model, loader(n=64))
  return trainer, recorder.records


def test_accumulation_is_two_micro_batches():
  model = make_model()
  assert model.config.trainer.accumulate_grad_batches == 2


@pytest.mark.parametrize('schedule', ['uniform', 'flop_proportional'])
def test_step_level_schedules_draw_one_depth_per_step(schedule):
  model = make_model(schedule=schedule)
  before = [p.detach().clone() for p in model.backbone.parameters()]
  trainer, records = _fit(model)
  vals, probs = model.loop_pmf
  depths = [s for _, _, s in records]
  assert len(depths) == 8  # 4 optimizer steps x 2 micro-batches
  expected = [ls.sample_from_pmf(vals, probs, ls.step_seed(1, g))
              for g in range(4)]
  # Both micro-batches of an optimizer step share its global_step-seeded draw.
  assert depths[0::2] == expected and depths[1::2] == expected
  after = list(model.backbone.parameters())
  assert any(not torch.equal(a, b) for a, b in zip(before, after))
  assert torch.isfinite(trainer.callback_metrics['trainer/loss'])


def test_noise_proportional_assigns_bands_to_accumulation_steps():
  # One device x 2 accumulation steps -> 2 bands: {1..6} (low noise), {7..12}.
  model = make_model(schedule='noise_proportional')
  _, records = _fit(model, steps=6)
  for _, accum, s in records:
    assert (1 <= s <= 6) if accum == 0 else (7 <= s <= 12)


def test_training_flops_telemetry_accumulates_and_checkpoints(tmp_path):
  model = make_model()
  trainer, records = _fit(model)
  expected = sum(4 * flops.training_flops_per_seq(model.flop_arch, s)
                 for _, _, s in records)
  assert model.train_flops == pytest.approx(expected)
  assert trainer.callback_metrics['trainer/train_flops'] == pytest.approx(
    expected)
  assert 1 <= trainer.callback_metrics['trainer/loop_count'] <= 12

  path = tmp_path / 'model.ckpt'
  trainer.save_checkpoint(path)
  state = torch.load(path, weights_only=False)
  assert state['train_flops'] == pytest.approx(expected)
  fresh = make_model()
  fresh.on_load_checkpoint(state)
  assert fresh.train_flops == pytest.approx(expected)


def test_baseline_trains_without_loop_telemetry():
  model = make_model(loop=False)
  trainer, records = _fit(model)
  assert all(s is None for _, _, s in records)
  assert 'trainer/loop_count' not in trainer.callback_metrics
