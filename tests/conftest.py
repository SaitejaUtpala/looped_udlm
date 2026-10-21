"""Tiny CPU fixtures: a 6-block DiT (looped slice = blocks 1-2) on a 64-token vocab."""

import os
import sys

import hydra
import omegaconf
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import algo  # noqa: E402
import metrics  # noqa: E402

for name, fn in (('cwd', os.getcwd), ('device_count', lambda: 1),
                 ('eval', eval), ('div_up', lambda x, y: (x + y - 1) // y)):
  if not omegaconf.OmegaConf.has_resolver(name):
    omegaconf.OmegaConf.register_new_resolver(name, fn)

VOCAB = 64
LENGTH = 16


class TinyTokenizer:
  vocab_size = VOCAB
  mask_token = None
  mask_token_id = None
  pad_token = '<pad>'
  pad_token_id = 0
  bos_token_id = 1
  eos_token_id = 2
  eos_token = '<eos>'

  def __len__(self):
    return VOCAB

  def batch_decode(self, ids, **kwargs):
    return [' '.join(str(int(i)) for i in row) for row in ids]


@pytest.fixture(autouse=True)
def _no_hub_downloads(monkeypatch):
  monkeypatch.setattr(metrics.transformers.AutoTokenizer, 'from_pretrained',
                      lambda *a, **k: TinyTokenizer())


def make_config(*overrides, loop=True, schedule='uniform'):
  base = [
    'algo=duo_base',
    'trainer.devices=1',
    'trainer.accelerator=cpu',
    'trainer.precision=32',
    'loader.global_batch_size=8',
    'loader.batch_size=4',
    'loader.eval_batch_size=4',
    'loader.num_workers=0',
    'model.hidden_size=32',
    'model.cond_dim=16',
    'model.n_heads=2',
    'model.n_blocks=6',
    f'model.length={LENGTH}',
    'model.dropout=0.0',
    'training.ema=0',
    'sampling.num_sample_batches=0',
    'eval.generate_samples=False',
    f'model.loop.enabled={loop}',
    f'model.loop.schedule={schedule}',
    'model.loop.eval_count=4',
    'model.loop.train_max=12',
  ]
  with hydra.initialize_config_dir(os.path.join(REPO, 'configs'),
                                   version_base=None):
    return hydra.compose('config', overrides=base + list(overrides))


def make_model(*overrides, **kwargs):
  torch.manual_seed(0)
  model = algo.DUO_BASE(make_config(*overrides, **kwargs), TinyTokenizer())
  make_nontrivial_(model.backbone)
  return model


def make_nontrivial_(backbone, scale=0.3, seed=0):
  """adaLN-Zero init makes every block an identity; open the gates so the
  output depends on depth."""
  g = torch.Generator().manual_seed(seed)
  with torch.no_grad():
    for p in backbone.parameters():
      p.add_(scale * torch.randn(p.shape, generator=g))
  return backbone


def token_batch(n=4, seed=0):
  g = torch.Generator().manual_seed(seed)
  return torch.randint(3, VOCAB, (n, LENGTH), generator=g)


class TokenDataset(torch.utils.data.Dataset):
  def __init__(self, n=32, seed=0):
    self.ids = token_batch(n, seed)

  def __len__(self):
    return len(self.ids)

  def __getitem__(self, i):
    return {'input_ids': self.ids[i],
            'attention_mask': torch.ones(LENGTH, dtype=torch.long)}


def loader(n=32, batch_size=4, seed=0):
  return torch.utils.data.DataLoader(TokenDataset(n, seed),
                                     batch_size=batch_size)
