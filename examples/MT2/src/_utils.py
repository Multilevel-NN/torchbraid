import time
import torch
import torch.nn as nn

from autoinit import AutoInitializer

inf = float('inf')

class Ctr:
  def __init__(self, *things_to_count):
    for thing_to_count in things_to_count: setattr(self, thing_to_count, 0)

@AutoInitializer
class Monitor:
  def is_time(self): pass
  def start  (self): pass
  def step   (self): pass

@AutoInitializer
class Chronometer(Monitor):
  def __init__(self, period): self.t0 = -inf
  def __repr__(self): return f'Chronometer w/ period {self.period} minutes'

  def is_time(self):
    t1 = time.time()
    
    if t1 - self.t0 > self.period * 60: 
      print(f'Time! {self.t0}, {t1}')
      self.t0 = t1
      return True

    else: return False

  def start(self): self.t0 = time.time(); print(f'Initial time: {self.t0}')

class FakeObject:
  def __getattribute__(self, x): return lambda *args, **kwargs: None

@AutoInitializer
class Loop(Monitor):
  def __init__(self, frequency): self.ctr = -inf
  def __repr__(self): return f'Loop w/ frequency {self.frequency} batches'
  def is_time (self): return self.ctr % self.frequency == 0
  def start(self): self.ctr = 0
  def step(self): self.ctr += 1

@AutoInitializer
class LabelSmoothingDistribution(nn.Module):
  def __init__(self, smoothing_value, pad_token_id, trg_vocab_size, device):
    assert 0.0 <= smoothing_value <= 1.0
    super().__init__()
    self.confidence_value = 1.0 - smoothing_value

  def forward(self, trg_token_ids_batch):
    batch_size = trg_token_ids_batch.shape[0]
    smooth_target_distributions = torch.zeros(
      (batch_size, self.trg_vocab_size), device=self.device,
    )
    smooth_target_distributions.fill_(
      self.smoothing_value / (self.trg_vocab_size - 2)
    )
    smooth_target_distributions.scatter_(
      1, trg_token_ids_batch, self.confidence_value,
    )
    # smooth_target_distributions[:, self.pad_token_id] = 0.
    smooth_target_distributions.masked_fill_(
      trg_token_ids_batch == self.pad_token_id, 0.,
    )
    return smooth_target_distributions

@AutoInitializer
class Timeit:
  def __init__(self, function): pass
  def __call__(self, *args, **kwargs):
    t0 = time.time()
    rv = self.function(*args, **kwargs)
    t1 = time.time()
    print(f'Running time of function {self.function.__name__}: '
          f'{t1 - t0} seconds\n')
    return rv



