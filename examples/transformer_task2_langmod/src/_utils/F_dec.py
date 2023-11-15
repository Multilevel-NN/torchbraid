import torch.nn as nn

from .block import Block

class F_dec(Block):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def forward(self, x):
    SA_x = self.sa_block(x)
    FF_x = self.ff_block(x + SA_x)
    return SA_x + FF_x

  def sa_block(self, x): return self.sa(self.ln1(x))
  def ff_block(self, x): return self.ffwd(self.ln2(x))



