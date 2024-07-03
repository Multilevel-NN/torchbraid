import time
import torch.nn as nn

class F_enc(nn.TransformerEncoderLayer):
  def __init__(self, d_model, nhead, dim_feedforward, dropout, batch_first):
    super().__init__(
      d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
      dropout=dropout, batch_first=batch_first,
    )

  def forward(self, x, src_mask, src_key_padding_mask):
    t0 = time.time()
    SA_x = self.sa_block(
      x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask,
    )
    t1 = time.time()
    FF_x = self.ff_block(x + SA_x)
    t2 = time.time()

    if 0: print(f'ENC: SA_time={t1-t0:.4f}, FF_time={t2-t1:.4f}')

    return SA_x + FF_x

  def sa_block(self, x, **kwargs): 
    return self._sa_block(self.norm1(x), **kwargs)

  def ff_block(self, x):
    return self._ff_block(self.norm2(x))




