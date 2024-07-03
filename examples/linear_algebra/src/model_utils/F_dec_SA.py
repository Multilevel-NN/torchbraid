import time
import torch.nn as nn

class F_dec_SA(nn.TransformerDecoderLayer):
  def __init__(self, d_model, nhead, dropout, batch_first):
    super().__init__(
      d_model=d_model, nhead=nhead, dropout=dropout, batch_first=batch_first,
    )

  def forward(
    self, x, tgt_mask, tgt_key_padding_mask,
  ):
    t0 = time.time()
    SA_x = self.sa_block(
      x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask,
    )
    t1 = time.time()
    if 0: print(f'DEC: SA-time={t1-t0:.4f}')
    
    return SA_x

  def sa_block(self, x, **kwargs): 
    return self._sa_block(self.norm1(x), **kwargs)



