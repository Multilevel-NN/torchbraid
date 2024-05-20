import torch.nn as nn

class F_dec_MHA_FF(nn.TransformerDecoderLayer):
  def __init__(self, d_model, nhead, dim_feedforward, dropout, batch_first):
    super().__init__(
      d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
      dropout=dropout, batch_first=batch_first,
    )

  def forward(
    self, x, memory, mem_key_padding_mask,
  ):
    MHA_x = self.mha_block(
      x, mem=memory, attn_mask=None, 
      key_padding_mask=mem_key_padding_mask,
    )
    FF_x = self.ff_block(x + MHA_x)
    
    return MHA_x + FF_x

  def mha_block(self, x, **kwargs): 
    return self._mha_block(self.norm2(x), **kwargs)

  def ff_block(self, x): 
    return self._ff_block(self.norm3(x))




