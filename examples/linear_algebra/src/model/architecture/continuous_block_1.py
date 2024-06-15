import torch.nn as nn

from ..model_utils.F_enc import F_enc

class ContinuousResidualLayer(nn.Module):
  name = 'encoder'
  state_symbol = 'x'
  
  def __init__(self, model_dimension, num_heads, dim_ff, dropout, **kwargs):
    super().__init__()

    self.F = F_enc(
      d_model=model_dimension, nhead=num_heads, dim_feedforward=dim_ff, 
      dropout=dropout, batch_first=False,
    )

    # self.apply(init_weights)

  def forward(self, x, src_padding_mask, **kwargs):  # x: [L, b, d]
    x = self.F(
      x=x,
      src_mask=None,
      src_key_padding_mask=src_padding_mask,
    )

    return {'x': x}




