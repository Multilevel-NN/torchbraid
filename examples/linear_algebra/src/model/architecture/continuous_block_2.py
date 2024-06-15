import torch.nn as nn

from ..model_utils.F_dec import F_dec

class ContinuousResidualLayer(nn.Module):
  name = 'decoder'
  state_symbol = 'y'
  
  def __init__(self, model_dimension, num_heads, dim_ff, dropout, **kwargs):
    super().__init__()

    self.F = F_dec(
      d_model=model_dimension, nhead=num_heads, dim_feedforward=dim_ff, 
      dropout=dropout, batch_first=False,
    )

    # self.apply(init_weights)

  def forward(
    self, y, x, tgt_attention_mask, tgt_padding_mask, mem_padding_mask, 
    **kwargs,
  ):  # x: [L , b, d]
      # y: [L', b, d]
    '''variable corresponding to state symbol (y) must be
    the first argument.'''

    y = self.F(                                                   
      x=y, memory=x, tgt_mask=tgt_attention_mask, 
      tgt_key_padding_mask=tgt_padding_mask, 
      mem_key_padding_mask=mem_padding_mask,
    )

    return {'y': y}




