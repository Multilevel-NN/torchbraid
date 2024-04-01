import torch
import torch.nn as nn

from model.mlp import MLP
from model.self_attention import SelfAttention

class TransformerEncoderLayer(nn.Module):
  def __init__(self, d, num_heads, dim_ff, norm_first):
    super().__init__()
    self.self_attn = SelfAttention(d, num_heads)
    self.mlp = MLP(d, dim_ff)
    self.self_attn_layer_norm = nn.LayerNorm(
      (d,), 
      eps=1e-5, 
      elementwise_affine=True
    )
    self.final_layer_norm = nn.LayerNorm(
      (d,), 
      eps=1e-5, 
      elementwise_affine=True
    )
    self.norm_first = norm_first

  def forward(self, x, mask_pad_src=None):
    if self.norm_first:
      x = x + self.self_attn(
          x=self.self_attn_layer_norm(x),
          mask_pad=mask_pad_src,
          add_mask_attn=False,
        )
      x = x + self.mlp(self.final_layer_norm(x))

    else: 
      x = self.self_attn_layer_norm(
        x + self.self_attn(
          x=x,
          mask_pad=mask_pad_src,
          add_mask_attn=False,
        )
      )
      x = self.final_layer_norm(x + self.mlp(x))

    return x






















