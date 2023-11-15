import torch
import torch.nn as nn

from model.mlp import MLP
from model.self_attention import SelfAttention

class TransformerEncoderResidualLayer(nn.Module):
  def __init__(self, d, num_heads, dim_ff):
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

  def forward(self, x, mask_pad_src=None):
    x1 = self.self_attn(
      x=self.self_attn_layer_norm(x),
      mask_pad=mask_pad_src,
      add_mask_attn=False,
    )
    x2 = self.mlp(self.final_layer_norm(x + x1))
    x = x1 + x2
    return x






















