import torch
import torch.nn as nn

from model.mlp import MLP
from model.multihead_attention import MultiHeadAttention
from model.self_attention import SelfAttention

class TransformerDecoderResidualLayer(nn.Module):
  def __init__(self, d, num_heads, dim_ff):
    super().__init__()
    self.self_attn = SelfAttention(d, num_heads)
    self.cross_attn = MultiHeadAttention(d, num_heads)
    self.mlp = MLP(d, dim_ff)
    self.self_attn_layer_norm = nn.LayerNorm(
      (d,), 
      eps=1e-5, 
      elementwise_affine=True
    )
    self.cross_attn_layer_norm = nn.LayerNorm(
      (d,), 
      eps=1e-5, 
      elementwise_affine=True
    )
    self.final_layer_norm = nn.LayerNorm(
      (d,), 
      eps=1e-5, 
      elementwise_affine=True
    )

  def forward(self, x, memory, mask_pad_tgt=None, mask_pad_mem=None):
    x1 = self.self_attn(
      self.self_attn_layer_norm(x), 
      mask_pad=mask_pad_tgt, 
      add_mask_attn=True,
    )
    x2 = self.cross_attn(
      _K=memory, 
      _V=memory, 
      _Q=self.cross_attn_layer_norm(x + x1),
      mask_attn=None,
      mask_pad=mask_pad_mem,
    )
    x3 = self.mlp(self.final_layer_norm(x + x1 + x2))
    x = x1 + x2 + x3

    return x




















