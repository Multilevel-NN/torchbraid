import torch
import torch.nn as nn

from model.multihead_attention import MultiHeadAttention

class SelfAttention(nn.Module):
  def __init__(self, d, num_heads, max_length, device):
    super().__init__()
    self.attn = MultiHeadAttention(d, num_heads)
    self.causal_attn_mask = nn.Transformer.generate_square_subsequent_mask(max_length).to(device)

  def forward(self, x, mask_pad=None, add_mask_attn=False):  # x: [b, L, d]
    L = x.shape[1]
    mask_attn = self.causal_attn_mask[:L, :L] if add_mask_attn else None

    out = self.attn(
      _K=x, 
      _V=x, 
      _Q=x,
      mask_attn=mask_attn,
      mask_pad=mask_pad,
    )
    return out
