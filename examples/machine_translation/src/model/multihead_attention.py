import numpy as np
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
  def __init__(self, d, num_heads):
    super().__init__()

    self.d = d
    self.num_heads = num_heads
    self.dim_keys = self.d // self.num_heads
    self.dim_values = self.dim_keys

    self.k_proj = nn.Linear(
      in_features=self.d,
      out_features=self.num_heads*self.dim_keys,
      bias=True
    )
    self.v_proj = nn.Linear(
      in_features=self.d,
      out_features=self.num_heads*self.dim_values,
      bias=True
    )
    self.q_proj = nn.Linear(
      in_features=self.d,
      out_features=self.num_heads*self.dim_keys,
      bias=True
    )
    self.out_proj = nn.Linear(
      in_features=self.num_heads*self.dim_values,
      out_features=self.d,
      bias=True
    )

  def forward(self, _K, _V, _Q, mask_attn=None, mask_pad=None):  
    '''
        _K, _V: [b, L , d]
            _Q: [b, L', d]
     mask_attn: [L', L]
     mask_pad : [b, L]
    '''
    b, L, d, Lp = *_K.shape, _Q.shape[1]
    nh, dk, dv = self.num_heads, self.dim_keys, self.dim_values

    K = self.k_proj(_K).reshape(b, L , nh, dk).transpose(1, 2)  # K: [b, nh, L , dk]
    V = self.v_proj(_V).reshape(b, L , nh, dv).transpose(1, 2)  # V: [b, nh, L , dv]
    Q = self.q_proj(_Q).reshape(b, Lp, nh, dk).transpose(1, 2)  # Q: [b, nh, L', dk]
    
    KQ = (K @ Q.transpose(-2, -1)) / np.sqrt(dk)  # KQ: [b, nh, L, L']

    if mask_pad is not None:
      mask_pad = mask_pad.reshape(b, 1, L, 1)  # mask_pad: [b, 1, L, 1]
      KQ += mask_pad  # KQ: [b, nh, L, L']

    if mask_attn is not None:  # only in self-attention --> L=L'.
      mask_attn = mask_attn.reshape(1, 1, Lp, L)  # mask_attn: [1, 1, L', L]
      KQ += mask_attn.transpose(-2, -1)  # KQ: [b, nh, L, L']

    α = KQ.softmax(2)  # α: [b, nh, L, L']
    O = (α.transpose(-2, -1) @ V)  # O: [b, nh, L', dv]
    O = O.transpose(1, 2).reshape(b, Lp, nh*dv)  # O: [b, L', nh·dv]
    out = self.out_proj(O)  # out: [b, L', d]

    return out













