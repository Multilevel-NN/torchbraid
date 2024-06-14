import torch
import torch.nn as nn
from torchbraid.utils import LPDropout

from .head import Head

class MultiHeadAttention(nn.Module):
  def __init__(
    self, num_heads, head_size, model_dimension, context_window, dropout
  ):
    super().__init__()
    self.heads = nn.ModuleList([
      Head(head_size, model_dimension, context_window, dropout) \
      for _ in range(num_heads)
    ])
    self.proj = nn.Linear(head_size * num_heads, model_dimension)
    self.dropout = LPDropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    return out
