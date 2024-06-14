import torch.nn as nn

from .feed_forward_network import FeedForward
from .multi_head_attention import MultiHeadAttention

class Block(nn.Module):
  def __init__(
    self, model_dimension, num_heads, context_window, dropout, **kwargs
  ):
    super().__init__()
    head_size = model_dimension // num_heads
    self.sa = MultiHeadAttention(
      num_heads, head_size, model_dimension, context_window, dropout
    )
    self.ffwd = FeedForward(model_dimension, dropout)
    self.ln1 = nn.LayerNorm(model_dimension)
    self.ln2 = nn.LayerNorm(model_dimension)

    #self.dropout1 = LPDropout(dropout)
    #self.dropout2 = LPDropout(dropout)

  def forward(self, x, dt:float=1.0):
    x = x + dt * self.sa(self.ln1(x))
    x = x + dt * self.ffwd(self.ln2(x))
    return x
