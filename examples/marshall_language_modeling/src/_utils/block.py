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

    #self.fc1 = nn.Linear(model_dimension, model_dimension, bias=False)

    #self.dropout1 = LPDropout(dropout)
    #self.dropout2 = LPDropout(dropout)

  def forward(self, x, dt:float=1.0):
    #x1 = self.ln1(x)
    #x1 = x + dt * self.mha(
    #        x1, x1, x1, mask
    #)

    #x2 = self.ln2(x1)
    #x2 = x1 + dt * self.feed_forward(x2)
    #return x1


    # Original
    x = x + dt * self.sa(self.ln1(x))
    x = x + dt * self.ffwd(self.ln2(x))
    #x = x + dt * self.sa(self.ln2(x))
    #x = x + dt * self.fc1(self.ln2(x))
    #x = x + dt * self.ffwd(x)
    return x
